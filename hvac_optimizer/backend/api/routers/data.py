import os
import uuid
import json
import pandas as pd
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException
from hvac_optimizer.backend.api.schemas import UploadResponse, MappingSuggestResponse, MappingRequest
from hvac_optimizer.backend.api.store import ACTIVE_DATASETS
from hvac_optimizer.backend.services.data_service import DataService
from hvac_optimizer.backend.services.core_hvac_service import CoreHVACService
from hvac_optimizer.backend.services.history_service import append_completed_import
from hvac_optimizer.backend.services.site_migration import finalize_trained_site

router = APIRouter()

STORAGE_DIR = os.path.join(Path(__file__).resolve().parents[4], "data", "history", "hvac_optimizer_storage")
os.makedirs(STORAGE_DIR, exist_ok=True)
METADATA_FILE = os.path.join(STORAGE_DIR, "projects_metadata.json")

def save_metadata():
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(ACTIVE_DATASETS, f, ensure_ascii=False, indent=2)

def load_metadata():
    if os.path.exists(METADATA_FILE):
        try:
            with open(METADATA_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                ACTIVE_DATASETS.update(data)
        except Exception:
            pass

# Load on module import
load_metadata()

@router.post("/upload", response_model=UploadResponse)
async def upload_csv(site_id: str, file: UploadFile = File(...)):
    if not file.filename.endswith(('.csv', '.xlsx', '.xls')):
        raise HTTPException(status_code=400, detail="Unsupported file format")

    dataset_id = f"ds_{uuid.uuid4().hex[:8]}"
    file_path = os.path.join(STORAGE_DIR, f"{dataset_id}_{file.filename}")
    
    # Save original
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    try:
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)

        prev = ACTIVE_DATASETS.get(site_id, {})
        prev_hist = list(prev.get("import_history") or [])
        ACTIVE_DATASETS[site_id] = {
            "dataset_id": dataset_id,
            "original_path": file_path,
            "cleaned_path": None,
            "cleaned_parquet_path": None,
            "quality_report_path": None,
            "cleaning_stats": {
                "initial_rows": int(len(df)),
                "duplicates_removed": int(len(df) - len(df.drop_duplicates())),
                "outliers_detected": 0,
                "nulls_filled": int(df.isna().sum().sum()),
                "final_rows": int(len(df.drop_duplicates())),
            },
            "columns": df.columns.tolist(),
            "mapping": None,
            "equipment": None,
            "import_history": prev_hist,
            "active_history_id": None,
            "ml_results": None,
            "optimization_results": None,
            "mpc_summary": None,
            "report_summary": None,
            "dashboard_payload": None,
        }
        save_metadata()

        return UploadResponse(
            dataset_id=dataset_id,
            columns=df.columns.tolist(),
            row_count=len(df),
            interval="Auto-detected",
            signature=f"sig_{uuid.uuid4().hex[:4]}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Data processing error: {str(e)}")

@router.get("/diagnostics")
async def get_diagnostics(site_id: str):
    if site_id not in ACTIVE_DATASETS:
        raise HTTPException(status_code=404, detail="No data for this site")
    return ACTIVE_DATASETS[site_id]["cleaning_stats"]

@router.get("/mapping/suggest", response_model=MappingSuggestResponse)
async def suggest_mapping(site_id: str):
    if site_id not in ACTIVE_DATASETS:
        raise HTTPException(status_code=404, detail="Dataset not found")
        
    cols = ACTIVE_DATASETS[site_id]["columns"]
    mappings = DataService.suggest_mappings(cols)

    return MappingSuggestResponse(
        mappings=mappings,
        equipment_suggestion={
            "chiller": [{"id": 1, "rt": 500, "cop": 5.8}, {"id": 2, "rt": 300, "cop": 5.2}],
            "cooling_tower": [{"id": 1, "kw": 18.5}, {"id": 2, "kw": 18.5}]
        }
    )

@router.post("/mapping")
async def confirm_mapping(site_id: str, request: MappingRequest):
    if site_id not in ACTIVE_DATASETS:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    ACTIVE_DATASETS[site_id]["mapping"] = request.mappings
    save_metadata()
    return {"status": "success"}

@router.post("/equipment")
async def confirm_equipment(site_id: str, equipment_data: dict):
    if site_id not in ACTIVE_DATASETS:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    ACTIVE_DATASETS[site_id]["equipment"] = equipment_data
    
    try:
        ds = ACTIVE_DATASETS[site_id]
        stage1 = CoreHVACService.run_stage1_pipeline(
            original_path=ds["original_path"],
            mappings=ds.get("mapping") or [],
            site_id=site_id,
            dataset_id=ds["dataset_id"],
        )
        train_res = CoreHVACService.train_models(
            site_id=site_id,
            dataset_id=ds["dataset_id"],
            cleaned_parquet_path=stage1["cleaned_parquet_path"],
        )
        ds["cleaned_path"] = stage1["cleaned_path"]
        ds["cleaned_parquet_path"] = stage1["cleaned_parquet_path"]
        ds["quality_report_path"] = stage1["quality_report_path"]
        ds["cleaning_stats"] = stage1["quality_report"]
        ACTIVE_DATASETS[site_id]["ml_results"] = train_res
        append_completed_import(ACTIVE_DATASETS[site_id])

        keys_no_draft = set(ACTIVE_DATASETS.keys()) - {site_id}
        final_site_id, project_display_name = finalize_trained_site(site_id, ds, keys_no_draft)
        if final_site_id != site_id:
            ACTIVE_DATASETS[final_site_id] = ds
            del ACTIVE_DATASETS[site_id]
            site_id = final_site_id

        save_metadata()

        return {
            "status": "success",
            "site_id": site_id,
            "project_display_name": project_display_name,
            "stage1": stage1,
            "ml": train_res,
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"ML Training Failed: {str(e)}"
        }
