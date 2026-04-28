from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class SiteCreate(BaseModel):
    name: str = Field(..., description="案場名稱")
    electricity_rate: float = Field(..., description="電費單價 NT$/kWh")
    contract_capacity: float = Field(..., description="契約容量 kW")
    timezone: Optional[str] = Field("Asia/Taipei", description="時區")
    peak_rates: Optional[bool] = Field(False, description="是否啟用尖離峰費率")

class SiteResponse(SiteCreate):
    site_id: str
    
class UploadResponse(BaseModel):
    dataset_id: str
    columns: List[str]
    row_count: int
    interval: str
    signature: str

class MappingSuggestResponse(BaseModel):
    mappings: List[Dict[str, Any]]
    equipment_suggestion: Dict[str, Any]
    
class MappingRequest(BaseModel):
    mappings: List[Dict[str, Any]]
    equipment: Optional[Dict[str, Any]] = None
    
class RealtimeAnalysisResponse(BaseModel):
    kpi: Dict[str, Any]
    equipment: Dict[str, Any]
    charts: Dict[str, Any]


class ActivateImportHistoryRequest(BaseModel):
    history_id: str = Field(..., description="import_history[].history_id to apply")
