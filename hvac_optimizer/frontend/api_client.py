import httpx

BASE_URL = "http://localhost:8000/api/v1"

def handle_response(response):
    if response.status_code == 200:
        return response.json()
    else:
        try:
            error_detail = response.json().get("detail", "Unknown server error")
        except:
            error_detail = response.text
        return {"error": error_detail, "status_code": response.status_code}

def upload_file(site_id, file):
    files = {"file": (file.name, file.getvalue(), file.type)}
    try:
        url = f"{BASE_URL}/sites/{site_id}/data/upload"
        with httpx.Client() as client:
            response = client.post(url, files=files, timeout=60.0)
            return handle_response(response)
    except Exception as e:
        return {"error": f"Connection failed: {str(e)}"}

def get_diagnostics(site_id):
    try:
        url = f"{BASE_URL}/sites/{site_id}/data/diagnostics"
        with httpx.Client() as client:
            response = client.get(url)
            return handle_response(response)
    except Exception as e:
        return {"error": str(e)}

def get_mapping_suggestions(site_id):
    try:
        url = f"{BASE_URL}/sites/{site_id}/data/mapping/suggest"
        with httpx.Client() as client:
            response = client.get(url)
            return handle_response(response)
    except Exception as e:
        return {"error": str(e)}

def save_mapping(site_id, mappings):
    try:
        url = f"{BASE_URL}/sites/{site_id}/data/mapping"
        with httpx.Client() as client:
            response = client.post(url, json={"mappings": mappings})
            return handle_response(response)
    except Exception as e:
        return {"error": str(e)}

def save_equipment(site_id, equipment_data):
    try:
        url = f"{BASE_URL}/sites/{site_id}/data/equipment"
        with httpx.Client(timeout=120.0) as client:
            response = client.post(url, json=equipment_data)
            return handle_response(response)
    except Exception as e:
        return {"error": str(e)}

def get_realtime_metrics(site_id):
    try:
        url = f"{BASE_URL}/sites/{site_id}/analysis/realtime"
        with httpx.Client() as client:
            response = client.get(url)
            return handle_response(response)
    except Exception as e:
        return {"error": str(e)}

def get_projects(completed_only: bool = False):
    """
    List workspace sites. When ``completed_only`` is True, only trained / analysis-ready projects.
    """
    try:
        url = f"{BASE_URL}/sites/list"
        if completed_only:
            url = f"{url}?completed_only=true"
        with httpx.Client() as client:
            response = client.get(url)
            return handle_response(response)
    except Exception as e:
        return {"error": str(e)}


def get_import_history(site_id: str):
    """
    Fetch completed import snapshots for navbar history (labels include data time range).

    Returns API JSON or ``{"error": ...}`` on failure.
    """
    try:
        url = f"{BASE_URL}/sites/{site_id}/import-history"
        with httpx.Client() as client:
            response = client.get(url)
            return handle_response(response)
    except Exception as e:
        return {"error": str(e)}


def activate_import_history(site_id: str, history_id: str):
    """
    Restore workspace (paths, mapping, equipment, ML results) from a history snapshot.

    Returns API JSON or ``{"error": ...}`` on failure.
    """
    try:
        url = f"{BASE_URL}/sites/{site_id}/import-history/activate"
        with httpx.Client() as client:
            response = client.post(url, json={"history_id": history_id})
            return handle_response(response)
    except Exception as e:
        return {"error": str(e)}

def get_analytics(site_id: str):
    """
    Fetch full chart analytics payload for the dashboard.

    Returns the analytics JSON dict or ``{"error": ...}`` on failure.
    """
    try:
        url = f"{BASE_URL}/sites/{site_id}/analysis/analytics"
        with httpx.Client(timeout=30.0) as client:
            response = client.get(url)
            return handle_response(response)
    except Exception as e:
        return {"error": str(e)}


def run_optimization(site_id, params):
    """
    Bound-constrained optimization reads CSV and runs scipy L-BFGS-B; httpx
    default (~5s) is too short and surfaces as "timed out".
    """
    try:
        url = f"{BASE_URL}/sites/{site_id}/analysis/optimize"
        with httpx.Client(timeout=120.0) as client:
            response = client.post(url, json=params)
            return handle_response(response)
    except Exception as e:
        return {"error": str(e)}
