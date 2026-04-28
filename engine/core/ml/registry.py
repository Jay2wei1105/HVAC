# engine/core/ml/registry.py
import joblib
from pathlib import Path
import datetime

class ModelRegistry:
    """本地模型持久化控制器（非 MLflow 雲端版）。"""
    def __init__(self, base_dir: str = "models"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)

    def save_model(self, model, domain: str, target: str, metrics: dict):
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        subdir = self.base_dir / domain / target
        subdir.mkdir(parents=True, exist_ok=True)
        
        path = subdir / f"model_{ts}.joblib"
        joblib.dump({
            "model": model,
            "metrics": metrics,
            "timestamp": ts,
            "target": target
        }, path)
        return path

    def load_latest(self, domain: str, target: str):
        subdir = self.base_dir / domain / target
        if not subdir.exists(): return None
        files = sorted(subdir.glob("*.joblib"))
        if not files: return None
        return joblib.load(files[-1])
