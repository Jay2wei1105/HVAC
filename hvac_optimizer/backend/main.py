from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from hvac_optimizer.backend.api.routers import sites, data, analysis

app = FastAPI(
    title="HVAC Optimizer API",
    description="Backend API for independent HVAC optimization tool.",
    version="0.1.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(sites.router, prefix="/api/v1/sites", tags=["sites"])
app.include_router(data.router, prefix="/api/v1/sites/{site_id}/data", tags=["data"])
app.include_router(analysis.router, prefix="/api/v1/sites/{site_id}/analysis", tags=["analysis"])

@app.get("/health")
def health_check():
    return {"status": "ok"}
