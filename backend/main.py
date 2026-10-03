from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    def load_dotenv(*args, **kwargs):
        return False

from backend.api.routes import initialize_engine, router
from src.utils.logger import setup_logger

load_dotenv()
logger = setup_logger("backend_main")

app = FastAPI(
    title="Smart Ambulance Routing API",
    description="ML-powered ambulance routing with bed availability prediction",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.on_event("startup")
async def startup_event():
    initialize_engine()
    logger.info("Smart Ambulance API startup completed")


@app.get("/")
async def root():
    return {"message": "Smart Ambulance Routing API", "docs": "/docs"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
