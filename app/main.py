from fastapi import FastAPI
from app.core.config import settings
from app.core.logger import logger
from app.api.router import router as api_router
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title=settings.PROJECT_NAME, version=settings.VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    logger.info("Starting Structural Drawing AI System API...")

app.include_router(api_router, prefix="/api/v1")

@app.get("/")
def read_root():
    return {"message": "Welcome to Structural Drawing AI System"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}
