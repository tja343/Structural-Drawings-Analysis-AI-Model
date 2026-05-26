import yaml
from pydantic_settings import BaseSettings
from pathlib import Path

CONFIG_PATH = Path("config/default.yaml")

class Settings(BaseSettings):
    PROJECT_NAME: str = "Structural Drawing AI System"
    VERSION: str = "0.1.0"
    API_V1_STR: str = "/api/v1"
    
    # YOLO Settings
    YOLO_MODEL_PATH: str = "models/yolov8_custom.pt"
    
    # OCR Settings
    USE_GPU: bool = True
    
    class Config:
        case_sensitive = True

def load_yaml_config(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f)

settings = Settings()
yaml_config = load_yaml_config(CONFIG_PATH)
