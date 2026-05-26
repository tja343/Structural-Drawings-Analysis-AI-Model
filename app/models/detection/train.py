import os
from contextlib import nullcontext
from ultralytics import YOLO
from pathlib import Path
from app.core.logger import logger

try:
    import mlflow
except ImportError:
    mlflow = None

class DetectionTrainer:
    def __init__(self, data_yaml: str, base_model: str = "yolov8n.pt", project_name: str = "structural_ai_detection"):
        self.data_yaml = str(Path(data_yaml).absolute())
        self.base_model = base_model
        self.project_name = project_name
        self.model = YOLO(self.base_model)
        
    def train(self, epochs: int = 100, imgsz: int = 640, batch: int = 16, device: str = "0"):
        logger.info(f"Starting YOLOv8 training for {epochs} epochs on device {device}")
        
        if mlflow is not None:
            mlflow.set_experiment(self.project_name)
            run_context = mlflow.start_run()
        else:
            run_context = None

        with run_context if run_context is not None else nullcontext():
            if mlflow is not None:
                mlflow.log_params({
                    "epochs": epochs,
                    "imgsz": imgsz,
                    "batch": batch,
                    "base_model": self.base_model
                })

            results = self.model.train(
                data=self.data_yaml,
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                device=device,
                project="runs/detect",
                name="train_run",
                exist_ok=True,
                save=True,
                save_period=10,
                val=True,
                workers=0,
                plots=False
            )

            if mlflow is not None:
                mlflow.log_metrics({
                    "map50": results.box.map50,
                    "map": results.box.map
                })

            logger.info("Training complete. Models saved to runs/detect/train_run/weights/")
        return results
