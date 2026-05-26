import sys
import argparse
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.absolute()))

from app.models.detection.train import DetectionTrainer
from app.core.config import yaml_config

def main():
    parser = argparse.ArgumentParser(description="Train the YOLO detection model.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()

    data_yaml = str(Path("data/yolo/dataset.yaml").absolute())
    
    if not Path(data_yaml).exists():
        print(f"Error: {data_yaml} not found. Please run scripts/prepare_dataset.py first.")
        sys.exit(1)
        
    print("Initializing YOLOv8 Detection Trainer...")
    trainer = DetectionTrainer(data_yaml=data_yaml, base_model="yolov8n.pt")
    
    # Check if GPU is available
    import torch
    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    trainer.train(
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device
    )

if __name__ == "__main__":
    main()
