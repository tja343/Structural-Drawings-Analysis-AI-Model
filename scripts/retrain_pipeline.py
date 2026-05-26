import sys
import subprocess
from pathlib import Path

def run_command(command: str):
    print(f"Executing: {command}")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"Error executing command: {command}")
        sys.exit(1)

def main():
    print("--- Structural AI MLOps Retraining Pipeline ---")
    
    # 1. Generate new synthetic data
    print("\n[1/4] Generating new synthetic data...")
    run_command("python scripts/generate_synthetic_data.py")
    
    # 2. Re-process real PDFs if any exist and build splits
    print("\n[2/4] Preparing dataset splits...")
    run_command("python scripts/prepare_dataset.py")
    
    # 3. Train the YOLOv8 model on the new dataset
    print("\n[3/4] Retraining Object Detection Model...")
    run_command("python scripts/train_detection.py")
    
    # 4. Run PyTest suite to ensure nothing broke
    print("\n[4/4] Running Validation Suite...")
    run_command("pytest tests/")
    
    print("\n✅ Retraining Pipeline Complete. New model weights are in runs/detect/train_run/weights/best.pt")

if __name__ == "__main__":
    main()
