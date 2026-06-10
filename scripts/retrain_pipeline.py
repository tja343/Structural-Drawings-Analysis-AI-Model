import subprocess
import sys


def run_command(command: str):
    print(f"Executing: {command}")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"Error executing command: {command}")
        sys.exit(1)


def main():
    print("--- Structural AI MLOps Retraining Pipeline ---")

    print("\nGenerating new synthetic data...")
    run_command(f"{sys.executable} -m scripts.generate_synthetic_data")

    print("\nPreparing dataset splits...")
    run_command(f"{sys.executable} -m scripts.prepare_dataset")

    print("\nRetraining object detection model...")
    run_command(f"{sys.executable} -m scripts.train_detection")

    print("\nRunning validation suite...")
    run_command(f"{sys.executable} -m pytest tests/")

    print("\nRetraining pipeline complete. New model weights are in runs/detect/train_run/weights/best.pt")


if __name__ == "__main__":
    main()
