# Structural Drawing AI System

An end-to-end AI workflow for generating synthetic structural drawing data, training a YOLO detector, previewing model output in Streamlit, and exposing inference through a FastAPI backend.

The project is designed around structural engineering drawings where the system needs to detect regions such as beams and text annotations, parse reinforcement notations, and return structured engineering data.

## What This Project Does

- Generates synthetic structural drawing images with matching YOLO labels and semantic metadata.
- Splits generated data into train, validation, and test sets.
- Trains a YOLOv8 object detector on drawing elements.
- Provides a Streamlit dashboard for dataset inspection and model preview.
- Provides a FastAPI backend for image and PDF inference endpoints.
- Includes parsing and spatial association components for converting detections and OCR output into structured engineering JSON.

## Current Capabilities

The repository currently includes:

- A generated 100-image synthetic dataset under `data/synthetic`.
- YOLO train, validation, and test splits under `data/yolo`.
- A trained YOLO model at `models/yolov8_custom.pt`.
- A Streamlit frontend at `streamlit_app.py`.
- A FastAPI backend at `app/main.py`.
- Unit tests for API health and reinforcement text parsing.

The YOLO detector is trained on the synthetic data included in this repository. Full OCR inference depends on PaddleOCR and PaddlePaddle, which may require a compatible Python and GPU environment.

## Project Structure

```text
app/
  api/                 FastAPI routes
  core/                Configuration and logging
  dataset/             PDF processing and dataset helpers
  exporters/           JSON output generation
  models/              Detection and OCR services
  parsing/             Reinforcement notation parser
  pipeline/            End-to-end inference orchestration
  schemas/             Pydantic response and engineering schemas
  spatial/             Spatial relationship logic
  synthetic/           Synthetic drawing generator and exporter

config/
  default.yaml         Runtime and path configuration

data/
  synthetic/           Generated synthetic images, labels, and semantics
  yolo/                YOLO train/validation/test split

models/
  yolov8_custom.pt     Trained YOLO detector weights

scripts/
  generate_synthetic_data.py
  prepare_dataset.py
  train_detection.py
  retrain_pipeline.py

tests/
  test_api.py
  test_parser.py

streamlit_app.py       Streamlit dashboard
Dockerfile             Docker image definition
docker-compose.yml     Docker Compose service definition
```

## Architecture

The workflow is split into modular stages:

1. Synthetic dataset generation creates drawing images, bounding boxes, and semantic text metadata.
2. Dataset preparation copies generated samples into YOLO-compatible train, validation, and test folders.
3. YOLOv8 detection identifies drawing regions such as text and beams.
4. PaddleOCR extracts raw text from detected text regions.
5. Regex parsing normalizes engineering annotations such as `H10@300`, `T12@150`, and `Y16 TOP`.
6. Spatial association links parsed annotations to nearby structural elements.
7. FastAPI exposes inference endpoints for uploaded images and PDFs.
8. Streamlit provides a user-facing dashboard for inspection and model preview.

## Requirements

Recommended local environment:

- Python 3.11 or 3.12 for best compatibility with the ML and OCR stack.
- Python 3.14 can run the Streamlit UI and YOLO preview, but some OCR dependencies may not install cleanly.
- Docker is recommended for a more reproducible GPU-enabled deployment.
- NVIDIA GPU support is recommended for full training and OCR performance.

Install dependencies:

```bash
pip install -r requirements.txt
```

If you only need the Streamlit dashboard and YOLO preview on a local CPU machine, the current repository state already includes a trained YOLO weight file.

## Running The Streamlit Frontend

Start the Streamlit dashboard:

```bash
streamlit run streamlit_app.py
```

Then open:

```text
http://127.0.0.1:8501
```

The dashboard includes:

- Overview of dataset and model status.
- Synthetic dataset viewer with YOLO boxes and semantic JSON.
- Model detection preview using the trained YOLO weights.
- API inference panel for calling the FastAPI backend.

## Running The FastAPI Backend

Start the API:

```bash
uvicorn app.main:app --reload
```

Open the API docs:

```text
http://127.0.0.1:8000/docs
```

Useful endpoints:

```text
GET  /health
POST /api/v1/inference/image
POST /api/v1/inference/pdf
```

The image and PDF inference endpoints run the full pipeline. The full pipeline requires the OCR dependencies to be installed successfully.

## Generating Synthetic Data

Generate 100 synthetic structural drawings:

```bash
python scripts/generate_synthetic_data.py
```

Output is written to:

```text
data/synthetic/images
data/synthetic/labels
data/synthetic/semantics
```

Each sample includes:

- A `.png` drawing image.
- A YOLO `.txt` label file.
- A semantic `.json` metadata file for generated text annotations.

## Preparing YOLO Splits

Create train, validation, and test splits from the synthetic dataset:

```bash
python scripts/prepare_dataset.py
```

Output is written to:

```text
data/yolo/train
data/yolo/val
data/yolo/test
```

The included split uses:

```text
80 train images
10 validation images
10 test images
```

## Training The Detection Model

Train YOLO with default settings:

```bash
python scripts/train_detection.py
```

For a shorter local CPU run:

```bash
python scripts/train_detection.py --epochs 10 --imgsz 256 --batch 4
```

The training script supports:

```text
--epochs
--imgsz
--batch
```

Trained weights are written by YOLO under:

```text
runs/detect/runs/detect/train_run/weights/best.pt
```

The application is configured to use:

```text
models/yolov8_custom.pt
```

## Docker Usage

Build and run with Docker Compose:

```bash
docker-compose up --build
```

The provided Docker setup is oriented toward GPU-enabled environments. You may need to adjust PaddlePaddle and CUDA versions depending on your hardware and driver stack.

## Testing

Run the test suite:

```bash
pytest
```

Current tests cover:

- Root API response.
- Health check endpoint.
- Upload validation behavior.
- Reinforcement notation parsing.

## Important Environment Notes

- `paddlepaddle-gpu` may not be available for every Python and operating system combination.
- On Windows, YOLO training is more stable with `workers=0`, which is already set in the training helper.
- The Streamlit model preview can run with the trained YOLO model even if the full OCR stack is not installed.
- The FastAPI health and docs routes do not require YOLO or OCR to load at import time because the inference pipeline is lazily initialized.

## Typical Development Workflow

```bash
python scripts/generate_synthetic_data.py
python scripts/prepare_dataset.py
python scripts/train_detection.py --epochs 10 --imgsz 256 --batch 4
uvicorn app.main:app --reload
streamlit run streamlit_app.py
```

Open:

```text
http://127.0.0.1:8501
http://127.0.0.1:8000/docs
```

## Repository Status

This repository includes code, sample synthetic data, YOLO split data, and a trained detector weight file so the dashboard can be explored immediately after installing dependencies.
