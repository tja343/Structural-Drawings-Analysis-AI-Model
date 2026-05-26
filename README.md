# Structural Drawing AI System

An end-to-end AI pipeline for parsing, detecting, and structurally analyzing PDF engineering drawings.

## Architecture

This project is built using a multi-stage modular architecture:
1. **Procedural Synthetic Dataset Generator**: Bootstraps training data without requiring manual human labeling.
2. **PyMuPDF Extraction**: Rips vectors and raster images from real PDF drawings.
3. **Ultralytics YOLO**: Performs deep learning object detection for Beams, Support Regions, Dimensions, and Text boxes.
4. **PaddleOCR**: Extracts raw string values from the text bounding boxes.
5. **Regex Parsing Engine**: Cleans up OCR errors and extracts engineering semantics (`bar_type`, `diameter`, `spacing`).
6. **Spatial Relationship Engine**: Associates text annotations to physical beams via Euclidean bounding box logic.
7. **FastAPI Backend**: Wraps the entire pipeline into an Async REST API.

## Installation

### Local Setup
```bash
pip install -r requirements.txt
```

### Docker (GPU Recommended)
```bash
docker-compose up --build
```

## Usage

### 1. Generate Synthetic Data
```bash
python scripts/generate_synthetic_data.py
python scripts/prepare_dataset.py
```

### 2. Train Models
```bash
python scripts/train_detection.py
```

### 3. Run Inference API
```bash
uvicorn app.main:app --reload
```
Navigate to `http://localhost:8000/docs` to test uploading PDFs or Images.

## Continuous Retraining
To run the automated MLOps script that regenerates data, splits it, trains the model, and validates it:
```bash
python scripts/retrain_pipeline.py
```
