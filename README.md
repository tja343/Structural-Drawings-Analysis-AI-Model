# Structural Drawing AI System

This project is a small structural drawing analysis stack. It can generate synthetic beam drawings, train a YOLO detector, preview the detector in Streamlit, and expose image/PDF inference through a FastAPI service.

The current model is trained against the synthetic drawing style included in this repository, so it is best suited for demos and experiments with similar inputs. Real project drawings will usually need more training data and tuning before the output is dependable.

## What's Included

- Synthetic drawing generation with PNG images, YOLO labels, and semantic JSON.
- A YOLOv8 detector for drawing regions such as text and beams.
- OCR and parsing for reinforcement labels such as `H10@300`, `T20 300 B2`, and `Y16 TOP`.
- Spatial association that attaches parsed text annotations to nearby structural elements.
- A React dashboard for inspection, preview, and API calls.
- The legacy Streamlit dashboard remains available in `streamlit_app.py`.
- A FastAPI backend for image and PDF inference.

Useful paths:

```text
app/                  backend, pipeline, OCR, parser, schemas
config/default.yaml   local runtime defaults
data/synthetic/       generated synthetic samples
data/yolo/            YOLO train/val/test split
models/               checked-in detector weights
scripts/              dataset, training, and utility scripts
frontend/             React/Vite dashboard
streamlit_app.py      legacy dashboard entry point
```

## Environment

Python 3.12 is the safest local choice for the current OCR stack.

```bash
python -m venv .venv312
.\.venv312\Scripts\activate
pip install -r requirements.txt
```

For local CPU inference, set:

```powershell
$env:USE_GPU="false"
```

The repository includes `models/yolov8_custom.pt`, so you can run the demo without retraining first.

## Running the App

Start the FastAPI backend in one terminal:

```powershell
$env:USE_GPU="false"
.\.venv312\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Start React in a second terminal:

```powershell
cd frontend
npm install
npm run dev -- --port 5173
```

Open:

```text
http://127.0.0.1:5173
```

The interface blueprint is documented in `docs/interface_blueprint.md`.

To run the legacy Streamlit dashboard instead:

```powershell
$env:USE_GPU="false"
.\.venv312\Scripts\python.exe -m streamlit run streamlit_app.py --server.address 127.0.0.1 --server.port 8501
```

FastAPI docs are available at:

```text
http://127.0.0.1:8000/docs
```

## Dataset Workflow

Generate synthetic samples:

```bash
python -m scripts.generate_synthetic_data
```

Create YOLO train/validation/test folders:

```bash
python -m scripts.prepare_dataset
```

The default split is 80 percent train, 10 percent validation, and 10 percent test. The split script shuffles the images, so the training set is not simply `0000` through `0079`.

## Training

Train the detector with default settings:

```bash
python -m scripts.train_detection
```

For a quicker local run:

```bash
python -m scripts.train_detection --epochs 10 --imgsz 256 --batch 4
```

YOLO writes new run artifacts under `runs/detect/`. The app first looks for `models/yolov8_custom.pt`, then falls back to the latest training run path configured in `streamlit_app.py`.

To run the whole refresh flow:

```bash
python -m scripts.retrain_pipeline
```

## Testing

```bash
pytest
```

The tests cover the API health routes, upload validation, color isolation, parsing, and synthetic data generation behavior.

## Notes

- If port `8000` is already in use, another FastAPI process is probably running. Stop that process or use a different port.
- OCR depends on PaddleOCR and PaddlePaddle. CPU inference works locally with the pinned dependencies; GPU setups may need a PaddlePaddle build that matches the installed CUDA stack.
- The Streamlit app is only the dashboard. The actual inference call is handled by FastAPI, which returns the JSON output.
