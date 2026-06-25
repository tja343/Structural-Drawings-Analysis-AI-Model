# React Interface Blueprint

## Objective

The original Streamlit dashboard was rebuilt as a React/Vite interface while keeping the same core workflows:

- overview metrics for generated samples, labels, semantics, model weights, and YOLO splits
- synthetic dataset browsing with rendered YOLO boxes, annotation tables, and semantic JSON
- image upload preprocessing with original, HSV mask, and cleaned previews
- FastAPI image and PDF inference calls with structured JSON output

## Frontend Architecture

The React app lives in `frontend/`.

- `frontend/src/App.jsx` contains the main workspace, navigation, dashboard views, sample explorer, upload workflow, and API inference workflow.
- `frontend/src/styles.css` defines the visual system: dark professional console, blueprint grid animation, scanning overlays, animated workflow steps, gradient status bars, and responsive layouts.
- `frontend/package.json` provides Vite scripts for local development and production builds.

The UI uses the project's own generated structural drawing samples as visual material instead of stock imagery, so the graphics stay relevant to the product.

## Backend Additions

FastAPI now exposes React-friendly helper endpoints while preserving the existing inference endpoints.

- `GET /api/v1/dashboard` returns counts, model weight status, split counts, and source summaries.
- `GET /api/v1/samples` returns indexed sample metadata and image URLs.
- `GET /api/v1/samples/{sample_id}` returns labels and semantic JSON for one sample.
- `GET /api/v1/samples/{sample_id}/image?boxes=true` returns a PNG preview with YOLO boxes drawn on top.
- `POST /api/v1/preprocess/image` returns original, HSV mask, cleaned image previews, and retained pixel statistics.
- Existing `POST /api/v1/inference/image` and `POST /api/v1/inference/pdf` remain the inference bridge.

## Visual Direction

The redesign uses a compact engineering-console layout:

- sticky left navigation for Overview, Synthetic Dataset, Model Detection, and API Inference
- first-screen animated hero with blueprint grid motion, luminous beam lines, and real annotated sample imagery
- teal, cyan, and magenta accents against a dark neutral base
- restrained card radius, dense tables, and workspace-oriented controls
- responsive layouts for desktop and smaller screens

## Running

Start FastAPI:

```powershell
$env:USE_GPU="false"
.\.venv312\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Start React:

```powershell
cd frontend
npm install
npm run dev -- --port 5173
```

Open `http://127.0.0.1:5173`.

If the API runs somewhere else, set `VITE_API_BASE` before starting Vite.
