from fastapi import APIRouter, File, UploadFile, HTTPException, Response
from app.schemas.api import InferenceResponse, BatchInferenceResponse
from app.core.logger import logger
from app.preprocessing.color_isolation import isolate_colored_annotations
from PIL import Image, ImageDraw
from pathlib import Path
from typing import Any
import base64
import cv2
import tempfile
import os
import json
import numpy as np

router = APIRouter()
_orchestrator = None
_pdf_processor = None

ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC_DIR = ROOT / "data" / "synthetic"
SYNTHETIC_OVERLAY_DIR = ROOT / "data" / "synthetic_overlays_cleaned"
YOLO_DIR = ROOT / "data" / "yolo"
TRAINED_WEIGHTS = ROOT / "models" / "yolov8_custom.pt"
RUN_WEIGHTS = ROOT / "runs" / "detect" / "train_run" / "weights" / "best.pt"
CLASS_NAMES = {
    0: "Text",
    1: "Rebar Region",
    2: "Arrow",
    3: "Beam",
    4: "Dimension",
    5: "Support",
}
PREVIEW_COLORS = {
    0: "#ec4899",
    1: "#22c55e",
    2: "#f59e0b",
    3: "#38bdf8",
    4: "#a78bfa",
    5: "#fb7185",
}


def get_orchestrator():
    global _orchestrator
    if _orchestrator is None:
        from app.pipeline.orchestrator import InferenceOrchestrator

        _orchestrator = InferenceOrchestrator()
    return _orchestrator


def get_pdf_processor():
    global _pdf_processor
    if _pdf_processor is None:
        from app.dataset.pdf_processor import PDFProcessor

        _pdf_processor = PDFProcessor(dpi=300)
    return _pdf_processor


def count_files(path: Path, pattern: str) -> int:
    return len(list(path.glob(pattern))) if path.exists() else 0


def get_weight_path() -> Path | None:
    if TRAINED_WEIGHTS.exists():
        return TRAINED_WEIGHTS
    if RUN_WEIGHTS.exists():
        return RUN_WEIGHTS
    return None


def source_roots() -> list[tuple[str, str, Path]]:
    return [
        ("synthetic", "Synthetic", SYNTHETIC_DIR),
        ("overlay", "Floor-plan overlay", SYNTHETIC_OVERLAY_DIR),
    ]


def load_yolo_labels(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cls, xc, yc, width, height = parts
        class_id = int(cls)
        rows.append(
            {
                "class_id": class_id,
                "class_name": CLASS_NAMES.get(class_id, str(class_id)),
                "x_center": float(xc),
                "y_center": float(yc),
                "width": float(width),
                "height": float(height),
            }
        )
    return rows


def sample_index() -> list[dict[str, Any]]:
    samples = []
    for source_id, source_name, root in source_roots():
        image_root = root / "images"
        for image_path in sorted(image_root.glob("*.png")) if image_root.exists() else []:
            sample_id = f"{source_id}:{image_path.stem}"
            label_path = root / "labels" / f"{image_path.stem}.txt"
            semantic_path = root / "semantics" / f"{image_path.stem}.json"
            samples.append(
                {
                    "id": sample_id,
                    "source": source_name,
                    "filename": image_path.name,
                    "image_url": f"/api/v1/samples/{sample_id}/image",
                    "boxed_image_url": f"/api/v1/samples/{sample_id}/image?boxes=true",
                    "label_count": len(load_yolo_labels(label_path)),
                    "has_semantics": semantic_path.exists(),
                }
            )
    return samples


def resolve_sample(sample_id: str) -> tuple[Path, Path, Path, str]:
    try:
        source_id, stem = sample_id.split(":", 1)
    except ValueError:
        raise HTTPException(status_code=404, detail="Unknown sample")
    roots = {source_id: (source_name, root) for source_id, source_name, root in source_roots()}
    if source_id not in roots:
        raise HTTPException(status_code=404, detail="Unknown sample")
    source_name, root = roots[source_id]
    image_path = root / "images" / f"{stem}.png"
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Unknown sample")
    return image_path, root / "labels" / f"{stem}.txt", root / "semantics" / f"{stem}.json", source_name


def draw_yolo_boxes(image: Image.Image, labels: list[dict[str, Any]]) -> Image.Image:
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    width, height = canvas.size
    for item in labels:
        xc = item["x_center"] * width
        yc = item["y_center"] * height
        box_w = item["width"] * width
        box_h = item["height"] * height
        x1 = xc - box_w / 2
        y1 = yc - box_h / 2
        x2 = xc + box_w / 2
        y2 = yc + box_h / 2
        color = PREVIEW_COLORS.get(item["class_id"], "#94a3b8")
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        draw.text((x1 + 4, max(0, y1 - 18)), item["class_name"], fill=color)
    return canvas


def png_response(image: Image.Image) -> Response:
    from io import BytesIO

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return Response(content=buffer.getvalue(), media_type="image/png")


def data_url_from_array(image: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise HTTPException(status_code=500, detail="Could not encode preview image")
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/png;base64,{payload}"


@router.get("/dashboard")
async def dashboard_summary():
    synthetic_image_count = count_files(SYNTHETIC_DIR / "images", "*.png")
    synthetic_label_count = count_files(SYNTHETIC_DIR / "labels", "*.txt")
    synthetic_semantic_count = count_files(SYNTHETIC_DIR / "semantics", "*.json")
    overlay_image_count = count_files(SYNTHETIC_OVERLAY_DIR / "images", "*.png")
    overlay_label_count = count_files(SYNTHETIC_OVERLAY_DIR / "labels", "*.txt")
    overlay_semantic_count = count_files(SYNTHETIC_OVERLAY_DIR / "semantics", "*.json")
    return {
        "metrics": {
            "sample_images": synthetic_image_count + overlay_image_count,
            "yolo_labels": synthetic_label_count + overlay_label_count,
            "semantics": synthetic_semantic_count + overlay_semantic_count,
            "trained_weights": bool(get_weight_path()),
        },
        "model": {
            "weight_path": str(get_weight_path()) if get_weight_path() else None,
        },
        "splits": [
            {"split": "train", "images": count_files(YOLO_DIR / "train" / "images", "*.png")},
            {"split": "val", "images": count_files(YOLO_DIR / "val" / "images", "*.png")},
            {"split": "test", "images": count_files(YOLO_DIR / "test" / "images", "*.png")},
        ],
        "sources": [
            {
                "source": "Synthetic",
                "images": synthetic_image_count,
                "labels": synthetic_label_count,
                "semantics": synthetic_semantic_count,
            },
            {
                "source": "Cleaned floor-plan overlays",
                "images": overlay_image_count,
                "labels": overlay_label_count,
                "semantics": overlay_semantic_count,
            },
        ],
    }


@router.get("/samples")
async def list_samples():
    return {"samples": sample_index()}


@router.get("/samples/{sample_id}")
async def get_sample(sample_id: str):
    image_path, label_path, semantic_path, source_name = resolve_sample(sample_id)
    semantic = json.loads(semantic_path.read_text(encoding="utf-8")) if semantic_path.exists() else {}
    return {
        "id": sample_id,
        "source": source_name,
        "filename": image_path.name,
        "image_url": f"/api/v1/samples/{sample_id}/image",
        "boxed_image_url": f"/api/v1/samples/{sample_id}/image?boxes=true",
        "labels": load_yolo_labels(label_path),
        "semantics": semantic,
    }


@router.get("/samples/{sample_id}/image")
async def get_sample_image(sample_id: str, boxes: bool = False):
    image_path, label_path, _, _ = resolve_sample(sample_id)
    image = Image.open(image_path).convert("RGB")
    if boxes:
        image = draw_yolo_boxes(image, load_yolo_labels(label_path))
    return png_response(image)


@router.post("/preprocess/image")
async def preprocess_image_upload(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    result = isolate_colored_annotations(image)
    return {
        "colored_pixel_count": int(result.colored_pixel_count),
        "retained_ratio": float(result.retained_ratio),
        "original": data_url_from_array(image),
        "mask": data_url_from_array(cv2.cvtColor(result.color_mask, cv2.COLOR_GRAY2BGR)),
        "cleaned": data_url_from_array(result.cleaned),
    }

@router.post("/inference/image", response_model=InferenceResponse)
async def process_image_upload(file: UploadFile = File(...)):
    """Upload a structural drawing image for end-to-end processing."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    import cv2
    import numpy as np
        
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
        
    drawing_id = file.filename or "uploaded_image"
    
    try:
        orchestrator = get_orchestrator()
        result = orchestrator.process_image(drawing_id, image)
        return InferenceResponse(status="success", message="Inference complete", data=result)
    except Exception as e:
        logger.error(f"Inference failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/inference/pdf", response_model=BatchInferenceResponse)
async def process_pdf_upload(file: UploadFile = File(...)):
    """Upload a structural PDF, extract high-res pages, and process each."""
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="File must be a PDF")

    import cv2
        
    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path = os.path.join(tmpdir, file.filename)
        with open(pdf_path, "wb") as f:
            f.write(await file.read())
            
        try:
            orchestrator = get_orchestrator()
            pdf_processor = get_pdf_processor()

            # 1. Convert PDF to PNGs
            image_paths = pdf_processor.pdf_to_images(pdf_path, tmpdir)
            
            # 2. Process each page through orchestrator
            results = []
            for img_path in image_paths:
                page_img = cv2.imread(img_path)
                page_id = os.path.basename(img_path)
                page_res = orchestrator.process_image(page_id, page_img)
                results.append(page_res)
                
            return BatchInferenceResponse(
                status="success", 
                message="PDF batch processing complete", 
                processed_count=len(results),
                data=results
            )
        except Exception as e:
            logger.error(f"PDF Inference failed: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
