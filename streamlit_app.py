from __future__ import annotations

import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import requests
import streamlit as st
from PIL import Image, ImageDraw

from app.preprocessing.color_isolation import ColorIsolationResult, isolate_colored_annotations

ROOT = Path(__file__).parent
SYNTHETIC_DIR = ROOT / "data" / "synthetic"
SYNTHETIC_OVERLAY_DIR = ROOT / "data" / "synthetic_overlays_cleaned"
YOLO_DIR = ROOT / "data" / "yolo"
TRAINED_WEIGHTS = ROOT / "models" / "yolov8_custom.pt"
RUN_WEIGHTS = ROOT / "runs" / "detect" / "train_run" / "weights" / "best.pt"
API_BASE_URL = "http://127.0.0.1:8000"
CLASS_NAMES = {
    0: "Text",
    1: "Rebar Region",
    2: "Arrow",
    3: "Beam",
    4: "Dimension",
    5: "Support",
}
PREVIEW_COLORS = {
    0: "#dc2626",
    3: "#2563eb",
}


st.set_page_config(
    page_title="Structural Drawing AI",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

def load_stylesheet(path: Path) -> None:
    css = path.read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


load_stylesheet(ROOT / "assets" / "style.css")


def count_files(path: Path, pattern: str) -> int:
    return len(list(path.glob(pattern))) if path.exists() else 0


def sample_sources() -> list[dict]:
    sources = [
        ("Synthetic", SYNTHETIC_DIR),
        ("Floor-plan overlay", SYNTHETIC_OVERLAY_DIR),
    ]
    samples = []
    for source_name, root in sources:
        for image_path in sorted((root / "images").glob("*.png")):
            samples.append(
                {
                    "source": source_name,
                    "image": image_path,
                    "label": root / "labels" / f"{image_path.stem}.txt",
                    "semantic": root / "semantics" / f"{image_path.stem}.json",
                }
            )
    return samples


def sample_label(sample: dict) -> str:
    return f"{sample['source']} / {sample['image'].name}"


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_yolo_labels(path: Path) -> list[dict]:
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


def draw_yolo_boxes(image: Image.Image, labels: list[dict]) -> Image.Image:
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
        color = PREVIEW_COLORS.get(item["class_id"], "#9ca3af")
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        draw.text((x1 + 4, max(0, y1 - 18)), item["class_name"], fill=color)
    return canvas


def pil_to_bgr(image: Image.Image) -> np.ndarray:
    rgb = np.array(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def bgr_to_pil(image: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def mask_to_pil(mask: np.ndarray) -> Image.Image:
    return Image.fromarray(mask).convert("RGB")


def preprocess_pil_image(image: Image.Image) -> ColorIsolationResult:
    return isolate_colored_annotations(pil_to_bgr(image))


def render_pdf_preview_pages(pdf_bytes: bytes, filename: str, max_pages: int = 3) -> list[tuple[str, Image.Image]]:
    from app.dataset.pdf_processor import PDFProcessor

    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path = Path(tmpdir) / filename
        pdf_path.write_bytes(pdf_bytes)
        processor = PDFProcessor(dpi=160)
        page_paths = processor.pdf_to_images(str(pdf_path), tmpdir)
        pages = []
        for page_path in page_paths[:max_pages]:
            page_image = Image.open(page_path).convert("RGB")
            pages.append((Path(page_path).name, page_image.copy()))
    return pages


def show_preprocessing_steps(title: str, image: Image.Image) -> ColorIsolationResult:
    result = preprocess_pil_image(image)
    st.subheader(title)
    st.caption(
        f"Colored pixels retained: {result.colored_pixel_count:,} "
        f"({result.retained_ratio:.2%} of the page)."
    )
    original_col, mask_col, cleaned_col = st.columns(3)
    original_col.image(image.convert("RGB"), caption="1. Original upload", use_container_width=True)
    mask_col.image(mask_to_pil(result.color_mask), caption="2. HSV color mask", use_container_width=True)
    cleaned_col.image(bgr_to_pil(result.cleaned), caption="3. Grayscale removed", use_container_width=True)
    return result


@st.cache_resource(show_spinner=False)
def load_detection_model(weights_path: str):
    from ultralytics import YOLO

    return YOLO(weights_path)


def get_weight_path() -> Path | None:
    if TRAINED_WEIGHTS.exists():
        return TRAINED_WEIGHTS
    if RUN_WEIGHTS.exists():
        return RUN_WEIGHTS
    return None


def workflow_panel() -> None:
    st.markdown(
        """
        <div class="step-row">
            <div class="step"><b>1. Synthetic data</b><span>Generated drawings provide image, YOLO label, and semantic JSON pairs.</span></div>
            <div class="step"><b>2. YOLO split</b><span>The dataset is split into train, validation, and test folders under data/yolo.</span></div>
            <div class="step"><b>3. Color isolation</b><span>HSV saturation removes grayscale floor-plan lines and keeps colored reinforcement marks.</span></div>
            <div class="step"><b>4. Detection and JSON</b><span>YOLO and OCR run on the cleaned image before the final engineering JSON is returned.</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def info_panel(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="info-panel">
            <strong>{title}</strong>
            <p>{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def draw_predictions(image: Image.Image, predictions) -> tuple[Image.Image, pd.DataFrame]:
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    rows = []
    for result in predictions:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = float(box.conf[0].item())
            class_id = int(box.cls[0].item())
            class_name = result.names[class_id]
            rows.append(
                {
                    "class": class_name,
                    "confidence": round(confidence, 4),
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2),
                }
            )
            draw.rectangle((x1, y1, x2, y2), outline="#2563eb", width=3)
            draw.text((x1 + 4, max(0, y1 - 18)), f"{class_name} {confidence:.2f}", fill="#2563eb")
    return canvas, pd.DataFrame(rows)


st.sidebar.title("Structural AI")
section = st.sidebar.radio(
    "Workspace",
    ["Overview", "Synthetic Dataset", "Model Detection", "API Inference"],
)
st.sidebar.divider()
st.sidebar.markdown("**Runtime**")
st.sidebar.caption(f"API endpoint: {API_BASE_URL}")
st.sidebar.caption("Frontend: http://127.0.0.1:8501")
st.sidebar.markdown("**Model**")
st.sidebar.caption(str(get_weight_path() or "not trained yet"))

st.markdown(
    """
    <div class="hero-panel">
        <h2>Structural Drawing AI Console</h2>
        <p>Monitor synthetic data, remove grayscale floor-plan backgrounds, preview detector output, and call the FastAPI inference service from one dark-mode workspace.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if section == "Overview":
    synthetic_image_count = count_files(SYNTHETIC_DIR / "images", "*.png")
    synthetic_label_count = count_files(SYNTHETIC_DIR / "labels", "*.txt")
    synthetic_semantic_count = count_files(SYNTHETIC_DIR / "semantics", "*.json")
    overlay_image_count = count_files(SYNTHETIC_OVERLAY_DIR / "images", "*.png")
    overlay_label_count = count_files(SYNTHETIC_OVERLAY_DIR / "labels", "*.txt")
    overlay_semantic_count = count_files(SYNTHETIC_OVERLAY_DIR / "semantics", "*.json")
    image_count = synthetic_image_count + overlay_image_count
    label_count = synthetic_label_count + overlay_label_count
    semantic_count = synthetic_semantic_count + overlay_semantic_count
    train_count = count_files(YOLO_DIR / "train" / "images", "*.png")
    val_count = count_files(YOLO_DIR / "val" / "images", "*.png")
    test_count = count_files(YOLO_DIR / "test" / "images", "*.png")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Sample Images", image_count)
    col2.metric("YOLO Labels", label_count)
    col3.metric("Semantics", semantic_count)
    col4.metric("Trained Weights", "Ready" if get_weight_path() else "Missing")

    workflow_panel()

    left_info, right_info = st.columns(2)
    with left_info:
        info_panel(
            "Current dataset state",
            f"{synthetic_image_count} base synthetic drawings and {overlay_image_count} cleaned floor-plan overlay drawings are available. "
            f"The active YOLO training split now contains {train_count} images.",
        )
    with right_info:
        model_text = str(get_weight_path()) if get_weight_path() else "No trained weight file found."
        st.markdown(
            f"""
            <div class="info-panel">
                <strong>Active model file</strong>
                <p><span class="path-chip">{model_text}</span></p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.subheader("Dataset Split")
    st.caption("The detector trains on train/images, tunes against val/images, and reserves test/images for a later independent check.")
    split_df = pd.DataFrame(
        [
            {"split": "train", "images": train_count},
            {"split": "val", "images": val_count},
            {"split": "test", "images": test_count},
        ]
    )
    if split_df["images"].sum() == 0:
        st.warning(
            "No prepared YOLO split was found. Run `python -m scripts.prepare_dataset` "
            "from the project root to create data/yolo/train, data/yolo/val, and data/yolo/test."
        )
    else:
        st.dataframe(split_df, use_container_width=True, hide_index=True)
        st.bar_chart(split_df, x="split", y="images", color="#2563eb")

    st.subheader("Sample Sources")
    source_df = pd.DataFrame(
        [
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
        ]
    )
    st.dataframe(source_df, use_container_width=True, hide_index=True)

    if get_weight_path():
        st.markdown('<span class="status-pill">Model weights available</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="warn-pill">Train the model to enable local detection preview</span>', unsafe_allow_html=True)

elif section == "Synthetic Dataset":
    info_panel(
        "Synthetic dataset explorer",
        "Use this view to inspect generated drawings, cleaned floor-plan overlays, their YOLO bounding boxes, and the semantic text metadata exported with each image.",
    )
    samples = sample_sources()
    if not samples:
        st.warning("No synthetic images found. Run python -m scripts.generate_synthetic_data first.")
    else:
        selected = st.selectbox("Sample image", samples, format_func=sample_label)
        image_path = selected["image"]
        label_path = selected["label"]
        semantic_path = selected["semantic"]
        labels = load_yolo_labels(label_path)
        base_image = Image.open(image_path)

        left, right = st.columns([1.4, 1])
        with left:
            st.subheader("Rendered Sample")
            st.caption("Boxes are drawn from the YOLO label file. The generated bars and reinforcement text use randomized visible colors.")
            st.image(draw_yolo_boxes(base_image, labels), caption=sample_label(selected), use_container_width=True)
        with right:
            st.markdown(f'<span class="path-chip">{image_path}</span>', unsafe_allow_html=True)
            st.subheader("Annotations")
            label_df = pd.DataFrame(labels)
            if label_df.empty:
                st.write("No labels found for this sample.")
            else:
                st.dataframe(label_df, use_container_width=True, hide_index=True)
            st.subheader("Semantics")
            st.json(read_json(semantic_path))

elif section == "Model Detection":
    info_panel(
        "Detector preview",
        "This view removes grayscale floor-plan linework first, then runs the trained YOLO detector on the cleaned image. It mirrors the preprocessing now used by the backend pipeline.",
    )
    weight_path = get_weight_path()
    if weight_path is None:
        st.warning("No trained YOLO weights found yet. Train first, then come back here.")
    else:
        st.info(f"Using weights: {weight_path}")
        uploaded = st.file_uploader("Upload a drawing image", type=["png", "jpg", "jpeg"])
        fallback_samples = sample_sources()
        sample_choice = st.selectbox(
            "Or use a sample image",
            fallback_samples,
            format_func=sample_label,
            disabled=uploaded is not None or not fallback_samples,
        )

        image = Image.open(uploaded).convert("RGB") if uploaded else Image.open(sample_choice["image"]).convert("RGB")
        confidence = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
        preprocessing = show_preprocessing_steps("Preprocessing", image)
        model = load_detection_model(str(weight_path))
        cleaned_image = bgr_to_pil(preprocessing.cleaned)
        predictions = model.predict(cleaned_image, conf=confidence, verbose=False)
        rendered, table = draw_predictions(cleaned_image, predictions)

        left, right = st.columns([1.4, 1])
        with left:
            st.subheader("4. Detection Preview")
            st.image(rendered, caption="YOLO detections on cleaned image", use_container_width=True)
        with right:
            st.subheader("Detections")
            if table.empty:
                st.write("No detections at this threshold.")
            else:
                st.metric("Detected Regions", len(table))
                st.dataframe(table, use_container_width=True, hide_index=True)

elif section == "API Inference":
    info_panel(
        "FastAPI inference bridge",
        "Upload an image or PDF here to see the preprocessing stages and then call the backend service. The backend also removes grayscale floor-plan linework before detection and OCR.",
    )
    st.write("The FastAPI backend must be running on port 8000 before using this panel.")
    uploaded = st.file_uploader("PDF or image", type=["pdf", "png", "jpg", "jpeg"])
    if uploaded:
        is_pdf = uploaded.type == "application/pdf"
        endpoint = "/api/v1/inference/pdf" if is_pdf else "/api/v1/inference/image"
        upload_bytes = uploaded.getvalue()

        if is_pdf:
            with st.spinner("Rendering PDF preview pages"):
                preview_pages = render_pdf_preview_pages(upload_bytes, uploaded.name)
            st.caption("Showing preprocessing for the first pages. FastAPI processes every page in the uploaded PDF.")
            for page_name, page_image in preview_pages:
                with st.expander(page_name, expanded=len(preview_pages) == 1):
                    show_preprocessing_steps("Preprocessing", page_image)
        else:
            preview_image = Image.open(uploaded).convert("RGB")
            show_preprocessing_steps("Preprocessing", preview_image)

        if st.button("Run API inference", type="primary"):
            with st.spinner("Sending file to FastAPI"):
                files = {"file": (uploaded.name, upload_bytes, uploaded.type)}
                try:
                    response = requests.post(f"{API_BASE_URL}{endpoint}", files=files, timeout=120)
                    st.code(f"HTTP {response.status_code}")
                    try:
                        st.json(response.json())
                    except requests.JSONDecodeError:
                        st.text(response.text)
                except requests.RequestException as exc:
                    st.error(f"Request failed: {exc}")
