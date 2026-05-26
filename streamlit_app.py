from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from PIL import Image, ImageDraw


ROOT = Path(__file__).parent
SYNTHETIC_DIR = ROOT / "data" / "synthetic"
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


st.set_page_config(
    page_title="Structural Drawing AI",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --bg: #0b1018;
        --panel: #111827;
        --panel-2: #172033;
        --border: #2a3447;
        --text: #f4f7fb;
        --muted: #aab7c8;
        --accent: #3dd6c6;
        --accent-2: #76a9ff;
        --warning: #f5b451;
        --danger: #ff7a7a;
    }
    .stApp {
        background: var(--bg);
        color: var(--text);
    }
    .stApp, .stMarkdown, .stText, p, span, label, div {
        color: var(--text);
    }
    section[data-testid="stSidebar"] {
        background: #080d14;
        border-right: 1px solid var(--border);
    }
    section[data-testid="stSidebar"] * {
        color: var(--text);
    }
    [data-testid="stHeader"] {
        background: rgba(11, 16, 24, 0.88);
    }
    h1, h2, h3 {
        color: var(--text);
        letter-spacing: 0;
    }
    p, .stCaptionContainer {
        color: var(--muted);
    }
    div[data-testid="stMetric"] {
        background: linear-gradient(180deg, #162032 0%, #111827 100%);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 16px 18px;
        box-shadow: 0 10px 24px rgba(0,0,0,0.22);
    }
    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: var(--text);
    }
    .hero-panel {
        background: linear-gradient(135deg, #121b2b 0%, #0d1824 58%, #102421 100%);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 22px 24px;
        margin: 4px 0 18px 0;
    }
    .hero-panel h2 {
        margin: 0 0 8px 0;
        font-size: 1.45rem;
    }
    .hero-panel p {
        margin: 0;
        line-height: 1.55;
        color: var(--muted);
    }
    .info-panel {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 16px 18px;
        margin-bottom: 14px;
    }
    .info-panel strong {
        color: var(--text);
    }
    .info-panel p {
        margin: 5px 0 0 0;
        line-height: 1.5;
        color: var(--muted);
    }
    .step-row {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 10px;
        margin: 12px 0 20px 0;
    }
    .step {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 12px 14px;
        min-height: 96px;
    }
    .step b {
        color: var(--accent);
        display: block;
        margin-bottom: 6px;
    }
    .step span {
        color: var(--muted);
        font-size: 0.92rem;
        line-height: 1.35;
    }
    .path-chip {
        display: inline-block;
        max-width: 100%;
        padding: 5px 8px;
        border-radius: 6px;
        background: #0d1521;
        border: 1px solid var(--border);
        color: var(--muted);
        font-family: Consolas, monospace;
        font-size: 0.78rem;
        overflow-wrap: anywhere;
    }
    .status-pill {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 999px;
        background: rgba(61, 214, 198, 0.14);
        border: 1px solid rgba(61, 214, 198, 0.4);
        color: #9ef2e8;
        font-size: 0.82rem;
        font-weight: 600;
    }
    .warn-pill {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 999px;
        background: rgba(245, 180, 81, 0.14);
        border: 1px solid rgba(245, 180, 81, 0.4);
        color: #ffd28a;
        font-size: 0.82rem;
        font-weight: 600;
    }
    .stDataFrame, [data-testid="stJson"] {
        border: 1px solid var(--border);
        border-radius: 8px;
        overflow: hidden;
    }
    .stAlert {
        background: var(--panel);
        color: var(--text);
        border: 1px solid var(--border);
    }
    button[kind="primary"] {
        background: var(--accent);
        color: #071013;
        border: 0;
    }
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        background: var(--panel);
        border-color: var(--border);
        color: var(--text);
    }
    @media (max-width: 900px) {
        .step-row { grid-template-columns: 1fr; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def count_files(path: Path, pattern: str) -> int:
    return len(list(path.glob(pattern))) if path.exists() else 0


def synthetic_images() -> list[Path]:
    return sorted((SYNTHETIC_DIR / "images").glob("*.png"))


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
        color = "#0f766e" if item["class_id"] == 3 else "#b42318"
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        draw.text((x1 + 4, max(0, y1 - 18)), item["class_name"], fill=color)
    return canvas


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
            <div class="step"><b>3. Detection model</b><span>YOLOv8 weights detect text regions and structural beam boxes.</span></div>
            <div class="step"><b>4. API handoff</b><span>FastAPI exposes upload endpoints for image and PDF inference workflows.</span></div>
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
        <p>Monitor the synthetic dataset, inspect YOLO labels, preview trained detector output, and call the FastAPI inference service from one dark-mode workspace.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if section == "Overview":
    image_count = count_files(SYNTHETIC_DIR / "images", "*.png")
    label_count = count_files(SYNTHETIC_DIR / "labels", "*.txt")
    semantic_count = count_files(SYNTHETIC_DIR / "semantics", "*.json")
    train_count = count_files(YOLO_DIR / "train" / "images", "*.png")
    val_count = count_files(YOLO_DIR / "val" / "images", "*.png")
    test_count = count_files(YOLO_DIR / "test" / "images", "*.png")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Synthetic Images", image_count)
    col2.metric("YOLO Labels", label_count)
    col3.metric("Semantics", semantic_count)
    col4.metric("Trained Weights", "Ready" if get_weight_path() else "Missing")

    workflow_panel()

    left_info, right_info = st.columns(2)
    with left_info:
        info_panel(
            "Current dataset state",
            f"{image_count} generated drawings are available. YOLO labels and semantic JSON files should match this count before training.",
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
    st.bar_chart(split_df, x="split", y="images", color="#2563eb")

    if get_weight_path():
        st.markdown('<span class="status-pill">Model weights available</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="warn-pill">Train the model to enable local detection preview</span>', unsafe_allow_html=True)

elif section == "Synthetic Dataset":
    info_panel(
        "Synthetic dataset explorer",
        "Use this view to inspect generated drawings, their YOLO bounding boxes, and the semantic text metadata exported with each image.",
    )
    images = synthetic_images()
    if not images:
        st.warning("No synthetic images found. Run scripts/generate_synthetic_data.py first.")
    else:
        selected = st.selectbox("Sample image", images, format_func=lambda p: p.name)
        label_path = SYNTHETIC_DIR / "labels" / f"{selected.stem}.txt"
        semantic_path = SYNTHETIC_DIR / "semantics" / f"{selected.stem}.json"
        labels = load_yolo_labels(label_path)
        base_image = Image.open(selected)

        left, right = st.columns([1.4, 1])
        with left:
            st.subheader("Rendered Sample")
            st.caption("Boxes are drawn from the YOLO label file. Beam boxes use teal; text boxes use red.")
            st.image(draw_yolo_boxes(base_image, labels), caption=selected.name, use_container_width=True)
        with right:
            st.markdown(f'<span class="path-chip">{selected}</span>', unsafe_allow_html=True)
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
        "This view runs the trained YOLO detector directly inside Streamlit. It is useful for checking whether the synthetic-trained model can find text and beam regions before passing files to the full API pipeline.",
    )
    weight_path = get_weight_path()
    if weight_path is None:
        st.warning("No trained YOLO weights found yet. Train first, then come back here.")
    else:
        st.info(f"Using weights: {weight_path}")
        uploaded = st.file_uploader("Upload a drawing image", type=["png", "jpg", "jpeg"])
        fallback_images = synthetic_images()
        sample_choice = st.selectbox(
            "Or use a synthetic sample",
            fallback_images,
            format_func=lambda p: p.name,
            disabled=uploaded is not None or not fallback_images,
        )

        image = Image.open(uploaded) if uploaded else Image.open(sample_choice)
        confidence = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
        model = load_detection_model(str(weight_path))
        predictions = model.predict(image, conf=confidence, verbose=False)
        rendered, table = draw_predictions(image, predictions)

        left, right = st.columns([1.4, 1])
        with left:
            st.subheader("Detection Preview")
            st.image(rendered, caption="Detection preview", use_container_width=True)
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
        "Upload an image or PDF here to call the backend service. Image uploads go to /api/v1/inference/image; PDFs go to /api/v1/inference/pdf.",
    )
    st.write("The FastAPI backend must be running on port 8000 before using this panel.")
    uploaded = st.file_uploader("PDF or image", type=["pdf", "png", "jpg", "jpeg"])
    if uploaded:
        is_pdf = uploaded.type == "application/pdf"
        endpoint = "/api/v1/inference/pdf" if is_pdf else "/api/v1/inference/image"
        if st.button("Run API inference", type="primary"):
            with st.spinner("Sending file to FastAPI"):
                files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
                try:
                    response = requests.post(f"{API_BASE_URL}{endpoint}", files=files, timeout=120)
                    st.code(f"HTTP {response.status_code}")
                    try:
                        st.json(response.json())
                    except requests.JSONDecodeError:
                        st.text(response.text)
                except requests.RequestException as exc:
                    st.error(f"Request failed: {exc}")
