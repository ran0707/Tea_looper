"""
TeaGuard AI — compact 5×5 batch detection dashboard.
Place this file beside best.pt and run: streamlit run app.py

Assumes best.pt is compatible with the installed Ultralytics package.
For custom SCCA layers, import the original model-definition module before YOLO().
"""

from pathlib import Path
import io
import time
import zipfile

import cv2
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO


# --------------------------- Configuration ---------------------------
APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "best.pt"
GRID_COLUMNS = 5
MAX_IMAGES_PER_PAGE = 25

st.set_page_config(
    page_title="Tea Looper Detection",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --------------------------- Compact styling ---------------------------
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 1800px;
    }
    .hero {
        background: linear-gradient(115deg, #174c35, #367c50);
        color: white;
        padding: 14px 20px;
        border-radius: 12px;
        margin-bottom: 12px;
    }
    .hero h1 { color: white; font-size: 1.65rem; margin: 0; }
    .hero p { color: #e5f3e8; font-size: .9rem; margin: 4px 0 0; }
    div[data-testid="stMetric"] {
        background: #f0f6f1;
        border: 1px solid #d9e8dc;
        padding: 8px 12px;
        border-radius: 10px;
    }
    div[data-testid="stImage"] img { border-radius: 6px; }
    .tile-caption { font-size: 11px; color: #53645a; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------- Model loading ---------------------------
@st.cache_resource
def load_model(path: str):
    if not Path(path).is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}. Put best.pt beside app.py."
        )
    # If custom SCCA layers are used, import their module before YOLO(path).
    return YOLO(path)


# --------------------------- Helpers ---------------------------
def class_name(names, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, names.get(str(class_id), f"class_{class_id}")))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return f"class_{class_id}"


def rows_for_result(result, names, filename, inference_ms):
    rows = []
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return rows

    coords_list = boxes.xyxy.cpu().tolist()
    confidences = boxes.conf.cpu().tolist()
    class_ids = boxes.cls.cpu().tolist()

    for coords, conf, cls_id in zip(coords_list, confidences, class_ids):
        x1, y1, x2, y2 = [round(float(v), 1) for v in coords]
        rows.append({
            "Image": filename,
            "Class": class_name(names, int(cls_id)),
            "Confidence": round(float(conf), 4),
            "X1": x1,
            "Y1": y1,
            "X2": x2,
            "Y2": y2,
            "Inference time (ms)": round(float(inference_ms), 2),
        })
    return rows


def pil_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def run_batch(files, model, confidence, image_size):
    outputs = []
    all_detection_rows = []
    batch_start = time.perf_counter()

    progress = st.progress(0, text="Starting batch inference...")

    for idx, file in enumerate(files):
        input_image = Image.open(file).convert("RGB")

        start = time.perf_counter()
        predictions = model.predict(
            source=input_image,
            conf=confidence,
            imgsz=image_size,
            verbose=False,
        )
        inference_ms = (time.perf_counter() - start) * 1000

        result = predictions[0]
        # Ultralytics result.plot() returns a BGR array.
        annotated_bgr = result.plot()
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        annotated_image = Image.fromarray(annotated_rgb)

        detections = rows_for_result(
            result, model.names, file.name, inference_ms
        )

        # Keep one summary row per image, including images with zero detections.
        class_counts = {}
        for item in detections:
            class_counts[item["Class"]] = class_counts.get(item["Class"], 0) + 1

        outputs.append({
            "filename": file.name,
            "original": input_image.copy(),
            "annotated": annotated_image,
            "detections": detections,
            "count": len(detections),
            "class_counts": class_counts,
            "inference_ms": inference_ms,
        })
        all_detection_rows.extend(detections)
        progress.progress(
            (idx + 1) / len(files),
            text=f"Processed {idx + 1}/{len(files)}: {file.name}",
        )

    total_ms = (time.perf_counter() - batch_start) * 1000
    progress.empty()
    return outputs, all_detection_rows, total_ms


# --------------------------- Header ---------------------------
st.markdown(
    """
    <div class="hero">
      <h1>🍃 Tea Looper Detection</h1>
      <p>YOLOv12-SCCA research prototype </p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Model settings")
    st.caption(f"Checkpoint: `{MODEL_PATH.name}`")
    st.caption("The checkpoint must be compatible with the installed model code.")

control1, control2, control3 = st.columns([2.2, 1, 1])
with control1:
    uploaded_files = st.file_uploader(
        "Upload multiple tea-leaf images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="Upload up to 25 images. Larger batches are also supported.",
    )
with control2:
    confidence = st.slider(
        "Confidence threshold", 0.05, 0.95, 0.25, 0.05
    )
with control3:
    image_size = st.selectbox(
        "Inference image size", [320, 416, 512, 640, 768, 1024], index=3
    )

run_col, clear_col, _ = st.columns([1, 1, 4])
with run_col:
    run_clicked = st.button(
        "▶ Run detection",
        type="primary",
        use_container_width=True,
        disabled=not uploaded_files,
    )
with clear_col:
    clear_clicked = st.button("Clear results", use_container_width=True)

if clear_clicked:
    for key in ("batch_outputs", "detection_rows", "batch_total_ms", "run_settings"):
        st.session_state.pop(key, None)
    st.rerun()

if run_clicked and uploaded_files:
    try:
        model = load_model(str(MODEL_PATH))
        with st.spinner("Running detection on uploaded images..."):
            outputs, detection_rows, total_ms = run_batch(
                uploaded_files, model, confidence, image_size
            )
        st.session_state["batch_outputs"] = outputs
        st.session_state["detection_rows"] = detection_rows
        st.session_state["batch_total_ms"] = total_ms
        st.session_state["run_settings"] = {
            "confidence": confidence,
            "image_size": image_size,
            "model": str(MODEL_PATH),
        }
    except Exception as exc:
        st.error("Could not run inference. Check the model checkpoint and dependencies.")
        st.exception(exc)
        st.stop()


# --------------------------- Results dashboard ---------------------------
if "batch_outputs" not in st.session_state:
    st.info(
        "Upload your field images and click **Run detection**. "
        
    )
    st.stop()

outputs = st.session_state["batch_outputs"]
detection_rows = st.session_state["detection_rows"]
total_ms = st.session_state["batch_total_ms"]
settings = st.session_state["run_settings"]

num_images = len(outputs)
num_detections = sum(item["count"] for item in outputs)
images_with_detection = sum(item["count"] > 0 for item in outputs)
avg_ms = sum(item["inference_ms"] for item in outputs) / max(num_images, 1)

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Images processed", num_images)
m2.metric("Detections", num_detections)
m3.metric("Images with detections", images_with_detection)
m4.metric("Mean inference / image", f"{avg_ms:.1f} ms")
m5.metric("Total batch time", f"{total_ms / 1000:.2f} s")

st.caption(
    f"Checkpoint: {Path(settings['model']).name}  ·  "
    f"Confidence: {settings['confidence']:.2f}  ·  "
    f"Image size: {settings['image_size']} px"
)

st.divider()
st.subheader("Annotated prediction gallery")

# Render in rows of five. Exactly 25 images form a 5 × 5 gallery.
for row_start in range(0, num_images, GRID_COLUMNS):
    tile_columns = st.columns(GRID_COLUMNS, gap="small")
    for col_index, column in enumerate(tile_columns):
        item_index = row_start + col_index
        if item_index >= num_images:
            break

        item = outputs[item_index]
        with column:
            st.image(
                item["annotated"],
                use_container_width=True,
            )
            st.markdown(
                f"**{item['filename']}**",
                help=item["filename"],
            )
            st.caption(
                f"{item['count']} detection(s) · "
                f"{item['inference_ms']:.1f} ms"
            )
            if item["class_counts"]:
                class_summary = ", ".join(
                    f"{name}: {count}"
                    for name, count in item["class_counts"].items()
                )
                st.caption(class_summary)
            else:
                st.caption("No detections")

st.divider()
st.subheader("Per-image inference summary")

summary_rows = []
for item in outputs:
    summary_rows.append({
        "Image": item["filename"],
        "Detections": item["count"],
        "Minor": item["class_counts"].get("Minor", 0),
        "Moderate": item["class_counts"].get("Moderate", 0),
        "Severe": item["class_counts"].get("Severe", 0),
        "Inference time (ms)": round(item["inference_ms"], 2),
    })

summary_df = pd.DataFrame(summary_rows)
st.dataframe(summary_df, use_container_width=True, hide_index=True)

st.divider()
st.subheader("Prediction details table")

if detection_rows:
    details_df = pd.DataFrame(detection_rows)
    st.dataframe(details_df, use_container_width=True, hide_index=True)
    st.download_button(
        "Download prediction details (CSV)",
        data=details_df.to_csv(index=False).encode("utf-8"),
        file_name="teaguard_prediction_details.csv",
        mime="text/csv",
    )
else:
    details_df = pd.DataFrame(columns=[
        "Image", "Class", "Confidence", "X1", "Y1", "X2", "Y2",
        "Inference time (ms)"
    ])
    st.warning("No detections passed the selected confidence threshold.")

st.divider()
st.subheader("Download paper/demo outputs")

download_col1, download_col2 = st.columns(2)

with download_col1:
    summary_csv = summary_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download per-image summary (CSV)",
        data=summary_csv,
        file_name="teaguard_image_summary.csv",
        mime="text/csv",
        use_container_width=True,
    )

with download_col2:
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(
        zip_buffer, "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for item in outputs:
            output_name = Path(item["filename"]).stem + "_prediction.png"
            archive.writestr(output_name, pil_png_bytes(item["annotated"]))

        archive.writestr(
            "per_image_summary.csv",
            summary_df.to_csv(index=False).encode("utf-8"),
        )
        archive.writestr(
            "prediction_details.csv",
            details_df.to_csv(index=False).encode("utf-8"),
        )

    st.download_button(
        "Download all annotated images + tables (ZIP)",
        data=zip_buffer.getvalue(),
        file_name="teaguard_batch_outputs.zip",
        mime="application/zip",
        use_container_width=True,
    )

st.caption(
    "Inference times are measured locally and may vary with hardware, image size, "
    "batch composition, and system load. Detection counts are model predictions, "
    "not accuracy measurements. Report mAP, precision, and recall from a labelled test set."
)
