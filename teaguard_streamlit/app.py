"""
TeaGuard AI — local Streamlit demo for tea-looper damage detection.

Assumptions:
- `best.pt` is an Ultralytics-compatible YOLO checkpoint.
- The model predicts the three classes Minor, Moderate, Severe.
- If the checkpoint uses custom SCCA layers, install/import the Python module
  that defines those layers before YOLO(MODEL_PATH) is called.
"""

from pathlib import Path
import time
import csv
import io

import streamlit as st
from PIL import Image
import pandas as pd
from ultralytics import YOLO

# ------------------------- Configuration -------------------------
APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "best.pt"

# Set this to your real class names if your checkpoint uses different labels.
EXPECTED_CLASSES = ["Minor", "Moderate", "Severe"]

st.set_page_config(
    page_title="Tea Looper Detection",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------- Styling -------------------------
st.markdown(
    """
    <style>
    .stApp { background: #f7faf7; }
    [data-testid="stSidebar"] { background: #edf5ee; }
    .hero {
        background: linear-gradient(120deg, #174c35, #2f7650);
        padding: 1.35rem 1.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 1rem;
    }
    .hero h1 { color: white; margin: 0; }
    .hero p { color: #e3f2e7; margin: .35rem 0 0 0; }
    .small-note { color: #637268; font-size: .88rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------- Model loading -------------------------
@st.cache_resource(show_spinner="Loading YOLO model...")
def load_model(model_path: str):
    """Load once and reuse the model between Streamlit interactions."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: {path}\n"
            "Copy your trained checkpoint into this app folder and name it best.pt, "
            "or update MODEL_PATH in app.py."
        )

    # If SCCA is implemented with custom Ultralytics modules, import the module
    # defining those layers above this line before calling YOLO(path).
    return YOLO(str(path))


# ------------------------- Helpers -------------------------
def get_class_name(names, class_id: int) -> str:
    """Handle both dict and list class-name formats."""
    if isinstance(names, dict):
        return str(names.get(class_id, names.get(str(class_id), f"class_{class_id}")))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return f"class_{class_id}"


def extract_detections(result, names):
    rows = []
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return rows

    xyxy = boxes.xyxy.cpu().tolist()
    confs = boxes.conf.cpu().tolist()
    classes = boxes.cls.cpu().tolist()

    for coords, conf, cls_id in zip(xyxy, confs, classes):
        x1, y1, x2, y2 = [round(float(v), 2) for v in coords]
        rows.append(
            {
                "Class": get_class_name(names, int(cls_id)),
                "Confidence": round(float(conf), 4),
                "X1": x1,
                "Y1": y1,
                "X2": x2,
                "Y2": y2,
            }
        )
    return rows


# ------------------------- Sidebar -------------------------
st.sidebar.markdown("## 🍃 TeaGuard AI")
st.sidebar.caption("Research prototype")
page = st.sidebar.radio(
    "Navigation",
    ["Tea Looper Detection", "About the Research"],
)
st.sidebar.divider()
st.sidebar.markdown("**Model checkpoint**")
st.sidebar.code(str(MODEL_PATH), language=None)
st.sidebar.caption("Local-only demo. Images are processed by this Python app.")

if page == "About the Research":
    st.markdown(
        """
        <div class="hero">
          <h1>About TeaGuard AI</h1>
          <p>YOLOv12 with Spatial–Channel Collaborative Attention (SCCA)</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.subheader("Research purpose")
    st.write(
        "This prototype provides an interface for testing a trained object-detection "
        "model on tea-leaf field images and inspecting its predicted bounding boxes, "
        "severity labels, confidence scores, and inference time."
    )
    st.subheader("Target classes")
    st.markdown("- **Minor** — leaf-damage severity class\n"
                "- **Moderate** — leaf-damage severity class\n"
                "- **Severe** — leaf-damage severity class")
    st.subheader("Methodology note")
    st.info(
        "Only report the model as YOLOv12-SCCA if the checkpoint was trained using "
        "that architecture. The app does not establish model accuracy by itself; "
        "quantitative metrics must be calculated against a labelled test set."
    )
    st.subheader("Suggested future additions")
    st.markdown(
        "- Held-out test-set evaluation and confusion matrix\n"
        "- Precision–recall curves and per-class AP\n"
        "- Baseline YOLOv12 versus YOLOv12-SCCA comparison\n"
        "- Model parameters, FLOPs, and repeatable latency measurements"
    )

else:
    st.markdown(
        """
        <div class="hero">
          <h1>Tea Looper Detection</h1>
          <p>Detect and inspect Minor, Moderate, and Severe leaf-damage predictions.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1, 1.15], gap="large")
    with left:
        st.subheader("1. Upload field image")
        uploaded_file = st.file_uploader(
            "Choose a tea-leaf image",
            type=["jpg", "jpeg", "png"],
            help="Supported formats: JPG, JPEG, PNG.",
        )
        confidence = st.slider(
            "Confidence threshold",
            min_value=0.05,
            max_value=0.95,
            value=0.25,
            step=0.05,
            help="Detections below this confidence threshold are filtered out.",
        )
        image_size = st.select_slider(
            "Inference image size",
            options=[320, 416, 512, 640, 768, 1024],
            value=640,
            help="Larger sizes may improve small-object detail but require more memory and time.",
        )
        run_button = st.button(
            "🔎 Run detection",
            type="primary",
            use_container_width=True,
            disabled=uploaded_file is None,
        )
        st.caption("For consistent scientific comparisons, use the same image size and settings.")

    with right:
        st.subheader("2. Detection output")
        if uploaded_file is None:
            st.info("Upload an image to preview it here.")
        else:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Input image", use_container_width=True)

    if run_button and uploaded_file is not None:
        try:
            model = load_model(str(MODEL_PATH))
            image = Image.open(uploaded_file).convert("RGB")

            with st.spinner("Running model inference..."):
                start = time.perf_counter()
                results = model.predict(
                    source=image,
                    conf=confidence,
                    imgsz=image_size,
                    verbose=False,
                )
                elapsed_ms = (time.perf_counter() - start) * 1000

            result = results[0]
            annotated_bgr = result.plot()
            # Ultralytics plot() returns a BGR NumPy array; convert to RGB for Streamlit.
            import cv2
            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

            rows = extract_detections(result, model.names)
            result_image = Image.fromarray(annotated_rgb)

            st.divider()
            st.subheader("Detection results")
            output_col, metrics_col = st.columns([1.5, 1])
            with output_col:
                st.image(result_image, caption="Annotated prediction", use_container_width=True)
                image_buffer = io.BytesIO()
                result_image.save(image_buffer, format="PNG")
                st.download_button(
                    "⬇️ Download annotated image",
                    data=image_buffer.getvalue(),
                    file_name="teaguard_detection.png",
                    mime="image/png",
                    use_container_width=True,
                )
            with metrics_col:
                st.metric("Detected instances", len(rows))
                st.metric("Inference time", f"{elapsed_ms:.1f} ms")
                st.metric("Confidence threshold", f"{confidence:.2f}")
                st.caption(
                    "Latency is measured for this local inference call and can vary "
                    "with hardware, image size, and system load."
                )

            if rows:
                df = pd.DataFrame(rows)
                st.subheader("Prediction table")
                st.dataframe(df, use_container_width=True, hide_index=True)
                csv_data = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download predictions (CSV)",
                    data=csv_data,
                    file_name="teaguard_predictions.csv",
                    mime="text/csv",
                )

                counts = df["Class"].value_counts().rename_axis("Class").reset_index(name="Count")
                st.subheader("Detected class counts")
                st.bar_chart(counts.set_index("Class"))
            else:
                st.warning(
                    "No detections passed the selected confidence threshold. "
                    "Try a lower threshold or inspect another image."
                )

            with st.expander("Model and inference details"):
                st.write("**Checkpoint:**", str(MODEL_PATH))
                st.write("**Model class names:**", model.names)
                st.write("**Image size:**", image_size)
                st.write("**Confidence threshold:**", confidence)
                st.write("**Device:**", getattr(model, "device", "managed by Ultralytics"))

        except Exception as exc:
            st.error("The model could not run. Check the checkpoint and model dependencies.")
            st.exception(exc)
            st.markdown(
                """
                **Common causes**
                - `best.pt` is not in the app folder or `MODEL_PATH` is incorrect.
                - The checkpoint depends on custom SCCA/YOLOv12 code that has not been imported.
                - The installed Ultralytics version differs from the one used for training.
                """
            )

    st.divider()
    st.caption(
        "Research prototype only. Predictions are not a substitute for expert field assessment. "
        "This page reports model outputs; it does not calculate test-set accuracy."
    )
