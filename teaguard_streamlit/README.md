# TeaGuard AI — Streamlit starter

## 1. Folder layout

Keep these files together:

```text
teaguard_streamlit/
├── app.py
├── requirements.txt
└── best.pt
```

Copy your trained checkpoint into this folder and name it `best.pt`.

## 2. Create and activate an environment (Windows PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For a CUDA-enabled PyTorch install, install the PyTorch build appropriate for your
CUDA/driver setup using the official PyTorch installation selector, then install
the remaining requirements. A plain `pip install -r requirements.txt` may install
a CPU build depending on the environment.

## 3. Run the app

```powershell
streamlit run app.py
```

Streamlit will print a local address, usually `http://localhost:8501`.

## 4. Important custom-model note

This starter assumes `best.pt` is compatible with the installed Ultralytics package.
If your YOLOv12-SCCA model uses custom layers, the Python files defining those layers
must be available and imported before `YOLO(MODEL_PATH)` is called. Use the same
compatible Ultralytics/repository version used to train the checkpoint.

Check that the checkpoint's class names are Minor, Moderate, Severe (or update the
names in your training configuration). The application displays names saved in the
checkpoint; it does not remap classes automatically.

## 5. Research integrity

The app displays per-image predictions and an approximate single-call inference
latency. It does not calculate mAP, precision, recall, or accuracy from an uploaded
unlabelled image. Calculate those metrics separately on your labelled held-out test set.
