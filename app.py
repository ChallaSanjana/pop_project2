import os
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

LABELS = [
    "Cardiomegaly",
    "Emphysema",
    "Effusion",
    "Hernia",
    "Infiltration",
    "Mass",
    "Nodule",
    "Atelectasis",
    "Pneumothorax",
    "Pleural_Thickening",
    "Pneumonia",
    "Fibrosis",
    "Edema",
    "Consolidation",
]

IMG_SIZE = 320


@st.cache_resource(show_spinner=False)
def build_model(custom_weights_path: str | None = None) -> tf.keras.Model:
    """Builds DenseNet121-based multilabel classifier."""
    project_root = Path(__file__).resolve().parent
    base_weights = project_root / "densenet.hdf5"

    if not base_weights.exists():
        raise FileNotFoundError(
            "Missing densenet.hdf5 in project root. "
            "Extract densenet.7z first."
        )

    base_model = tf.keras.applications.DenseNet121(
        include_top=False,
        weights=str(base_weights),
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )

    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    outputs = tf.keras.layers.Dense(len(LABELS), activation="sigmoid")(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=outputs)

    if custom_weights_path:
        model.load_weights(custom_weights_path)

    return model


def preprocess_image(pil_image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    """Converts uploaded image to model-ready tensor and display array."""
    image_rgb = pil_image.convert("RGB")
    display_np = np.array(image_rgb)

    resized = image_rgb.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(resized).astype(np.float32)

    # Match notebook convention: sample-wise normalize.
    mean = arr.mean()
    std = arr.std() if arr.std() > 1e-6 else 1.0
    norm = (arr - mean) / std
    batch = np.expand_dims(norm, axis=0)
    return batch, display_np


def make_gradcam(model: tf.keras.Model, input_batch: np.ndarray, class_index: int) -> np.ndarray:
    """Generates a Grad-CAM heatmap for the chosen output class."""
    last_conv = model.get_layer("conv5_block16_concat")
    grad_model = tf.keras.Model([model.inputs], [last_conv.output, model.output])

    input_tensor = tf.convert_to_tensor(input_batch)
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(input_tensor)
        target = preds[:, class_index]

    grads = tape.gradient(target, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_out = conv_out[0]
    heatmap = tf.reduce_sum(conv_out * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    denom = tf.reduce_max(heatmap)
    if denom > 0:
        heatmap = heatmap / denom

    return heatmap.numpy()


def overlay_heatmap(display_image: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    """Resizes and overlays heatmap onto original uploaded image."""
    h, w = display_image.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_u8 = np.uint8(255 * heatmap_resized)
    color = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_MAGMA)

    # cv2 uses BGR, convert for proper blending/display.
    color_rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    overlaid = cv2.addWeighted(display_image, 0.65, color_rgb, 0.35, 0)
    return overlaid


def find_default_custom_weights() -> str | None:
    """Returns first known pretrained head weights path if present."""
    candidates = [
        "nih_new/pretrained_model.h5",
        "pretrained_model.h5",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def main() -> None:
    st.set_page_config(page_title="Chest X-Ray Diagnosis", layout="wide")
    st.title("Chest X-Ray Medical Diagnosis UI")
    st.write("Upload an X-ray image to get pathology probabilities and a Grad-CAM view.")

    with st.sidebar:
        st.header("Model Settings")
        default_custom = find_default_custom_weights() or ""
        custom_weights = st.text_input(
            "Custom classifier weights (.h5)",
            value=default_custom,
            help="Optional. If missing, predictions use only base DenseNet + untrained head.",
        ).strip()

        if not custom_weights:
            st.warning("No custom classifier weights set. Output scores will be non-clinical.")

        run_btn = st.button("Load / Reload Model", type="primary")

    # Load model on first render or explicit reload.
    if "model" not in st.session_state or run_btn:
        try:
            model = build_model(custom_weights if custom_weights else None)
            st.session_state["model"] = model
            st.success("Model ready.")
        except Exception as exc:
            st.session_state["model"] = None
            st.error(f"Model load failed: {exc}")

    uploaded = st.file_uploader("Upload X-ray image", type=["png", "jpg", "jpeg", "webp"])

    if uploaded is None:
        st.info("Upload an image to start.")
        return

    if st.session_state.get("model") is None:
        st.warning("Model is not loaded. Fix model setup in sidebar first.")
        return

    image = Image.open(uploaded)
    input_batch, display_np = preprocess_image(image)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Uploaded Image")
        st.image(display_np, use_container_width=True)

    with st.spinner("Running inference..."):
        model = st.session_state["model"]
        preds = model.predict(input_batch, verbose=0)[0]

    result_idx = np.argsort(preds)[::-1]
    top_k = st.slider("Top results", min_value=3, max_value=14, value=5)

    pred_rows = []
    for i in result_idx[:top_k]:
        pred_rows.append({"Pathology": LABELS[i], "Probability": float(preds[i])})

    with col2:
        st.subheader("Predictions")
        st.dataframe(pred_rows, use_container_width=True)

    class_for_cam = int(result_idx[0])
    with st.spinner("Generating Grad-CAM..."):
        heatmap = make_gradcam(model, input_batch, class_for_cam)
        overlaid = overlay_heatmap(display_np, heatmap)

    st.subheader(f"Grad-CAM for: {LABELS[class_for_cam]}")
    st.image(overlaid, use_container_width=True)


if __name__ == "__main__":
    main()
