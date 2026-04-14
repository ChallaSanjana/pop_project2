import hashlib
import io
import os
import json
import base64
import sqlite3
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import tensorflow as tf
from matplotlib.backends.backend_pdf import PdfPages
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
DB_PATH = Path(__file__).resolve().parent / "feedback.db"
CASES_PATH = Path(__file__).resolve().parent / "patient_cases.json"
SECTION_ORDER = ["Upload", "Predictions", "Explanation", "Feedback", "Report", "Patient History"]


def init_feedback_db() -> None:
    """Creates feedback table if it does not exist."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            image_hash TEXT NOT NULL,
            image_name TEXT,
            pathology TEXT NOT NULL,
            predicted_probability REAL NOT NULL,
            verdict TEXT NOT NULL,
            notes TEXT,
            age INTEGER,
            gender TEXT,
            symptoms TEXT,
            smoking_history TEXT
        )
        """
    )
    conn.commit()
    conn.close()


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


def inject_styles() -> None:
    """Adds app-wide dashboard styling."""
    st.markdown(
        """
        <style>
        .stApp {
            background: #f4f6f8;
            font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }

        [data-testid="stSidebar"] {
            background: #e5e7eb;
            border-right: 1px solid #d1d5db;
        }

        [data-testid="stSidebar"] * {
            color: #111827;
        }

        .sidebar-title {
            font-size: 1.15rem;
            font-weight: 800;
            margin-bottom: 0.2rem;
            color: #111827;
        }

        .sidebar-subtitle {
            font-size: 0.9rem;
            color: #4b5563;
            margin-bottom: 1rem;
        }

        [data-testid="stSidebar"] div[role="radiogroup"] {
            gap: 0.5rem;
        }

        [data-testid="stSidebar"] div[role="radiogroup"] label {
            padding: 0.7rem 0.9rem;
            border-radius: 12px;
            background: #f9fafb;
            border: 1px solid #d1d5db;
            transition: background 0.15s ease, border-color 0.15s ease, color 0.15s ease;
        }

        [data-testid="stSidebar"] div[role="radiogroup"] label:hover {
            background: #eef2ff;
            border-color: #c7d2fe;
        }

        [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
            background: #111827;
            border-color: #111827;
            color: #ffffff;
            box-shadow: 0 8px 20px rgba(17, 24, 39, 0.18);
        }

        [data-testid="stSidebar"] .stButton > button {
            width: 100%;
            border-radius: 12px;
            font-weight: 600;
            padding: 0.65rem 0.8rem;
        }

        [data-testid="stFileUploader"] {
            background: transparent;
            border: 0;
            padding: 0;
        }

        [data-testid="stFileUploader"] section {
            background: #ffffff;
            border: 1px solid #d8dee9;
            border-radius: 14px;
            padding: 0.25rem 0.5rem;
            min-height: 0;
        }

        [data-testid="stFileUploader"] button {
            border-radius: 999px;
        }

        .card {
            background: #ffffff;
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 22px;
            padding: 1.25rem 1.25rem 1.1rem 1.25rem;
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06);
            margin-bottom: 1rem;
        }

        .card h2, .card h3 {
            margin-top: 0;
        }

        .report-button a {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 0.6rem;
            width: 100%;
            padding: 1rem 1.2rem;
            border-radius: 16px;
            background: linear-gradient(90deg, #2563eb 0%, #14b8a6 100%);
            color: white !important;
            text-decoration: none;
            font-weight: 800;
            box-shadow: 0 12px 30px rgba(37, 99, 235, 0.25);
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }

        .report-button a:hover {
            transform: translateY(-1px);
            box-shadow: 0 16px 36px rgba(37, 99, 235, 0.35);
        }

        .feedback-note {
            color: #475569;
            font-size: 0.92rem;
            margin-top: 0.35rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


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


def image_to_base64(image_np: np.ndarray) -> str:
        """Encodes a numpy image array to base64 PNG for HTML rendering."""
        image_u8 = image_np.astype(np.uint8)
        image = Image.fromarray(image_u8)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")


def render_interactive_gradcam(image_np: np.ndarray, key: str, heatmap_np: np.ndarray | None = None) -> None:
        """Renders a hover-interactive Grad-CAM image with tooltip and highlight."""
        image_b64 = image_to_base64(image_np)
        heatmap_json = "null"

        if heatmap_np is not None:
                heatmap_small = cv2.resize(heatmap_np.astype(np.float32), (128, 128), interpolation=cv2.INTER_LINEAR)
                heatmap_json = json.dumps(heatmap_small.tolist())

        uid = f"gradcam_{hashlib.md5(key.encode('utf-8')).hexdigest()[:10]}"
        html = f"""
        <div id="{uid}_wrap" style="position:relative; width:100%; max-width:920px; margin:0 auto;">
            <img id="{uid}_img" src="data:image/png;base64,{image_b64}" style="width:100%; border-radius:12px; display:block;" />
            <div id="{uid}_highlight" style="position:absolute; width:72px; height:72px; border-radius:999px; border:1px solid rgba(14,165,233,0.55); background:rgba(14,165,233,0.14); pointer-events:none; display:none; transform:translate(-50%,-50%);"></div>
            <div id="{uid}_tooltip" style="position:absolute; pointer-events:none; background:rgba(15,23,42,0.92); color:#f8fafc; font-size:12px; padding:7px 10px; border-radius:8px; white-space:nowrap; display:none; box-shadow:0 8px 20px rgba(15,23,42,0.25);"></div>
        </div>
        <script>
            (function() {{
                const img = document.getElementById('{uid}_img');
                const tooltip = document.getElementById('{uid}_tooltip');
                const highlight = document.getElementById('{uid}_highlight');
                const heatmap = {heatmap_json};

                function activationAt(nx, ny) {{
                    if (!heatmap) return 0.5;
                    const h = heatmap.length;
                    const w = heatmap[0].length;
                    const x = Math.max(0, Math.min(w - 1, Math.floor(nx * (w - 1))));
                    const y = Math.max(0, Math.min(h - 1, Math.floor(ny * (h - 1))));
                    return Number(heatmap[y][x]) || 0;
                }}

                function tooltipText(score) {{
                    if (score >= 0.65) return 'High activation region';
                    if (score >= 0.35) return 'Model focused area';
                    return 'Lower activation region';
                }}

                img.addEventListener('mousemove', (e) => {{
                    const rect = img.getBoundingClientRect();
                    const x = e.clientX - rect.left;
                    const y = e.clientY - rect.top;
                    const nx = rect.width > 0 ? x / rect.width : 0.5;
                    const ny = rect.height > 0 ? y / rect.height : 0.5;

                    tooltip.textContent = tooltipText(activationAt(nx, ny));
                    tooltip.style.display = 'block';
                    highlight.style.display = 'block';
                    tooltip.style.left = (x + 16) + 'px';
                    tooltip.style.top = (y + 16) + 'px';
                    highlight.style.left = x + 'px';
                    highlight.style.top = y + 'px';
                }});

                img.addEventListener('mouseleave', () => {{
                    tooltip.style.display = 'none';
                    highlight.style.display = 'none';
                }});
            }})();
        </script>
        """

        height = int(max(320, min(760, image_np.shape[0] * 760 / max(image_np.shape[1], 1))))
        components.html(html, height=height, scrolling=False)


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


def hash_bytes(content: bytes) -> str:
    """Creates a stable ID for uploaded file content."""
    return hashlib.sha256(content).hexdigest()


def infer_confidence(probability: float) -> str:
    if probability >= 0.75:
        return "High"
    if probability >= 0.45:
        return "Moderate"
    return "Low"


def describe_activation_region(heatmap: np.ndarray) -> tuple[str, float]:
    """Summarizes where activation is strongest in Grad-CAM heatmap."""
    if np.max(heatmap) <= 0:
        return "diffuse lung region", 0.0

    h, w = heatmap.shape
    ys, xs = np.indices((h, w))
    weights = heatmap + 1e-8
    cx = float(np.sum(xs * weights) / np.sum(weights))
    cy = float(np.sum(ys * weights) / np.sum(weights))

    vertical = "upper" if cy < h / 3 else ("middle" if cy < 2 * h / 3 else "lower")
    horizontal = "left" if cx < w / 2 else "right"
    region = f"{vertical} {horizontal} lung region"
    strength = float(np.max(heatmap))
    return region, strength


def explanation_text(pathology: str, probability: float, heatmap: np.ndarray) -> str:
    conf = infer_confidence(probability)
    region, strength = describe_activation_region(heatmap)
    strength_desc = "strong" if strength >= 0.75 else ("moderate" if strength >= 0.45 else "faint")
    return (
        f"{conf} confidence for {pathology} due to {strength_desc} activation "
        f"in the {region}."
    )


def save_feedback(
    image_hash: str,
    image_name: str,
    pathology: str,
    probability: float,
    verdict: str,
    notes: str,
    patient_info: dict,
) -> None:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO feedback (
            created_at, image_hash, image_name, pathology,
            predicted_probability, verdict, notes,
            age, gender, symptoms, smoking_history
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.utcnow().isoformat(timespec="seconds"),
            image_hash,
            image_name,
            pathology,
            float(probability),
            verdict,
            notes,
            patient_info.get("age"),
            patient_info.get("gender"),
            patient_info.get("symptoms"),
            patient_info.get("smoking_history"),
        ),
    )
    conn.commit()
    conn.close()


def load_feedback(image_hash: str) -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(
        """
        SELECT created_at, pathology, predicted_probability, verdict, notes
        FROM feedback
        WHERE image_hash = ?
        ORDER BY id DESC
        """,
        (image_hash,),
    )
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def get_active_section() -> str:
    """Reads the current section from the query string or session state."""
    default_section = st.session_state.get("active_section", "Upload")
    try:
        section = st.query_params.get("section", default_section)
    except Exception:
        params = st.experimental_get_query_params()
        section = params.get("section", [default_section])[0]

    if isinstance(section, list):
        section = section[0] if section else default_section

    if section not in SECTION_ORDER:
        section = default_section if default_section in SECTION_ORDER else "Upload"

    st.session_state["active_section"] = section
    return section


def set_active_section(section: str) -> None:
    """Switches active section without losing Streamlit session state."""
    st.session_state["active_section"] = section
    try:
        st.query_params["section"] = section
    except Exception:
        st.experimental_set_query_params(section=section)
    st.rerun()


def is_patient_info_complete(patient_info: dict | None) -> bool:
    """Checks if required patient details are present."""
    if not patient_info:
        return False
    return all(
        [
            bool(str(patient_info.get("name", "")).strip()),
            patient_info.get("age") is not None,
            bool(str(patient_info.get("gender", "")).strip()),
            bool(str(patient_info.get("smoking_history", "")).strip()),
            bool(str(patient_info.get("symptoms", "")).strip()),
        ]
    )


def ensure_history_state() -> None:
    """Initializes session-backed patient case history."""
    if "patient_cases" not in st.session_state:
        st.session_state["patient_cases"] = load_cases_from_disk()
    if "selected_case_id" not in st.session_state:
        st.session_state["selected_case_id"] = None
    if "upload_widget_key" not in st.session_state:
        st.session_state["upload_widget_key"] = 0


def load_cases_from_disk() -> list[dict]:
    """Loads persisted cases from disk at app start."""
    if not CASES_PATH.exists():
        return []

    try:
        with CASES_PATH.open("r", encoding="utf-8") as handle:
            raw_cases = json.load(handle)
    except Exception:
        return []

    cases: list[dict] = []
    for case in raw_cases if isinstance(raw_cases, list) else []:
        try:
            cases.append(
                {
                    **case,
                    "image_display": np.array(case["image_display"], dtype=np.uint8),
                    "thumbnail": np.array(case["thumbnail"], dtype=np.uint8),
                    "top_overlay": np.array(case["top_overlay"], dtype=np.uint8),
                    "overlays": {
                        key: np.array(value, dtype=np.uint8)
                        for key, value in case.get("overlays", {}).items()
                    },
                    "heatmaps": {
                        key: np.array(value, dtype=np.float32)
                        for key, value in case.get("heatmaps", {}).items()
                    },
                }
            )
        except Exception:
            continue

    return cases


def save_cases_to_disk() -> None:
    """Persists cases to disk whenever a new case is added or updated."""
    serializable_cases: list[dict] = []
    for case in st.session_state.get("patient_cases", []):
        serializable_cases.append(
            {
                **case,
                "image_display": case["image_display"].tolist(),
                "thumbnail": case["thumbnail"].tolist(),
                "top_overlay": case["top_overlay"].tolist(),
                "overlays": {key: value.tolist() for key, value in case.get("overlays", {}).items()},
                "heatmaps": {key: value.tolist() for key, value in case.get("heatmaps", {}).items()},
            }
        )

    with CASES_PATH.open("w", encoding="utf-8") as handle:
        json.dump(serializable_cases, handle)


def reset_current_case() -> None:
    """Clears the active case and returns the app to its initial upload state."""
    for key in [
        "patient_info",
        "uploaded_bytes",
        "image_name",
        "image_display",
        "image_hash",
        "upload_context",
        "selected_case_id",
    ]:
        st.session_state.pop(key, None)

    st.session_state["active_section"] = "Upload"
    st.session_state["upload_widget_key"] = st.session_state.get("upload_widget_key", 0) + 1

    try:
        st.query_params["section"] = "Upload"
    except Exception:
        st.experimental_set_query_params(section="Upload")


def upsert_patient_case(
    image_name: str,
    image_hash: str,
    patient_info: dict,
    display_np: np.ndarray,
    pred_rows: list[dict],
    explain_rows: list[dict],
    overlays: dict[str, np.ndarray],
    heatmaps: dict[str, np.ndarray],
    top_overlay: np.ndarray,
) -> str:
    """Stores or updates a completed case in session history."""
    cases: list[dict] = st.session_state["patient_cases"]
    patient_name = str(patient_info.get("name", "Unknown")).strip() or "Unknown"
    case_id = f"{image_hash}:{patient_name.lower()}"
    thumbnail = cv2.resize(display_np, (120, 120), interpolation=cv2.INTER_AREA)

    case_payload = {
        "case_id": case_id,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "patient_name": patient_name,
        "patient_info": dict(patient_info),
        "image_name": image_name,
        "image_hash": image_hash,
        "image_display": display_np.copy(),
        "thumbnail": thumbnail,
        "predictions": [dict(row) for row in pred_rows],
        "explanations": [dict(row) for row in explain_rows],
        "overlays": {k: v.copy() for k, v in overlays.items()},
        "heatmaps": {k: v.copy() for k, v in heatmaps.items()},
        "top_overlay": top_overlay.copy(),
    }

    idx = next((i for i, c in enumerate(cases) if c.get("case_id") == case_id), None)
    if idx is None:
        cases.insert(0, case_payload)
    else:
        cases[idx] = case_payload

    st.session_state["patient_cases"] = cases
    save_cases_to_disk()
    if st.session_state.get("selected_case_id") is None:
        st.session_state["selected_case_id"] = case_id
    return case_id


def get_selected_case() -> dict | None:
    """Returns currently selected case from history if available."""
    selected = st.session_state.get("selected_case_id")
    for case in st.session_state.get("patient_cases", []):
        if case.get("case_id") == selected:
            return case
    return None


def build_report_pdf(
    image_name: str,
    uploaded_image: np.ndarray,
    top_overlay: np.ndarray,
    prediction_df: pd.DataFrame,
    explanation_df: pd.DataFrame,
    patient_info: dict,
    feedback_rows: list[dict],
) -> bytes:
    """Creates report PDF in memory and returns bytes."""
    buf = io.BytesIO()

    with PdfPages(buf) as pdf:
        # Page 1: metadata + predictions + explanations
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.suptitle("Chest X-Ray AI Second Opinion Report", fontsize=16, y=0.98)

        patient_lines = [
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Image: {image_name}",
            f"Age: {patient_info.get('age', 'N/A')}",
            f"Gender: {patient_info.get('gender', 'N/A')}",
            f"Symptoms: {patient_info.get('symptoms', 'N/A')}",
            f"Smoking History: {patient_info.get('smoking_history', 'N/A')}",
        ]
        fig.text(0.07, 0.90, "Patient Details", fontsize=12, fontweight="bold")
        fig.text(0.07, 0.84, "\n".join(patient_lines), fontsize=10, va="top")

        fig.text(0.07, 0.66, "Predictions", fontsize=12, fontweight="bold")
        pred_text = prediction_df.to_string(index=False)
        fig.text(0.07, 0.64, pred_text, family="monospace", fontsize=9, va="top")

        fig.text(0.07, 0.42, "Explanations", fontsize=12, fontweight="bold")
        explain_text = explanation_df.to_string(index=False)
        fig.text(0.07, 0.40, explain_text, family="monospace", fontsize=8.5, va="top")

        feedback_title_y = 0.14
        fig.text(0.07, feedback_title_y, "Doctor Feedback", fontsize=12, fontweight="bold")
        if feedback_rows:
            fb_df = pd.DataFrame(feedback_rows)
            fb_text = fb_df.head(8).to_string(index=False)
            fig.text(0.07, feedback_title_y - 0.02, fb_text, family="monospace", fontsize=8, va="top")
        else:
            fig.text(0.07, feedback_title_y - 0.03, "No feedback submitted yet.", fontsize=10)

        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 2: original image + Grad-CAM
        fig2, axes = plt.subplots(1, 2, figsize=(11, 5))
        axes[0].imshow(uploaded_image)
        axes[0].set_title("Uploaded X-ray")
        axes[0].axis("off")

        axes[1].imshow(top_overlay)
        axes[1].set_title("Grad-CAM (Top Prediction)")
        axes[1].axis("off")

        fig2.tight_layout()
        pdf.savefig(fig2)
        plt.close(fig2)

    buf.seek(0)
    return buf.read()


def main() -> None:
    st.set_page_config(page_title="AI Second Opinion System", layout="wide")
    inject_styles()
    init_feedback_db()
    ensure_history_state()

    active_section = get_active_section()

    with st.sidebar:
        st.markdown('<div class="sidebar-title">Chest X-ray Second Opinion</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="sidebar-subtitle">AI-assisted workflow with upload, explanation, feedback, and report.</div>',
            unsafe_allow_html=True,
        )

        st.markdown("### Navigation")
        for section in SECTION_ORDER:
            button_type = "primary" if section == active_section else "secondary"
            if st.button(section, key=f"nav_{section}", type=button_type):
                set_active_section(section)

        st.markdown("---")
        if st.button("New Case", key="new_case_button", type="secondary", use_container_width=True):
            reset_current_case()

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
        except Exception as exc:
            st.session_state["model"] = None
            st.error(f"Model load failed: {exc}")

    st.title("AI Second Opinion System: Chest X-ray Medical Diagnosis")
    header_left, header_right = st.columns([4, 1])
    with header_right:
        st.button(
            "New Case",
            key="new_case_top_button",
            type="primary",
            use_container_width=True,
            on_click=reset_current_case,
        )

    if active_section == "Upload":
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Patient Details")

        existing_patient = st.session_state.get("patient_info", {})
        default_name = existing_patient.get("name", "")
        default_age = int(existing_patient.get("age", 45)) if existing_patient.get("age") is not None else 45
        default_gender = existing_patient.get("gender", "Female")
        default_smoking = existing_patient.get("smoking_history", "Never")
        default_symptoms = existing_patient.get("symptoms", "")

        gender_options = ["Female", "Male", "Other", "Prefer not to say"]
        smoking_options = ["Never", "Former smoker", "Current smoker", "Unknown"]

        with st.form("upload_patient_info_form"):
            c1, c2 = st.columns(2)
            with c1:
                patient_name = st.text_input("Patient Name", value=default_name, placeholder="Enter patient full name")
                age = st.number_input("Age", min_value=0, max_value=120, value=default_age, step=1)
                gender = st.selectbox(
                    "Gender",
                    gender_options,
                    index=gender_options.index(default_gender) if default_gender in gender_options else 0,
                )
            with c2:
                smoking_history = st.selectbox(
                    "Smoking History",
                    smoking_options,
                    index=smoking_options.index(default_smoking) if default_smoking in smoking_options else 0,
                )
                symptoms = st.text_area(
                    "Symptoms",
                    value=default_symptoms,
                    placeholder="e.g., cough, shortness of breath, chest pain",
                )

            submitted_patient = st.form_submit_button("Save Patient Details")

        if submitted_patient:
            cleaned_name = patient_name.strip()
            cleaned_symptoms = symptoms.strip()
            if not cleaned_name:
                st.error("Patient name is required before upload.")
            elif not cleaned_symptoms:
                st.error("Symptoms are required before upload.")
            else:
                st.session_state["patient_info"] = {
                    "name": cleaned_name,
                    "age": int(age),
                    "gender": gender,
                    "symptoms": cleaned_symptoms,
                    "smoking_history": smoking_history,
                }
                st.success("Patient details saved.")

        patient_ready = is_patient_info_complete(st.session_state.get("patient_info"))
        st.markdown("---")
        st.subheader("Upload")
        if not patient_ready:
            st.info("Fill and save patient details above to enable image upload.")

        uploaded = st.file_uploader(
            "Upload X-ray image",
            type=["png", "jpg", "jpeg", "webp"],
            label_visibility="visible",
            disabled=not patient_ready,
            key=f"upload_widget_{st.session_state['upload_widget_key']}",
        )
        if uploaded is not None and patient_ready:
            uploaded_bytes = uploaded.getvalue()
            image = Image.open(io.BytesIO(uploaded_bytes))
            display_np = np.array(image.convert("RGB"))
            st.session_state["uploaded_bytes"] = uploaded_bytes
            st.session_state["image_name"] = uploaded.name
            st.session_state["image_display"] = display_np
            st.session_state["image_hash"] = hash_bytes(uploaded_bytes)
            st.session_state["upload_context"] = {
                "image_hash": st.session_state["image_hash"],
                "image_name": uploaded.name,
                "patient_info": dict(st.session_state.get("patient_info", {})),
            }
            st.success("Image uploaded and ready.")
            st.image(display_np, caption="Uploaded X-ray", use_container_width=True)

            if st.session_state.get("model") is not None:
                st.session_state["active_section"] = "Predictions"
                try:
                    st.query_params["section"] = "Predictions"
                except Exception:
                    st.experimental_set_query_params(section="Predictions")
                st.rerun()
            else:
                st.warning("Image uploaded. Load model in sidebar to run predictions.")
        elif st.session_state.get("uploaded_bytes") is not None:
            st.info("A file is already loaded. Switch to another section to continue.")
            st.image(st.session_state["image_display"], caption="Uploaded X-ray", use_container_width=True)
        else:
            st.info("Upload an image to begin.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    if active_section == "Patient History":
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Patient History")

        cases = st.session_state.get("patient_cases", [])
        if not cases:
            st.info("No completed cases yet. Upload and run a prediction to add history.")
            st.markdown("</div>", unsafe_allow_html=True)
            return

        left_col, right_col = st.columns([1, 2], gap="large")
        with left_col:
            st.markdown("### Cases")
            for idx, case in enumerate(cases):
                c1, c2 = st.columns([1, 2])
                with c1:
                    st.image(case["thumbnail"], use_container_width=True)
                with c2:
                    st.write(f"**{case['patient_name']}**")
                    st.caption(case["created_at"])
                    if st.button("Open", key=f"history_open_{idx}"):
                        st.session_state["selected_case_id"] = case["case_id"]
                        st.rerun()

        with right_col:
            selected_case = get_selected_case()
            if selected_case is None:
                st.info("Click Open on a patient to view details.")
            else:
                st.markdown(f"### {selected_case['patient_name']}")
                st.caption(f"Image: {selected_case['image_name']} | {selected_case['created_at']}")

                pinfo = selected_case.get("patient_info", {})
                st.write(
                    f"Age: {pinfo.get('age', 'N/A')} | "
                    f"Gender: {pinfo.get('gender', 'N/A')} | "
                    f"Smoking: {pinfo.get('smoking_history', 'N/A')}"
                )
                st.write(f"Symptoms: {pinfo.get('symptoms', 'N/A')}")

                img_col, pred_col = st.columns([1, 1], gap="large")
                with img_col:
                    st.image(selected_case["image_display"], caption="Uploaded X-ray", use_container_width=True)
                    st.image(selected_case["top_overlay"], caption="Top Grad-CAM", use_container_width=True)

                with pred_col:
                    st.dataframe(pd.DataFrame(selected_case.get("predictions", [])), use_container_width=True)

                explanations = selected_case.get("explanations", [])
                overlays_map = selected_case.get("overlays", {})
                heatmaps_map = selected_case.get("heatmaps", {})
                if explanations:
                    st.markdown("### Explanations")
                    tabs = st.tabs([row["Pathology"] for row in explanations])
                    for tab, row in zip(tabs, explanations):
                        with tab:
                            st.write(row["Reason"])
                            overlay = overlays_map.get(row["Pathology"])
                            if overlay is not None:
                                render_interactive_gradcam(
                                    overlay,
                                    key=f"history_{selected_case['case_id']}_{row['Pathology']}",
                                    heatmap_np=heatmaps_map.get(row["Pathology"]),
                                )

                history_feedback = load_feedback(selected_case["image_hash"])
                history_report = build_report_pdf(
                    image_name=selected_case["image_name"],
                    uploaded_image=selected_case["image_display"],
                    top_overlay=selected_case["top_overlay"],
                    prediction_df=pd.DataFrame(selected_case.get("predictions", [])),
                    explanation_df=pd.DataFrame(selected_case.get("explanations", [])),
                    patient_info=selected_case.get("patient_info", {}),
                    feedback_rows=history_feedback,
                )
                st.download_button(
                    "Download Case Report",
                    data=history_report,
                    file_name=f"history_report_{Path(selected_case['image_name']).stem}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    key="download_history_case_report",
                )

        st.markdown("</div>", unsafe_allow_html=True)
        return

    uploaded_bytes = st.session_state.get("uploaded_bytes")
    if not uploaded_bytes:
        st.info("Upload an image first from the Upload section.")
        return

    upload_context = st.session_state.get("upload_context", {})
    current_patient_info = upload_context.get("patient_info") or st.session_state.get("patient_info", {})
    if not is_patient_info_complete(current_patient_info):
        st.info("Fill patient details in Upload before running predictions.")
        return

    if st.session_state.get("model") is None:
        st.warning("Model is not loaded. Fix model setup in sidebar first.")
        return

    image_name = upload_context.get("image_name") or st.session_state.get("image_name", "uploaded_image")
    image_hash = upload_context.get("image_hash") or st.session_state.get("image_hash") or hash_bytes(uploaded_bytes)
    display_np = st.session_state.get("image_display")
    if display_np is None:
        image = Image.open(io.BytesIO(uploaded_bytes))
        input_batch, display_np = preprocess_image(image)
    else:
        image = Image.open(io.BytesIO(uploaded_bytes))
        input_batch, _ = preprocess_image(image)

    model = st.session_state["model"]
    preds = model.predict(input_batch, verbose=0)[0]
    result_idx = np.argsort(preds)[::-1]
    top_k = 5
    pred_rows = [{"Pathology": LABELS[i], "Probability": float(preds[i])} for i in result_idx[:top_k]]
    pred_df = pd.DataFrame(pred_rows)

    top_class_idx = int(result_idx[0])
    top_overlay = None
    overlays: dict[str, np.ndarray] = {}
    heatmaps: dict[str, np.ndarray] = {}
    explain_rows = []
    for i in result_idx[:3]:
        pathology = LABELS[int(i)]
        prob = float(preds[int(i)])
        heatmap = make_gradcam(model, input_batch, int(i))
        heatmaps[pathology] = cv2.resize(
            heatmap.astype(np.float32),
            (display_np.shape[1], display_np.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        overlaid = overlay_heatmap(display_np, heatmap)
        overlays[pathology] = overlaid
        if int(i) == top_class_idx:
            top_overlay = overlaid
        explain_rows.append(
            {
                "Pathology": pathology,
                "Probability": round(prob, 4),
                "Confidence": infer_confidence(prob),
                "Reason": explanation_text(pathology, prob, heatmap),
            }
        )

    explain_df = pd.DataFrame(explain_rows)
    if top_overlay is None:
        top_overlay = overlay_heatmap(display_np, make_gradcam(model, input_batch, top_class_idx))

    current_case_id = upsert_patient_case(
        image_name=image_name,
        image_hash=image_hash,
        patient_info=current_patient_info,
        display_np=display_np,
        pred_rows=pred_rows,
        explain_rows=explain_rows,
        overlays=overlays,
        heatmaps=heatmaps,
        top_overlay=top_overlay,
    )
    st.session_state["selected_case_id"] = current_case_id

    feedback_rows = load_feedback(image_hash)

    if active_section == "Predictions":
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Predictions")
        col1, col2 = st.columns([1.1, 1.3], gap="large")
        with col1:
            st.image(display_np, caption="Uploaded X-ray", use_container_width=True)
        with col2:
            st.dataframe(pred_df, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        return

    if active_section == "Explanation":
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Explanation")
        st.dataframe(explain_df, use_container_width=True)
        if explain_rows:
            tabs = st.tabs([row["Pathology"] for row in explain_rows])
            for tab, row in zip(tabs, explain_rows):
                with tab:
                    st.write(row["Reason"])
                    pathology = row["Pathology"]
                    render_interactive_gradcam(
                        overlays[pathology],
                        key=f"explain_{image_hash}_{pathology}",
                        heatmap_np=heatmaps.get(pathology),
                    )
        st.markdown("</div>", unsafe_allow_html=True)
        return

    if active_section == "Feedback":
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Doctor Feedback")
        st.markdown(
            '<div class="feedback-note">Mark whether the top prediction is correct. This is stored locally in feedback.db.</div>',
            unsafe_allow_html=True,
        )
        feedback_choices = [f"{r['Pathology']} ({r['Probability']:.2%})" for r in pred_rows]
        with st.form("feedback_form"):
            selected = st.selectbox("Prediction to review", feedback_choices)
            verdict = st.radio(
                "Doctor verdict",
                ["Correct", "Incorrect", "Uncertain"],
                horizontal=True,
            )
            notes = st.text_area("Doctor notes", placeholder="Optional notes")
            submit_feedback = st.form_submit_button("Save Feedback")

        if submit_feedback and feedback_choices:
            pathology = selected.split(" (")[0]
            prob = float(pred_df.loc[pred_df["Pathology"] == pathology, "Probability"].iloc[0])
            save_feedback(
                image_hash=image_hash,
                image_name=image_name,
                pathology=pathology,
                probability=prob,
                verdict=verdict,
                notes=notes.strip(),
                patient_info=current_patient_info,
            )
            st.success("Feedback saved.")

        if feedback_rows:
            st.dataframe(pd.DataFrame(feedback_rows), use_container_width=True)
        else:
            st.info("No feedback submitted for this image yet.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    if active_section == "Report":
        st.markdown('<div class="card" style="text-align:center;">', unsafe_allow_html=True)
        st.subheader("Report")
        report_bytes = build_report_pdf(
            image_name=image_name,
            uploaded_image=display_np,
            top_overlay=top_overlay,
            prediction_df=pred_df,
            explanation_df=explain_df,
            patient_info=current_patient_info,
            feedback_rows=feedback_rows,
        )
        st.download_button(
            "Download PDF Report",
            data=report_bytes,
            file_name=f"second_opinion_report_{Path(image_name).stem}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
        return

if __name__ == "__main__":
    main()
