"""
St. Xavier's College Campus Location Recognizer Dashboard
Custom-styled Streamlit app with DINOv2 visual place recognition.
"""

import base64
import sys
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import yaml
from PIL import Image

# Resolve repository root
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.dataset import build_label_maps
from src.data.transforms import get_eval_transforms
from src.inference.confidence import classify_with_confidence
from src.models.classifier_head import MLPClassifierHead
from src.models.dinov2_extractor import DINOv2Extractor

DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "config.yaml"

# File paths in the scripts folder
SCRIPT_DIR = Path(__file__).resolve().parent
BACKGROUND_IMG_PATH = SCRIPT_DIR / "maincol.jpeg"
LOGO_PATH = SCRIPT_DIR / "logo.png"

st.set_page_config(
    page_title="Campus Location Recognizer - St. Xavier's College",
    page_icon="📍",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# --- BACKGROUND & STYLING INJECTOR ---
def set_styled_theme(image_path: Path):
    bg_css = ""
    if image_path.exists():
        with open(image_path, "rb") as f:
            data = f.read()
        bin_str = base64.b64encode(data).decode()

        bg_css = f"""
        /* Fullscreen Soft-Blurred Background Layer */
        .stApp::before {{
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            background-image: url("data:image/jpeg;base64,{bin_str}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            filter: blur(4px) brightness(0.85);
            transform: scale(1.02);
            z-index: -1;
        }}
        """

    css = f"""
    <style>
    {bg_css}
    
    /* Transparent Canvas */
    .stApp {{
        background: transparent !important;
    }}
    
    /* Card Layouts */
    .white-card {{
        background-color: #ffffff !important;
        border-radius: 12px;
        padding: 24px;
        border: 1px solid #cbd5e1;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.25);
        margin-bottom: 20px;
    }}

    /* Text Colors inside White Cards */
    .white-card h1, .white-card h2, .white-card h3, .white-card h4 {{
        color: #0f172a !important;
        margin-top: 0 !important;
    }}

    .white-card p, .white-card span, .white-card label {{
        color: #334155 !important;
    }}

    /* Override Streamlit Info/Alert Box Transparency */
    div[data-testid="stAlert"] {{
        background-color: #f0f9ff !important;
        border: 1px solid #bae6fd !important;
        color: #0369a1 !important;
        border-radius: 8px !important;
        opacity: 1 !important;
    }}
    div[data-testid="stAlert"] p {{
        color: #0369a1 !important;
        font-weight: 500 !important;
    }}

    /* Prediction Status Cards */
    .status-card-success {{
        background-color: #d1e7dd !important;
        border: 1px solid #a3cfbb !important;
        color: #0f5132 !important;
        padding: 16px;
        border-radius: 10px;
        margin-bottom: 15px;
    }}
    .status-card-warning {{
        background-color: #fff3cd !important;
        border: 1px solid #ffe69c !important;
        color: #664d03 !important;
        padding: 16px;
        border-radius: 10px;
        margin-bottom: 15px;
    }}
    .prediction-card {{
        background-color: #f8fafc !important;
        border-radius: 10px;
        padding: 16px;
        border: 1px solid #cbd5e1 !important;
        margin-bottom: 15px;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# Apply theme
set_styled_theme(BACKGROUND_IMG_PATH)


# Model Loader
@st.cache_resource
def load_localization_system(config_path: str = str(DEFAULT_CONFIG_PATH)):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metadata_csv = REPO_ROOT / cfg["data"]["metadata_csv"]
    checkpoints_dir = REPO_ROOT / cfg["paths"]["checkpoints_dir"]

    location_to_idx, _ = build_label_maps(str(metadata_csv))
    idx_to_location = {v: k for k, v in location_to_idx.items()}

    backbone = (
        DINOv2Extractor(cfg["model"]["dinov2_variant"], freeze=True)
        .to(device)
        .eval()
    )
    head = MLPClassifierHead(
        cfg["model"]["embedding_dim"], len(location_to_idx)
    ).to(device)

    checkpoint_path = checkpoints_dir / "dinov2_frozen_head_best.pt"
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=True
    )
    head.load_state_dict(checkpoint)
    head.eval()

    return cfg, device, backbone, head, idx_to_location


cfg, device, backbone, head, idx_to_location = load_localization_system()

# --- HEADER BLOCK (MODERN SIDE-BY-SIDE HEADER) ---
logo_base64 = ""
if LOGO_PATH.exists():
    with open(LOGO_PATH, "rb") as f:
        logo_base64 = base64.b64encode(f.read()).decode()

logo_img_tag = (
    f'<img src="data:image/png;base64,{logo_base64}" style="width: 75px; height: auto; object-fit: contain;" />'
    if logo_base64
    else ""
)

st.markdown(
    f"""
<div class="white-card" style="display: flex; align-items: center; gap: 20px; padding: 20px 24px;">
    <div style="flex-shrink: 0; text-align: center;">
        {logo_img_tag}
    </div>
    <div style="border-left: 1px solid #cbd5e1; padding-left: 20px; flex-grow: 1;">
        <p style="margin: 0; color: #64748b; font-weight: 600; font-size: 0.8rem; letter-spacing: 0.5px;">ST. XAVIER'S COLLEGE • MUMBAI</p>
        <h1 style="margin: 4px 0 2px 0; font-size: 1.7rem; color: #0f172a;">📍 Campus Location Recognizer</h1>
        <p style="margin: 0; font-size: 0.95rem; color: #334155;">
            Upload a photo taken on campus — the model guesses where it was taken.
        </p>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# --- 2-COLUMN MAIN CANVAS ---
col_left, col_right = st.columns([1.1, 1], gap="large")

with col_left:
    st.markdown(
        """
    <div class="white-card">
        <h3>📷 Upload an image</h3>
    </div>
    """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Supported formats: JPG, PNG, JPEG", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Query Photo", use_container_width=True)

        with st.spinner("Processing image..."):
            transform = get_eval_transforms(cfg["data"]["image_size"])
            tensor = (
                transform(image=np.array(image))["image"]
                .unsqueeze(0)
                .to(device)
            )

            with torch.no_regret if hasattr(torch, "no_regret") else torch.no_grad():
                embedding = backbone(tensor)
                logits = head(embedding)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

            pred_idx, confidence, is_confident = classify_with_confidence(
                probs,
                softmax_threshold=cfg["confidence"]["softmax_threshold"],
                margin_threshold=cfg["confidence"]["margin_threshold"],
            )
            predicted_location = idx_to_location[pred_idx]

with col_right:
    st.markdown(
        """
    <div class="white-card">
        <h3>📊 Prediction Results</h3>
    </div>
    """,
        unsafe_allow_html=True,
    )

    if uploaded_file and "predicted_location" in locals():
        if is_confident:
            st.markdown(
                f"""
                <div class="status-card-success">
                    <h4 style="margin:0;">✅ Confirmed Match</h4>
                    <h2 style="margin:4px 0 0 0; color:#0f5132;">{predicted_location.replace('_', ' ').title()}</h2>
                </div>
            """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="status-card-warning">
                    <h4 style="margin:0;">⚠️ Uncertain Prediction</h4>
                    <p style="margin:4px 0 0 0;">
                        Best guess is <b>{predicted_location.replace('_', ' ').title()}</b> ({confidence:.0%} confidence).
                    </p>
                </div>
            """,
                unsafe_allow_html=True,
            )

        st.markdown(
            f"""
            <div class="prediction-card">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <span style="color:#666; font-size:0.9em;">Top Location</span>
                        <h2 style="margin:0; color:#8c2d4f;">{predicted_location.replace('_', ' ').title()}</h2>
                    </div>
                    <div style="text-align:right;">
                        <h1 style="margin:0; color:#8c2d4f;">{confidence:.0%}</h1>
                        <span style="color:#666; font-size:0.9em;">Confidence</span>
                    </div>
                </div>
            </div>
        """,
            unsafe_allow_html=True,
        )

        # --- SOLID WHITE BLOCK VIA HTML ---
        top_indices = np.argsort(probs)[::-1][:5]

        items_html = ""
        for idx in top_indices:
            loc_name = idx_to_location[idx].replace("_", " ").title()
            prob_val = float(probs[idx])
            percent_str = f"{prob_val:.0%}"
            bar_width = max(int(prob_val * 100), 2)  # ensure small bars remain visible

            items_html += f"""
            <div style="margin-bottom: 14px;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                    <span style="font-weight: 600; color: #0f172a; font-size: 0.95rem;">{loc_name}</span>
                    <span style="font-weight: 600; color: #0f172a; font-size: 0.95rem;">{percent_str}</span>
                </div>
                <div style="background-color: #e2e8f0; border-radius: 9999px; height: 8px; width: 100%; overflow: hidden;">
                    <div style="background-color: #2563eb; width: {bar_width}%; height: 100%; border-radius: 9999px;"></div>
                </div>
            </div>
            """

        card_markup = f"""
        <div style="
            background-color: #ffffff !important;
            border-radius: 12px;
            padding: 24px;
            border: 1px solid #cbd5e1;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.25);
            margin-bottom: 20px;
        ">
            <h3 style="color: #0f172a; margin-top: 0; margin-bottom: 20px; font-size: 1.25rem;">📊 Top Guesses</h3>
            {items_html}
        </div>
        """

        if hasattr(st, "html"):
            st.html(card_markup)
        else:
            st.markdown(card_markup, unsafe_allow_html=True)

    else:
        st.info("👈 Upload an image on the left to see the predicted campus location.")