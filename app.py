import streamlit as st
from PIL import Image
from pathlib import Path
import sys

# Allow importing from src/
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from predict import load_predictor


# =============================================================
# Page configuration
# =============================================================

st.set_page_config(
    page_title="Cucumber Disease Detector",
    page_icon="🌱",
    layout="centered",
)


# =============================================================
# Design tokens
# =============================================================

BG          = "#0e1117"
BG_CARD     = "#161b22"
BG_INPUT    = "#1c2129"
TEXT        = "#e6edf3"
TEXT_SEC    = "#8b949e"
BORDER      = "#30363d"
ACCENT      = "#58a6ff"
ACCENT2     = "#34d399"
SUCCESS_CLR = "#3fb950"
WARNING_CLR = "#d29922"
DANGER_CLR  = "#f85149"
BAR_TRACK   = "#21262d"
CHIP_BG     = "rgba(88,166,255,0.10)"
CHIP_BD     = "rgba(88,166,255,0.25)"


# =============================================================
# CSS — theme overrides for Streamlit internals
#
# Data display uses NATIVE Streamlit components so it never
# shows raw HTML.  This CSS only re-skins Streamlit's own
# surfaces for the selected theme.
# =============================================================


st.markdown(
    f"""
    <style>
    /* ---------- backgrounds ---------- */
    .stApp,
    .stApp > header,
    [data-testid="stHeader"] {{
        background-color: {BG} !important;
    }}
    .stMainBlockContainer,
    .block-container,
    [data-testid="stAppViewBlockContainer"] {{
        background-color: {BG} !important;
    }}

    /* ---------- text ---------- */
    .stApp,
    .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span,
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
    .stMarkdown h4, .stMarkdown h5, .stMarkdown h6,
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stMarkdownContainer"] span,
    [data-testid="stWidgetLabel"] p,
    label {{
        color: {TEXT} !important;
    }}
    .stCaption, .stCaption p,
    [data-testid="stCaptionContainer"],
    [data-testid="stCaptionContainer"] p,
    figcaption, small {{
        color: {TEXT_SEC} !important;
    }}

    /* ---------- file uploader ---------- */
    [data-testid="stFileUploader"] section {{
        background-color: {BG_INPUT} !important;
        border: 2px dashed {BORDER} !important;
        border-radius: 12px !important;
    }}
    [data-testid="stFileUploader"] div,
    [data-testid="stFileUploader"] span,
    [data-testid="stFileUploader"] small {{
        color: {TEXT} !important;
    }}
    [data-testid="stFileUploader"] button {{
        color: {ACCENT} !important;
    }}

    /* ---------- buttons ---------- */
    .stButton > button {{
        border-radius: 8px !important;
        font-weight: 600 !important;
        border: 1px solid {BORDER} !important;
        color: {TEXT} !important;
        background-color: {BG_CARD} !important;
        transition: border-color 0.15s, color 0.15s !important;
    }}
    .stButton > button:hover {{
        border-color: {ACCENT} !important;
        color: {ACCENT} !important;
    }}
    /* primary button */
    .stButton > button[kind="primary"],
    .stButton > button[data-testid="stBaseButton-primary"] {{
        background: linear-gradient(135deg, {ACCENT}, {ACCENT2}) !important;
        color: #ffffff !important;
        border: none !important;
    }}
    .stButton > button[kind="primary"]:hover,
    .stButton > button[data-testid="stBaseButton-primary"]:hover {{
        opacity: 0.9;
    }}

    /* ---------- bordered containers ---------- */
    [data-testid="stVerticalBlockBorderWrapper"] {{
        background-color: {BG_CARD} !important;
        border-color: {BORDER} !important;
        border-radius: 14px !important;
    }}

    /* ---------- expander ---------- */
    [data-testid="stExpander"] {{
        background-color: {BG_CARD} !important;
        border-color: {BORDER} !important;
        border-radius: 10px !important;
    }}
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary p {{
        color: {TEXT} !important;
    }}

    /* ---------- progress bars ---------- */
    .stProgress > div > div {{
        background-color: {BAR_TRACK} !important;
        border-radius: 4px !important;
    }}
    .stProgress > div > div > div {{
        background: linear-gradient(90deg, {ACCENT}, {ACCENT2}) !important;
        border-radius: 4px !important;
    }}

    /* ---------- metric ---------- */
    [data-testid="stMetricValue"] {{
        color: {TEXT} !important;
    }}
    [data-testid="stMetricLabel"] p {{
        color: {TEXT_SEC} !important;
    }}

    /* ---------- alerts ---------- */
    [data-testid="stAlert"] {{
        border-radius: 8px !important;
    }}

    /* ---------- divider ---------- */
    hr {{
        border-color: {BORDER} !important;
    }}

    /* ---------- image ---------- */
    [data-testid="stImage"] img {{
        border-radius: 10px !important;
    }}

    /* ---------- scrollbar ---------- */
    ::-webkit-scrollbar-track {{ background: {BG}; }}
    ::-webkit-scrollbar-thumb {{ background: {BORDER}; border-radius: 4px; }}

    /* ========== CUSTOM: hero title gradient ========== */
    .hero-wrap {{
        text-align: center;
        padding: 1.5rem 0 0.5rem;
    }}
    .hero-wrap .icon {{ font-size: 48px; }}
    .hero-wrap .hero-title {{
        font-size: 2.1rem;
        font-weight: 800;
        margin: 0.2rem 0 0;
        background: linear-gradient(135deg, {ACCENT}, {ACCENT2});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }}
    .hero-wrap .hero-sub {{
        color: {TEXT_SEC};
        font-size: 0.95rem;
        margin-top: 0.3rem;
    }}

    /* ========== CUSTOM: class chips ========== */
    .class-chip {{
        display: inline-block;
        background: {CHIP_BG};
        color: {ACCENT};
        border: 1px solid {CHIP_BD};
        padding: 5px 14px;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 600;
        margin: 3px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)




# =============================================================
# Hero header (only custom HTML in the entire app)
# =============================================================

st.markdown(
    f"""
    <div class="hero-wrap">
        <div class="icon">🥒</div>
        <h1 class="hero-title">Cucumber Disease Detector</h1>
        <p class="hero-sub">
            Upload a leaf or fruit image • Powered by Vision Transformer (ViT-B/16)
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# =============================================================
# Load ViT model (cached)
# =============================================================

CHECKPOINT = ROOT / "outputs" / "vit" / "best.pt"


@st.cache_resource
def load_model():
    return load_predictor("vit", str(CHECKPOINT))


if not CHECKPOINT.exists():
    st.error(f"Model checkpoint not found:\n\n`{CHECKPOINT}`")
    st.stop()

try:
    predictor = load_model()
except Exception as e:
    st.error("Could not load the ViT model.")
    st.exception(e)
    st.stop()


# =============================================================
# Image upload
# =============================================================

uploaded_file = st.file_uploader(
    "Upload a cucumber leaf or fruit image",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG, PNG",
)


# =============================================================
# Sample images from test set
# =============================================================

SAMPLE_DIR = ROOT / "data_clean" / "test"
sample_images: dict[str, Path] = {}
if SAMPLE_DIR.exists():
    for cls_dir in sorted(SAMPLE_DIR.iterdir()):
        if cls_dir.is_dir():
            imgs = sorted(cls_dir.glob("*.jpg"))
            if imgs:
                sample_images[cls_dir.name.replace("_", " ")] = imgs[0]

if sample_images and uploaded_file is None:
    st.markdown("**Or try a sample image**")
    cols = st.columns(4)
    for idx, (cls_name, img_path) in enumerate(sample_images.items()):
        with cols[idx % 4]:
            # Show image thumbnail with caption
            st.image(img_path, width=120, caption=cls_name)
            if st.button("Select", key=f"sample_{idx}"):
                st.session_state["sample_image_path"] = str(img_path)
                st.session_state.pop("result", None)
                st.rerun()


# =============================================================
# Resolve active image
# =============================================================

image = None
image_caption = "Uploaded image"

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.session_state.pop("sample_image_path", None)
elif "sample_image_path" in st.session_state:
    p = Path(st.session_state["sample_image_path"])
    if p.exists():
        image = Image.open(p).convert("RGB")
        image_caption = f"Sample — {p.parent.name.replace('_', ' ')}"


# =============================================================
# Prediction & results  — ALL native Streamlit components
# =============================================================

if image is not None:
    st.image(image, caption=image_caption)

    if st.button("🔍  Analyze Image", type="primary"):
        with st.spinner("Analyzing image…"):
            result = predictor(image)
        st.session_state["result"] = result

    # ----------------------------------------------------------
    # Display result (persisted in session_state)
    # ----------------------------------------------------------
    result = st.session_state.get("result")

    if result is not None:
        prediction    = result.get("prediction")
        confidence    = result.get("confidence")
        probabilities = result.get("probabilities")

        if prediction is None:
            st.error("Prediction was not returned by the model.")
            st.stop()

        # Normalise confidence to percentage
        if confidence is not None:
            confidence_pct = confidence * 100 if confidence <= 1 else confidence
        else:
            confidence_pct = 0.0

        # ---- Result card (native bordered container) ----
        with st.container(border=True):
            st.caption("🧠 AI PREDICTION")
            st.markdown(f"### {prediction.replace('_', ' ')}")
            st.metric(label="Confidence", value=f"{confidence_pct:.1f}%")
            st.progress(confidence_pct / 100)
            st.markdown("---")

        # ---- Confidence interpretation ----
        if confidence_pct >= 80:
            st.success(
                "✅  **High confidence prediction.** "
                "The model is confident in this classification."
            )
        elif confidence_pct >= 50:
            st.warning(
                "⚠️  **Moderate confidence.** "
                "Consider taking another clear image for verification."
            )
        else:
            st.error(
                "⛔  **Low confidence.** "
                "Try a clearer image with the affected area clearly visible."
            )

        # ---- Probability comparison (native progress bars) ----
        st.markdown("---")
        st.markdown("### Probability Comparison")

        sorted_probs = sorted(
            probabilities.items(), key=lambda x: x[1], reverse=True
        )

        for class_name, prob in sorted_probs:
            prob_norm = prob if prob <= 1 else prob / 100
            pct = prob_norm * 100
            is_predicted = (class_name == prediction)
            display_name = class_name.replace("_", " ")
            col_name, col_bar, col_pct = st.columns([3, 6, 1])
            with col_name:
                label = f"**🏷️ {display_name}**" if is_predicted else display_name
                st.markdown(label)
            with col_bar:
                st.progress(min(prob_norm, 1.0))
            with col_pct:
                st.markdown(f"{pct:.1f}%")

        st.caption(
            "Probabilities are from the ViT-B/16 model's softmax output"
        )


# =============================================================
# Supported classes
# =============================================================

with st.expander("ℹ️  Supported cucumber classes"):
    default_classes = [
        "Anthracnose", "Bacterial Wilt", "Belly Rot", "Downy Mildew",
        "Fresh Cucumber", "Fresh Leaf", "Gummy Stem Blight",
        "Pythium Fruit Rot",
    ]

    classes = default_classes
    if SAMPLE_DIR.exists():
        detected = sorted(
            d.name.replace("_", " ")
            for d in SAMPLE_DIR.iterdir() if d.is_dir()
        )
        if detected:
            classes = detected

    chips_html = "".join(
        f'<span class="class-chip">{c}</span>' for c in classes
    )
    st.markdown(
        f'<div style="text-align:center;margin:8px 0;">{chips_html}</div>',
        unsafe_allow_html=True,
    )


# =============================================================
# Footer
# =============================================================

st.divider()
st.caption(
    "Built with **ViT-B/16** · Streamlit"
)