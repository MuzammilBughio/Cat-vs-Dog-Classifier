
"""
Dog vs Cat Classifier - Streamlit Web App
Single file: Custom HTML/CSS frontend + TensorFlow backend.
Requires Streamlit >= 1.40
"""

import base64
import io
import time
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps


BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "cat_vs_dog_model.keras"
SAMPLES_DIR = BASE_DIR / "samples"

IMG_SIZE = (256, 256)
MAX_SAMPLES = 14
GALLERY_COLS = 7
DOG_LABEL_IS_ONE = True

st.set_page_config(
    page_title="Cat vs Dog Classifier",
    layout="wide",
    initial_sidebar_state="collapsed"
)

ss = st.session_state

ss.setdefault("sample_path", None)
ss.setdefault("uploader_key", 0)
ss.setdefault("result", None)


def pick_sample(path):
    ss.sample_path = path
    ss.uploader_key += 1
    ss.result = None



STATIC_CSS = """
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

:root {
    --background: #0b1120;
    --surface: #111c30;
    --surface-light: #17243a;
    --border: rgba(148,163,184,.16);
    --text: #e8eef8;
    --muted: #91a0b8;
    --blue: #3b82f6;
    --cyan: #38bdf8;
}

/* PAGE */

.stApp {
    background:
        radial-gradient(ellipse at 10% 0%, rgba(59,130,246,.10), transparent 38%),
        #0b1120;
    background-attachment: fixed;
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stHeader"],
[data-testid="stToolbar"],
#MainMenu,
footer {
    display: none !important;
}

.block-container,
[data-testid="stMainBlockContainer"] {
    max-width: 1250px !important;
    padding: 30px 38px 25px !important;
}

html, body, .stApp {
    color: var(--text);
}

/* HEADER */

.app-header {
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 75px;
    padding: 0 0 18px;
    margin-bottom: 30px;
    border-bottom: 1px solid var(--border);
}

.app-title {
    font-size: 34px;
    font-weight: 800;
    letter-spacing: -1.2px;
    color: #f1f5f9;
    text-align: center;
    line-height: 1.2;
}

.app-title span {
    color: #60a5fa;
}

/* BUTTONS */

.stButton button {
    background: linear-gradient(100deg, #2563eb, #4f70e8) !important;
    color: white !important;
    border: 1px solid rgba(96,165,250,.2) !important;
    border-radius: 9px !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    box-shadow: none !important;
    transition: background .2s, transform .2s !important;
}

.stButton button:hover {
    background: linear-gradient(100deg, #3b82f6, #6366f1) !important;
    border-color: #60a5fa !important;
    transform: translateY(-1px);
}

/* ABOUT DIALOG */

[data-testid="stDialog"] {
    background: #111c30 !important;
    border: 1px solid rgba(96,165,250,.25) !important;
    border-radius: 18px !important;
    box-shadow: 0 25px 80px rgba(0,0,0,.55) !important;
}

[data-testid="stDialog"] [data-testid="stMarkdownContainer"] {
    color: #b1bfd2;
}

[data-testid="stDialog"] h2 {
    color: #bfdbfe !important;
    font-weight: 800 !important;
}

[data-testid="stDialog"] h3 {
    color: #bfdbfe !important;
    font-size: 15px !important;
}

[data-testid="stDialog"] p {
    color: #b1bfd2;
    font-size: 13px;
    line-height: 1.8;
}

[data-testid="stDialog"] li {
    color: #b1bfd2;
    font-size: 13px;
    line-height: 1.8;
}

[data-testid="stDialog"] [data-testid="stDialogCloseButton"] {
    color: #cbd5e1 !important;
}

.about-stats {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    margin: 18px 0;
}

.about-stat {
    background: rgba(59,130,246,.10);
    border: 1px solid rgba(96,165,250,.2);
    border-radius: 9px;
    padding: 12px;
}

.about-stat span {
    display: block;
    color: #91a0b8;
    font-size: 11px;
    margin-bottom: 5px;
}

.about-stat strong {
    color: #bfdbfe;
    font-size: 13px;
}

.about-tag {
    display: inline-block;
    padding: 7px 11px;
    border-radius: 7px;
    background: rgba(59,130,246,.12);
    border: 1px solid rgba(96,165,250,.2);
    color: #93c5fd;
    font-size: 11px;
    font-weight: 600;
}

/* SECTIONS */

.section-label {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.4px;
    text-transform: uppercase;
    color: #94a3b8;
    margin: 0 0 12px 2px;
}

/* MAIN AREA */

.st-key-main_area {
    min-height: 350px;
}

.st-key-main_area [data-testid="stHorizontalBlock"] {
    align-items: flex-start;
}

/* UPLOAD */

[data-testid="stFileUploaderDropzone"] {
    background: rgba(23,36,58,.7) !important;
    border: 1px dashed rgba(96,165,250,.5) !important;
    border-radius: 11px !important;
    padding: 15px !important;
    transition: border-color .2s, background .2s;
}

[data-testid="stFileUploaderDropzone"]:hover {
    border-color: #60a5fa !important;
    background: rgba(37,99,235,.09) !important;
}

[data-testid="stFileUploaderDropzoneInstructions"] {
    color: #cbd5e1 !important;
}

[data-testid="stFileUploaderDropzoneInstructions"] span {
    color: #cbd5e1 !important;
    font-size: 12px !important;
}

[data-testid="stFileUploaderDropzone"] button {
    background: #2563eb !important;
    color: white !important;
    border: none !important;
    border-radius: 7px !important;
}

[data-testid="stFileUploaderFile"] {
    background: #17243a !important;
    border-radius: 8px !important;
}

/* PREVIEW */

.thumb {
    width: 100%;
    height: 215px;
    border-radius: 12px;
    overflow: hidden;
    margin: 16px 0;
    background: #111c30;
    border: 1px solid var(--border);
    display: flex;
    justify-content: center;
    align-items: center;
}

.thumb img {
    width: 100%;
    height: 100%;
    object-fit: contain;
    display: block;
}

/* PREDICT BUTTON */

.st-key-predict_btn {
    width: 100% !important;
}

.st-key-predict_btn button {
    width: 100% !important;
    height: 43px !important;
    min-height: 43px !important;
}

/* RESULT CARD */

.card {
    margin-top: 30px;
    max-width: 70%;
    border-radius: 14px;
    padding: 21px 23px;
    background: linear-gradient(145deg, #15243b, #101b2e);
    border: 1px solid rgba(96,165,250,.2);
    box-shadow: 0 8px 30px rgba(0,0,0,.12);
}

.card .eyebrow {
    font-size: 11px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #60a5fa;
    font-weight: 700;
    margin-bottom: 17px;
}

.card .muted {
    color: #9baac0;
    font-size: 13px;
    line-height: 1.8;
}

.card .label {
    font-size: 32px;
    font-weight: 700;
    line-height: 1.2;
    margin-bottom: 8px;
    color: #f1f5f9;
}

.card .conf {
    font-size: 13px;
    color: #aab8cc;
    margin-bottom: 12px;
}

.card .conf b {
    color: #67e8f9;
    font-size: 15px;
}

.meter {
    height: 7px;
    border-radius: 99px;
    background: rgba(148,163,184,.15);
    overflow: hidden;
}

.meter > span {
    display: block;
    height: 100%;
    border-radius: 99px;
    background: linear-gradient(90deg, #3b82f6, #38bdf8);
}

/* LOADING */

.loader-text {
    font-size: 14px;
    color: #e2e8f0;
    margin-bottom: 16px;
    font-weight: 600;
}

.loader-bar {
    height: 7px;
    border-radius: 99px;
    background: rgba(148,163,184,.15);
    overflow: hidden;
    position: relative;
}

.loader-bar > span {
    position: absolute;
    top: 0;
    left: -40%;
    height: 100%;
    width: 40%;
    border-radius: 99px;
    background: linear-gradient(90deg, #3b82f6, #38bdf8);
    animation: slide 1.1s ease-in-out infinite;
}

@keyframes slide {
    0% { left: -40%; }
    100% { left: 100%; }
}

/* GALLERY */

.gallery-heading {
    margin-top: 20px;
    margin-bottom: 15px;
}

.gallery-heading .gallery-title {
    font-size: 19px;
    font-weight: 700;
    color: #e8eef8;
    margin-bottom: 4px;
}

.gallery-heading .gallery-subtitle {
    font-size: 12px;
    color: #91a0b8;
}

.st-key-gallery {
    border: 1px solid var(--border);
    background: rgba(17,28,48,.75);
    border-radius: 13px;
    padding: 16px 16px 8px;
}

[class*="st-key-pick_"] button:hover {
    transform: translateY(-3px);
    border-color: #60a5fa !important;
}

/* FOOTER */

.footer-note {
    text-align: center;
    color: #64748b;
    font-size: 11px;
    margin-top: 25px;
    padding-top: 15px;
    border-top: 1px solid var(--border);
}

/* RESPONSIVE */

@media (max-width: 900px) {
    .block-container,
    [data-testid="stMainBlockContainer"] {
        padding: 22px 20px !important;
    }
}

@media (max-width: 600px) {
    .block-container,
    [data-testid="stMainBlockContainer"] {
        padding: 15px 12px !important;
    }

    .app-title {
        font-size: 25px;
    }

    .card {
        padding: 19px;
        max-width: 100%;
    }

    .card .label {
        font-size: 29px;
    }

    .thumb {
        height: 190px;
    }

    .about-stats {
        grid-template-columns: 1fr 1fr;
    }
}

</style>
"""


def find_model_path():
    if MODEL_PATH.exists():
        return MODEL_PATH

    found = sorted(BASE_DIR.glob("*.keras"))
    return found[0] if found else None


@st.cache_resource(show_spinner=False)
def load_model():
    path = find_model_path()

    if path is None:
        raise FileNotFoundError("No .keras model found.")

    return tf.keras.models.load_model(path)


def preprocess(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize(IMG_SIZE)

    arr = np.expand_dims(
        np.array(img, dtype="float32"),
        axis=0
    )

    return arr / 255.0


def predict(img: Image.Image):
    prob = float(
        load_model().predict(
            preprocess(img),
            verbose=0
        ).ravel()[0]
    )

    is_class_one = prob > 0.5
    conf = prob if is_class_one else 1 - prob

    is_dog = (
        is_class_one
        if DOG_LABEL_IS_ONE
        else not is_class_one
    )

    return ("Dog" if is_dog else "Cat"), conf


def open_image(data) -> Image.Image:
    img = Image.open(data)
    return ImageOps.exif_transpose(img).convert("RGB")


def to_b64(img: Image.Image, size: int) -> str:
    img = ImageOps.fit(img, (size, size))

    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85)

    return base64.b64encode(
        buf.getvalue()
    ).decode()


def list_samples():
    exts = {".jpg", ".jpeg", ".png", ".webp"}

    if not SAMPLES_DIR.exists():
        return []

    files = sorted(
        p for p in SAMPLES_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in exts
    )

    return [str(p) for p in files[:MAX_SAMPLES]]


@st.dialog("About the Model", width="small")
def show_about_model():

    st.markdown("""
    This Dog vs Cat Classifier uses **MobileNetV2**, a lightweight
    deep learning architecture designed for image classification.

    The model predicts whether an uploaded image contains a cat or dog.

    ### How it works

    1. Upload a cat or dog image.
    2. The image is resized to 256 × 256 pixels.
    3. Pixel values are normalized between 0 and 1.
    4. The neural network processes the image.
    5. The predicted class and confidence are displayed.
    """)

    st.markdown("""
    <div class="about-stats">
        <div class="about-stat">
            <span>Architecture</span>
            <strong>MobileNetV2</strong>
        </div>
        <div class="about-stat">
            <span>Task</span>
            <strong>Binary Classification</strong>
        </div>
        <div class="about-stat">
            <span>Classes</span>
            <strong>Cat & Dog</strong>
        </div>
        <div class="about-stat">
            <span>Input Size</span>
            <strong>256 × 256</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="about-tag">
        TensorFlow · MobileNetV2 · Deep Learning
    </div>
    """, unsafe_allow_html=True)



st.markdown(STATIC_CSS, unsafe_allow_html=True)

if find_model_path() is None:
    st.error(
        f"Model file not found. Put your .keras model in this folder: {BASE_DIR}"
    )
    st.stop()


header_left, header_right = st.columns([5, 1])

with header_left:
    st.markdown(
        """
        <div class="app-header">
            <div class="app-title">
                Cat vs <span>Dog</span> Classifier
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with header_right:
    st.markdown(
        '<div style="height:19px"></div>',
        unsafe_allow_html=True
    )

    if st.button(
        "About Model",
        key="about_model_btn",
        use_container_width=True
    ):
        show_about_model()



with st.container(key="main_area"):

    col_a, col_b, col_c = st.columns(
        [0.35, 1.15, 2.1],
        gap="large"
    )

    result_slot = col_c.empty()

    with col_b:

        st.markdown(
            '<div class="section-label">Upload image</div>',
            unsafe_allow_html=True
        )

        with st.container(key="upload_box"):

            uploaded = st.file_uploader(
                "Upload image",
                type=["jpg", "jpeg", "png", "webp"],
                key=f"uploader_{ss.uploader_key}",
                label_visibility="collapsed"
            )

        image, image_id = None, None

        if uploaded is not None:

            image = open_image(
                io.BytesIO(uploaded.getvalue())
            )

            image_id = f"upload:{uploaded.name}:{uploaded.size}"

        elif ss.sample_path:

            image = open_image(ss.sample_path)
            image_id = f"sample:{ss.sample_path}"

        clicked = False

        if image is not None:

            st.markdown(
                f'<div class="thumb"><img src="data:image/jpeg;base64,{to_b64(image, 400)}"></div>',
                unsafe_allow_html=True
            )

            clicked = st.button(
                "Predict",
                key="predict_btn"
            )

    if clicked and image is not None:

        result_slot.markdown(
            """
            <div class="card">
                <div class="loader-text">Predicting...</div>
                <div class="loader-bar"><span></span></div>
            </div>
            """,
            unsafe_allow_html=True
        )

        t0 = time.time()

        try:
            label, conf = predict(image)

            time.sleep(
                max(0.0, 0.8 - (time.time() - t0))
            )

            ss.result = (image_id, label, conf)

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            ss.result = None

    if ss.result and image_id is not None and ss.result[0] == image_id:

        _, label, conf = ss.result

        result_slot.markdown(
            f"""
            <div class="card">
                <div class="eyebrow">Prediction Result</div>
                <div class="label">{label}</div>
                <div class="conf">
                    Confidence: <b>{conf * 100:.2f}%</b>
                </div>
                <div class="meter">
                    <span style="width:{conf * 100:.2f}%"></span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        result_slot.markdown(
            """
            <div class="card">
                <div class="eyebrow">Prediction Result</div>
                <div class="muted">
                    Upload an image, or choose one from the samples below,
                    then press Predict.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )



def gallery_css(samples):

    css = ""

    for i, path in enumerate(samples):

        img = Image.open(path).convert("RGB")
        img = ImageOps.fit(img, (300, 300))

        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)

        encoded = base64.b64encode(
            buffer.getvalue()
        ).decode("ascii")

        css += f"""
        .st-key-pick_{i} button {{
            width: 100% !important;
            aspect-ratio: 1 / 1 !important;
            height: auto !important;
            min-height: 0 !important;
            padding: 0 !important;
            background-image: url("data:image/jpeg;base64,{encoded}") !important;
            background-size: cover !important;
            background-position: center !important;
            background-repeat: no-repeat !important;
            border-radius: 12px !important;
            border: 1px solid rgba(148,163,184,0.25) !important;
            color: transparent !important;
            font-size: 0 !important;
            transition: all 0.2s ease !important;
        }}

        .st-key-pick_{i} button p {{
            visibility: hidden !important;
            font-size: 0 !important;
        }}

        .st-key-pick_{i} button:hover {{
            border: 2px solid #38bdf8 !important;
            transform: translateY(-3px);
        }}
        """

    return f"<style>{css}</style>"


samples = list_samples()

with st.container(key="gallery"):

    if not samples:

        st.info("Add dog/cat images to your samples folder.")

    else:

        st.markdown(
            gallery_css(samples),
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="gallery-heading">
                <div class="gallery-title">Sample Images</div>
                <div class="gallery-subtitle">
                    Select an image to try the classifier
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        for start in range(0, len(samples), GALLERY_COLS):

            cols = st.columns(GALLERY_COLS, gap="small")

            batch = list(enumerate(samples))[
                start:start + GALLERY_COLS
            ]

            for col, (i, path) in zip(cols, batch):

                with col:

                    st.button(
                        " ",
                        key=f"pick_{i}",
                        on_click=pick_sample,
                        args=(path,),
                        use_container_width=True
                    )


st.markdown(
    """
    <div class="footer-note">
        TensorFlow · MobileNetV2 · Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
