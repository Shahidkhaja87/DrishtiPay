import json
from datetime import datetime
from pathlib import Path

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import tensorflow as tf
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "best_currency_model.keras"
LABELS_PATH = ROOT / "models" / "class_names.json"

st.set_page_config(
    page_title="DrishtiPay Lite",
    page_icon="🔊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------- Design system --------------------
NAVY = "#123B72"
BLUE = "#1769E0"
PALE_BLUE = "#EAF2FF"
TEXT = "#172B4D"
MUTED = "#65758B"
BORDER = "#D9E3F0"
GREEN = "#147D54"
AMBER = "#9A6700"

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] {{ font-family: Inter, Arial, sans-serif; }}
    .stApp {{ background:#F5F8FD; color:{TEXT}; }}
    [data-testid="stHeader"] {{ background:rgba(245,248,253,.92); }}
    .block-container {{ max-width:1180px; padding-top:1.25rem; padding-bottom:3rem; }}
    .hero {{
      background:linear-gradient(120deg,{NAVY},{BLUE});
      padding:1.7rem 1.9rem; border-radius:22px; color:#fff;
      margin-bottom:1.15rem; box-shadow:0 12px 32px rgba(18,59,114,.14);
    }}
    .hero h1 {{ color:#fff; font-size:clamp(1.8rem,4vw,2.65rem); margin:0 0 .35rem; }}
    .hero p {{ color:#E5EEFF; margin:0; font-size:1rem; }}
    .eyebrow {{ color:#D8E7FF; font-size:.74rem; font-weight:800; letter-spacing:.13em; text-transform:uppercase; }}
    .panel {{ background:#fff; border:1px solid {BORDER}; border-radius:18px; padding:1.2rem; margin-bottom:1rem; }}
    .step {{ color:{BLUE}; font-size:.75rem; font-weight:800; letter-spacing:.1em; text-transform:uppercase; }}
    .panel-title {{ color:{NAVY}; font-size:1.35rem; font-weight:750; margin:.25rem 0 .8rem; }}
    .hint {{ background:{PALE_BLUE}; border-left:4px solid {BLUE}; border-radius:10px; padding:.85rem 1rem; }}
    .safety {{ background:#FFF8E5; border-left:4px solid #D99A23; border-radius:10px; padding:.9rem 1rem; color:#714D00; }}
    .result-value {{ color:{NAVY}; font-size:2.7rem; font-weight:800; line-height:1.15; }}
    div.stButton > button, div.stDownloadButton > button {{
      min-height:2.8rem; border-radius:12px; font-weight:700;
      border:1px solid {BORDER}; transition:all .15s ease;
    }}
    div.stButton > button[kind="primary"] {{ background:{BLUE}; color:white; border-color:{BLUE}; }}
    div.stButton > button[kind="primary"]:hover {{ background:{NAVY}; border-color:{NAVY}; color:white; }}
    [data-testid="stMetric"] {{ background:#fff; border:1px solid {BORDER}; border-radius:14px; padding:.8rem; }}
    [data-testid="stFileUploader"] section {{ border-radius:12px; }}
    footer {{ visibility:hidden; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------- Speech and model helpers --------------------
def speech_button(message, button_label="🔊 Speak"):
    """Browser-native text-to-speech. Browser voice availability varies by device."""
    message_js = json.dumps(message)
    label_html = (
        button_label.replace("&", "&amp;").replace("<", "&lt;")
        .replace(">", "&gt;").replace('"', "&quot;")
    )
    lang = st.session_state.get("speech_lang", "en-IN")
    components.html(
        f"""
        <div style="font-family:Arial,sans-serif;padding:4px 0;">
          <button id="speakBtn" aria-label="{label_html}"
            style="font-size:16px;font-weight:600;padding:11px 16px;
            border:1px solid #D9E3F0;border-radius:11px;background:#fff;
            cursor:pointer;">{label_html}</button>
          <p id="speechStatus" aria-live="polite" style="font-size:13px;color:#52647C;"></p>
        </div>
        <script>
          const textToSpeak = {message_js};
          const btn = document.getElementById("speakBtn");
          const status = document.getElementById("speechStatus");
          btn.addEventListener("click", function() {{
            if (!("speechSynthesis" in window)) {{
              status.textContent = "Speech is not supported in this browser.";
              return;
            }}
            window.speechSynthesis.cancel();
            const utterance = new SpeechSynthesisUtterance(textToSpeak);
            utterance.lang = {json.dumps(lang)};
            utterance.rate = 0.9;
            utterance.onstart = () => status.textContent = "Speaking…";
            utterance.onend = () => status.textContent = "Speech finished.";
            utterance.onerror = () => status.textContent = "Speech failed. Check device volume and browser settings.";
            window.speechSynthesis.speak(utterance);
            status.textContent = "Speech requested.";
          }});
        </script>
        """,
        height=82,
    )


@st.cache_resource
def load_currency_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not LABELS_PATH.is_file():
        raise FileNotFoundError(f"Labels file not found: {LABELS_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        class_names = json.load(f)
    if not isinstance(class_names, list):
        raise ValueError("class_names.json must contain a JSON list of class labels.")
    if len(class_names) != int(model.output_shape[-1]):
        raise ValueError(
            f"Found {len(class_names)} labels but model outputs "
            f"{model.output_shape[-1]} scores."
        )
    return model, class_names


def prepare_image(uploaded_image):
    image = ImageOps.exif_transpose(Image.open(uploaded_image)).convert("RGB")
    image = image.resize((224, 224), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32)
    # Keep this only if training used MobileNetV2 preprocess_input.
    array = tf.keras.applications.mobilenet_v2.preprocess_input(array)
    return np.expand_dims(array, axis=0)


def format_label(label):
    label = str(label)
    if label.lower() == "background":
        return "Background / unrecognised object"
    if label.startswith("₹"):
        return f"{label} banknote"
    return f"₹{label} banknote"


def get_speech_messages(label, status, language):
    if language == "Hindi":
        if status == "uncertain":
            return "नतीजा स्पष्ट नहीं है। कृपया रोशनी सुधारें और दोबारा तस्वीर लें। इस अनुमान पर भुगतान के लिए भरोसा न करें।"
        if status == "background":
            return "बैंकनोट की पहचान नहीं हो सकी। कृपया नोट को बीच में रखें और दोबारा तस्वीर लें।"
        return f"मॉडल का अनुमान {label} रुपये का नोट है। यह असली होने की पुष्टि नहीं है। कृपया स्वतंत्र रूप से जांच करें।"
    if status == "uncertain":
        return "The result is uncertain. Improve the lighting and take another photo. Do not rely on this prediction for payment."
    if status == "background":
        return "I could not identify a banknote. Centre the note in the frame and try again."
    return f"The model predicts a {label} rupee banknote. This is not authentication. Please check independently."


def clear_current_scan():
    st.session_state.pop("last_prediction", None)
    st.session_state.pop("last_photo_key", None)


# -------------------- Sidebar settings --------------------
with st.sidebar:
    st.markdown(f"<h2 style='color:{NAVY};margin-bottom:0'>🔊 DrishtiPay Lite</h2>", unsafe_allow_html=True)
    st.caption("Voice-guided currency accessibility")
    st.divider()
    st.markdown("### Accessibility settings")
    language = st.selectbox("Voice language", ["English", "Hindi"])
    st.session_state["speech_lang"] = "hi-IN" if language == "Hindi" else "en-IN"
    threshold = st.slider(
        "Minimum score to announce",
        min_value=0.40, max_value=0.95, value=0.70, step=0.05,
        help="This is a cautious display threshold, not a guarantee or calibrated probability.",
    )
    large_text = st.toggle("Larger text", value=False)
    high_contrast = st.toggle("High-contrast mode", value=False)
    st.divider()
    st.markdown("### How to use")
    st.markdown("1. Test voice guidance.\n2. Capture or upload a note.\n3. Review the photo.\n4. Analyze the image.\n5. Listen to the result.")
    st.caption("Prototype only. Does not authenticate currency.")

if large_text:
    st.markdown("<style>html,body,[class*='css']{font-size:1.08rem!important}</style>", unsafe_allow_html=True)
if high_contrast:
    st.markdown("<style>.stApp{background:#fff!important;color:#000!important}.panel{border:2px solid #111!important}p,label{color:#111!important}</style>", unsafe_allow_html=True)

# -------------------- Header --------------------
st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">AI for social good · accessibility prototype</div>
      <h1>🔊 DrishtiPay Lite</h1>
      <p>Identify currency denominations with simple guidance and voice-first interactions.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="safety"><strong>Safety first:</strong> AI predictions can be wrong. '
    'This app does not verify whether a banknote is genuine. Do not rely on it alone for payments.</div>',
    unsafe_allow_html=True,
)
st.write("")

# -------------------- Voice test and guidance --------------------
voice_col, guide_col = st.columns([1, 1.5], gap="large")
with voice_col:
    st.markdown('<div class="panel"><div class="step">Step 1</div><div class="panel-title">Test voice</div><p>Check that your device can read instructions aloud.</p></div>', unsafe_allow_html=True)
    welcome = (
        "दृष्टिपे लाइट में आपका स्वागत है। आवाज़ की जांच सफल है।"
        if language == "Hindi"
        else "Welcome to DrishtiPay Lite. Voice feedback test successful."
    )
    speech_button(welcome, "🔊 Test voice")
with guide_col:
    st.markdown(
        '<div class="panel"><div class="step">Before scanning</div><div class="panel-title">Get a clear photo</div>'
        '<p>• Place one note on a flat, contrasting surface.<br>'
        '• Use bright, even lighting and avoid glare.<br>'
        '• Keep the entire note inside the frame.<br>'
        '• Hold the camera steady and keep fingers away.</p></div>',
        unsafe_allow_html=True,
    )

# -------------------- Capture workflow --------------------
st.markdown('<div class="panel"><div class="step">Step 2</div><div class="panel-title">Capture a banknote</div>', unsafe_allow_html=True)
source = st.radio("Image source", ["Camera", "Upload photo"], horizontal=True, label_visibility="collapsed")
photo = None
if source == "Camera":
    photo = st.camera_input("Position the complete banknote inside the frame", key="banknote_camera")
else:
    photo = st.file_uploader("Upload a clear photo of one banknote", type=["png", "jpg", "jpeg", "webp"])

if photo is not None:
    try:
        preview = ImageOps.exif_transpose(Image.open(photo)).convert("RGB")
        st.image(preview, caption="Review your captured banknote", use_container_width=True)
        left, right = st.columns(2)
        with left:
            if st.button("↻ Clear / retake photo", use_container_width=True):
                clear_current_scan()
                st.rerun()
        with right:
            analyze_clicked = st.button("✦ Analyze banknote", type="primary", use_container_width=True)
        if analyze_clicked:
            with st.spinner("Analyzing the image with your AI model…"):
                model, class_names = load_currency_model()
                scores = np.asarray(model.predict(prepare_image(photo), verbose=0))[0]
                if scores.ndim != 1 or len(scores) != len(class_names):
                    raise ValueError("The model output does not match the label list.")
                if not np.all(np.isfinite(scores)):
                    raise ValueError("The model returned non-finite values.")
                is_probabilities = (
                    np.all(scores >= 0) and np.all(scores <= 1)
                    and np.isclose(float(np.sum(scores)), 1.0, atol=0.03)
                )
                best_index = int(np.argmax(scores))
                best_label = str(class_names[best_index])
                best_score = float(scores[best_index])
                status = "experimental"
                if is_probabilities and best_label.lower() == "background":
                    status = "background"
                elif is_probabilities and best_score < threshold:
                    status = "uncertain"
                elif is_probabilities:
                    status = "predicted"
                result = {
                    "label": best_label,
                    "score": best_score,
                    "scores": scores.tolist(),
                    "class_names": [str(x) for x in class_names],
                    "is_probabilities": bool(is_probabilities),
                    "status": status,
                    "time": datetime.now().strftime("%H:%M:%S"),
                }
                st.session_state["last_prediction"] = result
                st.session_state["last_photo_key"] = getattr(photo, "name", "camera-photo")
                st.session_state.setdefault("scan_history", []).append({
                    "Time": result["time"],
                    "Prediction": format_label(best_label),
                    "Raw model score": f"{best_score:.0%}",
                    "Status": status.title(),
                })
    except Exception as exc:
        st.error("The image could not be analyzed. Check your model files and preprocessing, then try again.")
        st.code(f"{type(exc).__name__}: {exc}")
st.markdown("</div>", unsafe_allow_html=True)

# -------------------- Results --------------------
st.markdown('<div class="panel"><div class="step">Step 3</div><div class="panel-title">AI classification result</div>', unsafe_allow_html=True)
result = st.session_state.get("last_prediction")

if result:
    label = result["label"]
    score = float(result["score"])
    status = result["status"]
    st.markdown(f'<div class="result-value">{format_label(label)}</div>', unsafe_allow_html=True)
    st.progress(max(0.0, min(1.0, score)), text=f"Raw model score: {score:.0%}")

    if not result["is_probabilities"]:
        st.warning("Model outputs do not look like probabilities. Treat this as experimental; the score should not be interpreted as confidence.")
        message = "The model returned an experimental classification. Please check the banknote independently."
    elif status == "background":
        st.warning("No banknote confidently identified. Try a clearer photo with the whole note in view.")
        message = get_speech_messages(label, "background", language)
    elif status == "uncertain":
        st.warning(f"Uncertain result: score is below your selected {threshold:.0%} threshold.")
        message = get_speech_messages(label, "uncertain", language)
    else:
        st.success("Prediction meets your selected display threshold.")
        st.warning("This prediction may be incorrect. Check the note independently before any payment.")
        message = get_speech_messages(label, "predicted", language)

    speech_button(message, "🔊 Listen to result")
    with st.expander("View all model scores"):
        score_rows = sorted(
            zip(result["class_names"], result["scores"]),
            key=lambda item: item[1], reverse=True,
        )
        for score_label, raw_score in score_rows:
            st.progress(
                max(0.0, min(1.0, float(raw_score))),
                text=f"{format_label(score_label)} — {float(raw_score):.1%}",
            )
    st.caption("Raw model scores are not proof of authenticity and may not be calibrated probabilities.")
else:
    st.info("Capture or upload a photo, then select **Analyze banknote**. Your result will appear here.")
st.markdown("</div>", unsafe_allow_html=True)

# -------------------- Session history --------------------
with st.expander("🕘 Scan history for this session"):
    history = st.session_state.get("scan_history", [])
    if history:
        st.dataframe(history, use_container_width=True, hide_index=True)
        if st.button("Clear scan history"):
            st.session_state["scan_history"] = []
            st.rerun()
    else:
        st.caption("No analyzed banknotes in this session yet.")

st.divider()
st.caption(
    "DrishtiPay Lite is an educational image-classification prototype. "
    "It cannot authenticate currency. Session history is temporary and is not saved to a database."
)