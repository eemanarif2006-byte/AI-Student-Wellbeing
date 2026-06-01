import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Student Impact Predictor",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0a0a0f;
    color: #e8e8f0;
}

/* Hero */
.hero {
    text-align: center;
    padding: 2.5rem 1rem 1rem;
}
.hero h1 {
    font-family: 'Space Mono', monospace;
    font-size: clamp(1.6rem, 5vw, 2.4rem);
    font-weight: 700;
    color: #ffffff;
    letter-spacing: -0.02em;
    line-height: 1.2;
    margin-bottom: 0.4rem;
}
.hero h1 span {
    color: #7c6af7;
}
.hero p {
    font-size: 0.95rem;
    color: #8888a8;
    margin-top: 0;
}

/* Card */
.card {
    background: #12121e;
    border: 1px solid #1e1e30;
    border-radius: 16px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.2rem;
}
.card-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #7c6af7;
    margin-bottom: 1rem;
}

/* Result boxes */
.result-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
    margin-top: 1rem;
}
.result-box {
    background: #0d0d1a;
    border-radius: 12px;
    padding: 1.4rem 1.2rem;
    text-align: center;
    border: 1px solid #1e1e30;
}
.result-label {
    font-size: 0.72rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #6666aa;
    margin-bottom: 0.5rem;
}
.result-value {
    font-family: 'Space Mono', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    color: #ffffff;
}
.burnout-high   { color: #f76c6c; }
.burnout-medium { color: #f7c46c; }
.burnout-low    { color: #6cf7a8; }

/* Pill badge */
.badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-top: 0.4rem;
}
.badge-green  { background: #0f2e1e; color: #6cf7a8; border: 1px solid #1a5e35; }
.badge-yellow { background: #2e2500; color: #f7c46c; border: 1px solid #5e4800; }
.badge-red    { background: #2e0f0f; color: #f76c6c; border: 1px solid #5e1a1a; }

/* Slider label override */
.stSlider > label {
    font-size: 0.85rem !important;
    color: #ccccee !important;
}

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #7c6af7, #5b4fe0) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.04em !important;
    padding: 0.65rem 1.8rem !important;
    width: 100% !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* Divider */
hr { border-color: #1e1e30 !important; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0d0d1a;
}

/* Hide streamlit branding */
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Load models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    base = os.path.dirname(__file__)
    gpa_model     = joblib.load(os.path.join(base, "gpa_model.pkl"))
    burnout_model = joblib.load(os.path.join(base, "burnout_model.pkl"))
    scaler        = joblib.load(os.path.join(base, "scaler.pkl"))
    return gpa_model, burnout_model, scaler

@st.cache_data
def load_defaults():
    """Load the cleaned CSV to get column means as default fill values."""
    base = os.path.dirname(__file__)
    path = os.path.join(base, "cleaned_students_data.csv")
    df = pd.read_csv(path)
    return df.drop(columns=["Post_Semester_GPA", "Burnout_Risk_Level"], errors="ignore").mean().to_dict()

try:
    gpa_model, burnout_model, scaler = load_models()
    defaults = load_defaults()
    models_loaded = True
except Exception as e:
    models_loaded = False
    load_error = str(e)

# ── Feature columns (same order as training) ──────────────────────────────────
FEATURE_COLS = [
    "Pre_Semester_GPA",
    "Weekly_GenAI_Hours",
    "Traditional_Study_Hours",
    "Anxiety_Level_During_Exams",
    "Skill_Retention_Score",
]

BURNOUT_LABELS = {0: "High", 1: "Low", 2: "Medium"}
BURNOUT_COLORS = {"High": "burnout-high", "Medium": "burnout-medium", "Low": "burnout-low"}
BURNOUT_BADGES = {"High": "badge-red",    "Medium": "badge-yellow",   "Low": "badge-green"}
BURNOUT_EMOJI  = {"High": "🔴",           "Medium": "🟡",             "Low": "🟢"}

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>AI <span>Student</span> Impact<br>Predictor</h1>
  <p>GPA &amp; Burnout Risk · Powered by Random Forest · BSAI Semester 2</p>
</div>
""", unsafe_allow_html=True)

if not models_loaded:
    st.error(f"⚠️ Could not load models. Make sure `gpa_model.pkl`, `burnout_model.pkl`, and `scaler.pkl` are in the same folder as `app.py`.\n\n`{load_error}`")
    st.stop()

# ── Input form ────────────────────────────────────────────────────────────────
st.markdown('<div class="card"><div class="card-title">📋 Student Profile</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    pre_gpa = st.slider(
        "Pre-Semester GPA",
        min_value=0.0, max_value=4.0,
        value=float(round(defaults.get("Pre_Semester_GPA", 2.5), 2)),
        step=0.01,
        help="GPA at the start of the semester (0.0 – 4.0)"
    )
    weekly_ai = st.slider(
        "Weekly GenAI Hours",
        min_value=0.0, max_value=40.0,
        value=float(round(defaults.get("Weekly_GenAI_Hours", 5.0), 1)),
        step=0.5,
        help="Hours per week spent using generative AI tools"
    )
    traditional_hrs = st.slider(
        "Traditional Study Hours / week",
        min_value=0.0, max_value=60.0,
        value=float(round(defaults.get("Traditional_Study_Hours", 15.0), 1)),
        step=0.5,
        help="Hours per week studying without AI assistance"
    )

with col2:
    anxiety = st.slider(
        "Exam Anxiety Level (1 – 10)",
        min_value=1, max_value=10,
        value=int(round(defaults.get("Anxiety_Level_During_Exams", 5))),
        help="Self-reported anxiety during exams"
    )
    skill_retention = st.slider(
        "Skill Retention Score (0 – 100)",
        min_value=0.0, max_value=100.0,
        value=float(round(defaults.get("Skill_Retention_Score", 60.0), 1)),
        step=0.5,
        help="Higher = better conceptual understanding retained"
    )

st.markdown('</div>', unsafe_allow_html=True)

# ── Predict ───────────────────────────────────────────────────────────────────
predict_btn = st.button("⚡ Predict GPA & Burnout Risk")

if predict_btn:
    # Build input row with dataset means for remaining features
    input_dict = dict(defaults)           # fill all columns with means first
    input_dict["Pre_Semester_GPA"]           = pre_gpa
    input_dict["Weekly_GenAI_Hours"]          = weekly_ai
    input_dict["Traditional_Study_Hours"]     = traditional_hrs
    input_dict["Anxiety_Level_During_Exams"]  = float(anxiety)
    input_dict["Skill_Retention_Score"]       = skill_retention

    # Remove target columns if accidentally in defaults
    input_dict.pop("Post_Semester_GPA", None)
    input_dict.pop("Burnout_Risk_Level", None)

    input_df = pd.DataFrame([input_dict])

    predicted_gpa     = gpa_model.predict(input_df)[0]
    predicted_burnout_code = burnout_model.predict(input_df)[0]
    burnout_label     = BURNOUT_LABELS[int(predicted_burnout_code)]
    b_color           = BURNOUT_COLORS[burnout_label]
    b_badge           = BURNOUT_BADGES[burnout_label]
    b_emoji           = BURNOUT_EMOJI[burnout_label]

    # GPA badge
    if predicted_gpa >= 3.5:
        gpa_badge_cls, gpa_badge_txt = "badge-green", "Excellent 🎉"
    elif predicted_gpa >= 2.5:
        gpa_badge_cls, gpa_badge_txt = "badge-yellow", "Good 📈"
    else:
        gpa_badge_cls, gpa_badge_txt = "badge-red", "Needs Attention ⚠️"

    st.markdown(f"""
    <div class="card">
      <div class="card-title">📊 Prediction Results</div>
      <div class="result-grid">
        <div class="result-box">
          <div class="result-label">Predicted GPA</div>
          <div class="result-value">{predicted_gpa:.2f}</div>
          <span class="badge {gpa_badge_cls}">{gpa_badge_txt}</span>
        </div>
        <div class="result-box">
          <div class="result-label">Burnout Risk</div>
          <div class="result-value {b_color}">{burnout_label}</div>
          <span class="badge {b_badge}">{b_emoji} {burnout_label} Risk</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Tips
    st.markdown('<div class="card"><div class="card-title">💡 Insights</div>', unsafe_allow_html=True)
    tips = []
    if anxiety >= 7:
        tips.append("🧘 High anxiety detected — consider mindfulness or structured exam prep routines.")
    if weekly_ai > 20:
        tips.append("🤖 Heavy GenAI usage — make sure you're building core skills, not just shortcuts.")
    if traditional_hrs < 10:
        tips.append("📚 Traditional study hours are low — balancing AI tools with deep reading helps retention.")
    if skill_retention < 50:
        tips.append("🔁 Low skill retention — try active recall and spaced repetition techniques.")
    if predicted_gpa < 2.5:
        tips.append("📈 GPA prediction is below average — talk to your academic advisor early.")
    if not tips:
        tips.append("✅ You're on a great track! Keep balancing AI tools with active learning.")
    for t in tips:
        st.markdown(f"- {t}")
    st.markdown("</div>", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; font-size:0.72rem; color:#44446a;'>"
    "BSAI Semester 2 · AI130 Programming for AI · "
    "Zainab Qasim · Eeman Arif · Khizran Fatima</p>",
    unsafe_allow_html=True
)
