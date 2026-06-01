import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
 
st.set_page_config(
    page_title="StudyPulse · AI Student Predictor",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="collapsed",
)
 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;0,700;1,400;1,600&family=Lato:wght@300;400;700&display=swap');
 
html, body, [class*="css"] {
    font-family: 'Lato', sans-serif;
    color: #2d2040;
}
 
[data-testid="stAppViewContainer"] {
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}
 
[data-testid="stAppViewContainer"]::before {
    content: "";
    position: fixed;
    inset: 0;
    background: rgba(255,248,252,0.78);
    backdrop-filter: blur(3px);
    z-index: 0;
}
 
[data-testid="stAppViewContainer"] > * { position: relative; z-index: 1; }
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stToolbar"] { display: none; }
 
/* ── URL INPUT ── */
.url-wrap {
    margin: 0 auto 0.5rem;
    max-width: 560px;
    text-align: center;
}
.url-hint {
    font-size: 0.72rem;
    color: #c0a8c8;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
    font-family: 'Lato', sans-serif;
}
 
/* ── HERO ── */
.hero-wrap {
    text-align: center;
    padding: 2.8rem 1rem 1.6rem;
}
.hero-eyebrow {
    font-family: 'Lato', sans-serif;
    font-size: 0.68rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: #c4a0c8;
    margin-bottom: 0.9rem;
    font-weight: 700;
}
.hero-title {
    font-family: 'Playfair Display', 'Times New Roman', serif;
    font-size: clamp(2rem, 5.5vw, 3rem);
    font-weight: 700;
    line-height: 1.15;
    color: #2d1a40;
    margin: 0 0 0.8rem;
    letter-spacing: -0.01em;
}
.hero-title em {
    font-style: italic;
    color: #9b5fc4;
}
.hero-sub {
    font-family: 'Lato', sans-serif;
    font-size: 0.95rem;
    color: #8a7aa0;
    font-weight: 300;
    max-width: 440px;
    margin: 0 auto;
    line-height: 1.7;
}
.hero-divider {
    width: 48px;
    height: 2px;
    background: linear-gradient(90deg, #d4a8e8, #f0c8d8);
    margin: 1.4rem auto 0;
    border-radius: 2px;
}
 
/* ── CARDS ── */
.glass-card {
    background: rgba(255,250,255,0.75);
    border: 1px solid rgba(210,180,230,0.35);
    border-radius: 20px;
    padding: 1.5rem 1.8rem;
    margin-bottom: 1.1rem;
    backdrop-filter: blur(14px);
    box-shadow: 0 2px 20px rgba(180,140,210,0.08);
}
.card-label {
    font-family: 'Lato', sans-serif;
    font-size: 0.65rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #c4a0c8;
    margin-bottom: 1.1rem;
    font-weight: 700;
}
 
/* ── STEPPER ROW ── */
.step-row {
    display: flex;
    align-items: center;
    margin-bottom: 0.85rem;
    gap: 12px;
}
.step-meta { flex: 1; }
.step-name {
    font-family: 'Playfair Display', serif;
    font-size: 0.9rem;
    font-weight: 600;
    color: #2d1a40;
    line-height: 1.3;
}
.step-hint {
    font-size: 0.7rem;
    color: #c0a8c8;
    font-weight: 300;
    margin-top: 1px;
}
.step-ctrl {
    display: flex;
    align-items: center;
    background: rgba(220,200,240,0.18);
    border: 1px solid rgba(200,170,220,0.3);
    border-radius: 10px;
    overflow: hidden;
}
.s-btn {
    background: none;
    border: none;
    width: 30px;
    height: 32px;
    font-size: 1rem;
    font-weight: 600;
    color: #9b5fc4;
    cursor: pointer;
    line-height: 1;
}
.s-btn:hover { background: rgba(180,140,210,0.15); }
.s-val {
    font-family: 'Playfair Display', serif;
    font-size: 0.88rem;
    font-weight: 600;
    color: #2d1a40;
    min-width: 46px;
    text-align: center;
}
 
/* ── RESULTS ── */
.result-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.9rem;
    margin-top: 0.4rem;
}
.res-box {
    background: rgba(255,252,255,0.88);
    border-radius: 14px;
    padding: 1rem 0.8rem;
    text-align: center;
    border: 1px solid rgba(210,180,230,0.3);
}
.res-lbl {
    font-family: 'Lato', sans-serif;
    font-size: 0.62rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #c4a0c8;
    font-weight: 700;
    margin-bottom: 0.35rem;
}
.res-val {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 700;
    color: #2d1a40;
    line-height: 1;
}
.burnout-high   { color: #c44a5a !important; }
.burnout-medium { color: #b07020 !important; }
.burnout-low    { color: #3a8e6a !important; }
 
.pill {
    display: inline-block;
    padding: 0.18rem 0.75rem;
    border-radius: 999px;
    font-size: 0.68rem;
    font-weight: 700;
    margin-top: 0.4rem;
    letter-spacing: 0.05em;
    font-family: 'Lato', sans-serif;
}
.pill-green  { background: #eaf7f0; color: #1a6e45; border: 1px solid #c0e8d4; }
.pill-yellow { background: #fef9ec; color: #7a5000; border: 1px solid #f0dfa0; }
.pill-red    { background: #fceef0; color: #8e2233; border: 1px solid #f0c0c8; }
 
/* ── BUTTON ── */
.stButton > button {
    background: linear-gradient(135deg, #c49ad8, #e8b4d0) !important;
    color: #2d1a40 !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Playfair Display', serif !important;
    font-size: 0.92rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.03em !important;
    padding: 0.7rem 2rem !important;
    width: 100% !important;
    box-shadow: 0 3px 14px rgba(180,140,210,0.28) !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.88 !important; }
 
/* number input styling */
[data-testid="stNumberInput"] input {
    font-family: 'Playfair Display', serif !important;
    font-size: 0.88rem !important;
    color: #2d1a40 !important;
    border-color: rgba(200,170,220,0.4) !important;
    border-radius: 8px !important;
    background: rgba(255,252,255,0.8) !important;
}
[data-testid="stNumberInput"] label {
    font-family: 'Playfair Display', serif !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    color: #2d1a40 !important;
}
 
.stTextInput input {
    font-family: 'Lato', sans-serif !important;
    font-size: 0.84rem !important;
    border-radius: 10px !important;
    border-color: rgba(200,170,220,0.4) !important;
    background: rgba(255,252,255,0.8) !important;
    color: #2d1a40 !important;
}
.stTextInput label { display: none; }
 
hr { border-color: rgba(210,180,230,0.25) !important; }
.footer-txt {
    text-align: center;
    font-size: 0.68rem;
    color: #c4b0d0;
    letter-spacing: 0.08em;
    padding-bottom: 2rem;
    font-family: 'Lato', sans-serif;
    text-transform: uppercase;
}
 
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)
 
# ── Background URL input ──────────────────────────────────────────────────────
if "bg_url" not in st.session_state:
    st.session_state.bg_url = ""
 
st.markdown('<div class="url-wrap"><div class="url-hint">✦ Paste a background image URL (Pinterest, Unsplash, etc.)</div></div>', unsafe_allow_html=True)
bg_url_input = st.text_input("bg", value=st.session_state.bg_url, placeholder="https://i.pinimg.com/...")
if bg_url_input != st.session_state.bg_url:
    st.session_state.bg_url = bg_url_input
 
if st.session_state.bg_url.strip():
    st.markdown(f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background-image: url('{st.session_state.bg_url.strip()}');
    }}
    </style>
    """, unsafe_allow_html=True)
 
# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
  <div class="hero-eyebrow">✦ &nbsp; BSAI · AI130 · Semester 2 &nbsp; ✦</div>
  <h1 class="hero-title">Know Your <em>Academic</em><br>Future, Now.</h1>
  <p class="hero-sub">Enter your study habits and let our machine learning models predict your GPA & burnout risk — trained on 50,000 student records.</p>
  <div class="hero-divider"></div>
</div>
""", unsafe_allow_html=True)
 
# ── Load models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    base = os.path.dirname(__file__)
    return (
        joblib.load(os.path.join(base, "gpa_model.pkl")),
        joblib.load(os.path.join(base, "burnout_model.pkl")),
        joblib.load(os.path.join(base, "scaler.pkl")),
    )
 
@st.cache_data
def load_defaults():
    base = os.path.dirname(__file__)
    df = pd.read_csv(os.path.join(base, "cleaned_students_data.csv"))
    return df.drop(columns=["Post_Semester_GPA", "Burnout_Risk_Level"], errors="ignore").mean().to_dict()
 
try:
    gpa_model, burnout_model, scaler = load_models()
    defaults = load_defaults()
    models_ok = True
except Exception as e:
    models_ok = False
    load_err = str(e)
 
if not models_ok:
    st.error(f"Could not load models: {load_err}")
    st.stop()
 
# ── Stepper + number_input combo ─────────────────────────────────────────────
def field(label, hint, key, mn, mx, step, default):
    if key not in st.session_state:
        st.session_state[key] = float(default)
 
    # Label + hint
    st.markdown(f"""
    <div class="step-row" style="margin-bottom:4px;">
      <div class="step-meta">
        <div class="step-name">{label}</div>
        <div class="step-hint">{hint}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)
 
    col_m, col_n, col_p = st.columns([0.12, 0.76, 0.12])
    with col_m:
        if st.button("−", key=f"{key}_m"):
            st.session_state[key] = round(max(mn, st.session_state[key] - step), 4)
    with col_n:
        val = st.number_input(
            label, label_visibility="collapsed",
            min_value=float(mn), max_value=float(mx),
            value=float(st.session_state[key]),
            step=float(step), key=f"{key}_num",
            format="%.2f" if step < 1 else "%.0f",
        )
        st.session_state[key] = val
    with col_p:
        if st.button("+", key=f"{key}_p"):
            st.session_state[key] = round(min(mx, st.session_state[key] + step), 4)
 
    st.markdown("<div style='margin-bottom:0.5rem;'></div>", unsafe_allow_html=True)
    return st.session_state[key]
 
# ── Input card ────────────────────────────────────────────────────────────────
st.markdown('<div class="glass-card"><div class="card-label">✦ Your Student Profile</div>', unsafe_allow_html=True)
 
pre_gpa   = field("Pre-Semester GPA",          "0.0 – 4.0",                "pre_gpa",  0.0,  4.0,  0.1, round(defaults.get("Pre_Semester_GPA", 2.5), 1))
weekly_ai = field("Weekly GenAI Hours",         "Hours/week using AI tools","weekly_ai",0.0, 40.0,  0.5, round(defaults.get("Weekly_GenAI_Hours", 5.0), 1))
trad_hrs  = field("Traditional Study Hours",    "Hours/week without AI",    "trad_hrs", 0.0, 60.0,  0.5, round(defaults.get("Traditional_Study_Hours", 15.0), 1))
anxiety   = field("Exam Anxiety Level",         "1 = calm · 10 = extreme",  "anxiety",  1.0, 10.0,  1.0, float(int(round(defaults.get("Anxiety_Level_During_Exams", 5)))))
skill_ret = field("Skill Retention Score",      "0 = poor · 100 = great",   "skill_ret",0.0,100.0,  1.0, round(defaults.get("Skill_Retention_Score", 60.0), 1))
 
st.markdown("</div>", unsafe_allow_html=True)
 
# ── Predict ───────────────────────────────────────────────────────────────────
BURNOUT_LABELS = {0: "High", 1: "Low", 2: "Medium"}
BURNOUT_COLOR  = {"High": "burnout-high", "Medium": "burnout-medium", "Low": "burnout-low"}
BURNOUT_PILL   = {"High": "pill-red",     "Medium": "pill-yellow",    "Low": "pill-green"}
BURNOUT_EMOJI  = {"High": "↑ High",       "Medium": "~ Medium",       "Low": "↓ Low"}
 
if st.button("✦ Predict My GPA & Burnout Risk"):
    inp = dict(defaults)
    inp.pop("Post_Semester_GPA", None)
    inp.pop("Burnout_Risk_Level", None)
    inp["Pre_Semester_GPA"]           = pre_gpa
    inp["Weekly_GenAI_Hours"]          = weekly_ai
    inp["Traditional_Study_Hours"]     = trad_hrs
    inp["Anxiety_Level_During_Exams"]  = float(anxiety)
    inp["Skill_Retention_Score"]       = skill_ret
 
    df_in = pd.DataFrame([inp])
 
    pred_gpa     = gpa_model.predict(df_in)[0]
    burnout_code = int(burnout_model.predict(df_in)[0])
    b_label      = BURNOUT_LABELS[burnout_code]
    b_color      = BURNOUT_COLOR[b_label]
    b_pill       = BURNOUT_PILL[b_label]
    b_emoji      = BURNOUT_EMOJI[b_label]
 
    if pred_gpa >= 3.5:
        g_pill, g_txt = "pill-green",  "Excellent"
    elif pred_gpa >= 2.5:
        g_pill, g_txt = "pill-yellow", "On Track"
    else:
        g_pill, g_txt = "pill-red",    "Needs Attention"
 
    st.markdown(f"""
    <div class="glass-card" style="margin-top:0.3rem;">
      <div class="card-label">✦ Results</div>
      <div class="result-row">
        <div class="res-box">
          <div class="res-lbl">Predicted GPA</div>
          <div class="res-val">{pred_gpa:.2f}</div>
          <span class="pill {g_pill}">{g_txt}</span>
        </div>
        <div class="res-box">
          <div class="res-lbl">Burnout Risk</div>
          <div class="res-val {b_color}">{b_label}</div>
          <span class="pill {b_pill}">{b_emoji}</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)
 
# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<p class="footer-txt">StudyPulse &nbsp;·&nbsp; BSAI Semester 2 &nbsp;·&nbsp; AI130 Programming for AI<br>'
    'Zainab Qasim &nbsp;·&nbsp; Eeman Arif &nbsp;·&nbsp; Khizran Fatima</p>',
    unsafe_allow_html=True,
)
