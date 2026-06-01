import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import base64
 
st.set_page_config(
    page_title="StudyPulse · AI Student Predictor",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="collapsed",
)
 
# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Nunito:wght@300;400;500;600&display=swap');
 
html, body, [class*="css"] {
    font-family: 'Nunito', sans-serif;
    color: #1a1a2e;
}
 
.stApp {
    background: transparent;
}
 
/* Background image set via inline style below */
[data-testid="stAppViewContainer"] {
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}
 
[data-testid="stAppViewContainer"]::before {
    content: "";
    position: fixed;
    inset: 0;
    background: rgba(255,255,255,0.82);
    backdrop-filter: blur(2px);
    z-index: 0;
}
 
[data-testid="stAppViewContainer"] > * {
    position: relative;
    z-index: 1;
}
 
[data-testid="stHeader"] { background: transparent !important; }
 
/* ── HERO ── */
.hero-wrap {
    text-align: center;
    padding: 3rem 1rem 1.5rem;
}
.hero-tag {
    display: inline-block;
    background: #e8e0ff;
    color: #5b3fc4;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    padding: 0.3rem 1rem;
    border-radius: 999px;
    margin-bottom: 1.1rem;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2rem, 6vw, 3.2rem);
    font-weight: 800;
    line-height: 1.1;
    color: #16094a;
    margin: 0 0 0.7rem;
    letter-spacing: -0.03em;
}
.hero-title em {
    font-style: normal;
    background: linear-gradient(135deg, #7c5cbf, #c47fc4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    font-size: 1rem;
    color: #5a5a7a;
    font-weight: 400;
    max-width: 480px;
    margin: 0 auto;
    line-height: 1.6;
}
 
/* ── CARDS ── */
.glass-card {
    background: rgba(255,255,255,0.72);
    border: 1.5px solid rgba(180,160,255,0.25);
    border-radius: 20px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.2rem;
    backdrop-filter: blur(12px);
    box-shadow: 0 4px 24px rgba(92,63,196,0.06);
}
.card-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #9b87c8;
    margin-bottom: 1.2rem;
    font-weight: 700;
}
 
/* ── STEPPER ── */
.stepper-row {
    display: flex;
    align-items: center;
    gap: 0;
    margin-bottom: 1.1rem;
}
.stepper-label {
    font-size: 0.88rem;
    font-weight: 600;
    color: #2e2060;
    flex: 1;
}
.stepper-sublabel {
    font-size: 0.73rem;
    color: #9b87c8;
    font-weight: 400;
}
.stepper-controls {
    display: flex;
    align-items: center;
    gap: 0;
    background: rgba(92,63,196,0.07);
    border-radius: 12px;
    overflow: hidden;
    border: 1.5px solid rgba(92,63,196,0.15);
}
.step-btn {
    background: none;
    border: none;
    width: 36px;
    height: 36px;
    font-size: 1.1rem;
    font-weight: 700;
    color: #5b3fc4;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: background 0.15s;
}
.step-btn:hover { background: rgba(92,63,196,0.13); }
.step-val {
    font-family: 'Syne', sans-serif;
    font-size: 0.95rem;
    font-weight: 700;
    color: #16094a;
    min-width: 52px;
    text-align: center;
}
 
/* ── BG PICKER ── */
.bg-option {
    display: inline-flex;
    flex-direction: column;
    align-items: center;
    gap: 6px;
    cursor: pointer;
    margin: 0 6px 6px 0;
}
.bg-swatch {
    width: 48px;
    height: 48px;
    border-radius: 12px;
    border: 2.5px solid transparent;
    transition: border-color 0.2s, transform 0.15s;
    background-size: cover;
    background-position: center;
}
.bg-swatch:hover { transform: scale(1.07); }
.bg-swatch.active { border-color: #7c5cbf; }
.bg-name {
    font-size: 0.65rem;
    color: #9b87c8;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
 
/* ── RESULTS ── */
.result-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
    margin-top: 0.5rem;
}
.result-box {
    background: rgba(255,255,255,0.85);
    border-radius: 16px;
    padding: 1.4rem 1rem;
    text-align: center;
    border: 1.5px solid rgba(180,160,255,0.2);
}
.result-label {
    font-size: 0.7rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #9b87c8;
    font-weight: 700;
    margin-bottom: 0.5rem;
}
.result-value {
    font-family: 'Syne', sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    line-height: 1;
    color: #16094a;
}
.burnout-high   { color: #c44a4a !important; }
.burnout-medium { color: #c4954a !important; }
.burnout-low    { color: #3a9e6f !important; }
 
/* Badge pills */
.pill {
    display: inline-block;
    padding: 0.22rem 0.85rem;
    border-radius: 999px;
    font-size: 0.73rem;
    font-weight: 700;
    margin-top: 0.5rem;
    letter-spacing: 0.04em;
}
.pill-green  { background: #e0f7ee; color: #1a7a4a; }
.pill-yellow { background: #fff5e0; color: #8a5c00; }
.pill-red    { background: #fce8e8; color: #9e2222; }
 
/* Tip rows */
.tip-row {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 0.65rem 0;
    border-bottom: 1px solid rgba(180,160,255,0.13);
    font-size: 0.88rem;
    color: #3a3060;
    line-height: 1.5;
}
.tip-row:last-child { border-bottom: none; }
.tip-icon { font-size: 1.1rem; flex-shrink: 0; margin-top: 1px; }
 
/* ── BUTTON ── */
.stButton > button {
    background: linear-gradient(135deg, #7c5cbf, #b07fd4) !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 0.9rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em !important;
    padding: 0.75rem 2rem !important;
    width: 100% !important;
    box-shadow: 0 4px 18px rgba(124,92,191,0.3) !important;
    transition: opacity 0.2s, transform 0.15s !important;
}
.stButton > button:hover {
    opacity: 0.9 !important;
    transform: translateY(-1px) !important;
}
 
/* ── DIVIDER & FOOTER ── */
hr { border-color: rgba(180,160,255,0.2) !important; }
.footer-txt {
    text-align: center;
    font-size: 0.72rem;
    color: #b0a0cc;
    padding-bottom: 2rem;
}
 
/* Streamlit widget labels */
.stSlider > label, .stSelectbox > label, .stNumberInput > label {
    font-size: 0.84rem !important;
    color: #2e2060 !important;
    font-weight: 600 !important;
}
 
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)
 
# ── Background picker (Unsplash sourced via direct URL) ───────────────────────
BG_OPTIONS = {
    "Aurora":    "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=1600&q=80",
    "Bloom":     "https://images.unsplash.com/photo-1490750967868-88df5691cc6a?w=1600&q=80",
    "Cosmos":    "https://images.unsplash.com/photo-1419242902214-272b3f66ee7a?w=1600&q=80",
    "Studio":    "https://images.unsplash.com/photo-1558618666-fcd25c85cd64?w=1600&q=80",
    "Minimal":   "https://images.unsplash.com/photo-1557683316-973673baf926?w=1600&q=80",
    "Forest":    "https://images.unsplash.com/photo-1448375240586-882707db888b?w=1600&q=80",
}
 
if "bg_choice" not in st.session_state:
    st.session_state.bg_choice = "Aurora"
 
# Inject chosen background
bg_url = BG_OPTIONS[st.session_state.bg_choice]
st.markdown(f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-image: url('{bg_url}');
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}}
</style>
""", unsafe_allow_html=True)
 
# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
  <div class="hero-tag">✦ BSAI · AI130 · Semester 2</div>
  <h1 class="hero-title">Know Your <em>Academic</em><br>Future, Now.</h1>
  <p class="hero-sub">Enter your study habits and let our ML models predict your GPA and burnout risk — powered by 50,000 real student records.</p>
</div>
""", unsafe_allow_html=True)
 
# ── Background Picker ─────────────────────────────────────────────────────────
st.markdown('<div class="glass-card"><div class="card-label">✦ Choose Your Vibe</div>', unsafe_allow_html=True)
 
cols = st.columns(len(BG_OPTIONS))
for i, (name, url) in enumerate(BG_OPTIONS.items()):
    with cols[i]:
        active_cls = "active" if st.session_state.bg_choice == name else ""
        if st.button(name, key=f"bg_{name}"):
            st.session_state.bg_choice = name
            st.rerun()
        st.markdown(f"""
        <div style="text-align:center; margin-top:-10px;">
          <div class="bg-swatch {active_cls}" style="background-image:url('{url}'); margin: 0 auto;"></div>
        </div>
        """, unsafe_allow_html=True)
 
st.markdown('</div>', unsafe_allow_html=True)
 
# ── Load Models ───────────────────────────────────────────────────────────────
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
    models_loaded = True
except Exception as e:
    models_loaded = False
    load_error = str(e)
 
if not models_loaded:
    st.error(f"Could not load models. Ensure gpa_model.pkl, burnout_model.pkl, scaler.pkl and cleaned_students_data.csv are in the same folder.\n\n{load_error}")
    st.stop()
 
# ── Stepper helper ────────────────────────────────────────────────────────────
def stepper(label, sublabel, key, min_val, max_val, step, default, fmt="{:.2f}"):
    if key not in st.session_state:
        st.session_state[key] = float(default)
 
    col_label, col_minus, col_val, col_plus = st.columns([3, 0.5, 0.8, 0.5])
 
    with col_label:
        st.markdown(f'<div class="stepper-label">{label}<br><span class="stepper-sublabel">{sublabel}</span></div>', unsafe_allow_html=True)
    with col_minus:
        if st.button("−", key=f"{key}_minus"):
            st.session_state[key] = max(min_val, round(st.session_state[key] - step, 10))
    with col_val:
        display = fmt.format(st.session_state[key]) if isinstance(fmt, str) else str(int(st.session_state[key]))
        st.markdown(f'<div class="step-val" style="padding:6px 0; text-align:center;">{display}</div>', unsafe_allow_html=True)
    with col_plus:
        if st.button("+", key=f"{key}_plus"):
            st.session_state[key] = min(max_val, round(st.session_state[key] + step, 10))
 
    return st.session_state[key]
 
# ── Inputs ────────────────────────────────────────────────────────────────────
st.markdown('<div class="glass-card"><div class="card-label">✦ Your Student Profile</div>', unsafe_allow_html=True)
 
pre_gpa        = stepper("Pre-Semester GPA",         "Scale: 0.0 – 4.0",         "pre_gpa",    0.0, 4.0,  0.1,  round(defaults.get("Pre_Semester_GPA", 2.5), 1))
weekly_ai      = stepper("Weekly GenAI Hours",        "Hours/week using AI tools", "weekly_ai",  0.0, 40.0, 0.5,  round(defaults.get("Weekly_GenAI_Hours", 5.0), 1))
trad_hrs       = stepper("Traditional Study Hours",   "Hours/week without AI",     "trad_hrs",   0.0, 60.0, 0.5,  round(defaults.get("Traditional_Study_Hours", 15.0), 1))
anxiety        = stepper("Exam Anxiety Level",        "1 = calm · 10 = very high", "anxiety",    1,   10,   1,    int(round(defaults.get("Anxiety_Level_During_Exams", 5))), fmt="{:.0f}")
skill_ret      = stepper("Skill Retention Score",     "0 = poor · 100 = excellent","skill_ret",  0.0, 100.0,1.0,  round(defaults.get("Skill_Retention_Score", 60.0), 1))
 
st.markdown('</div>', unsafe_allow_html=True)
 
# ── Predict ───────────────────────────────────────────────────────────────────
BURNOUT_LABELS = {0: "High", 1: "Low", 2: "Medium"}
BURNOUT_COLOR  = {"High": "burnout-high", "Medium": "burnout-medium", "Low": "burnout-low"}
BURNOUT_PILL   = {"High": "pill-red",     "Medium": "pill-yellow",    "Low": "pill-green"}
BURNOUT_EMOJI  = {"High": "🔴",           "Medium": "🟡",             "Low": "🟢"}
 
predict = st.button("✦ Predict My GPA & Burnout Risk")
 
if predict:
    input_dict = dict(defaults)
    input_dict.pop("Post_Semester_GPA", None)
    input_dict.pop("Burnout_Risk_Level", None)
    input_dict["Pre_Semester_GPA"]          = pre_gpa
    input_dict["Weekly_GenAI_Hours"]         = weekly_ai
    input_dict["Traditional_Study_Hours"]    = trad_hrs
    input_dict["Anxiety_Level_During_Exams"] = float(anxiety)
    input_dict["Skill_Retention_Score"]      = skill_ret
 
    input_df = pd.DataFrame([input_dict])
 
    predicted_gpa = gpa_model.predict(input_df)[0]
    burnout_code  = int(burnout_model.predict(input_df)[0])
    burnout_label = BURNOUT_LABELS[burnout_code]
 
    b_color = BURNOUT_COLOR[burnout_label]
    b_pill  = BURNOUT_PILL[burnout_label]
    b_emoji = BURNOUT_EMOJI[burnout_label]
 
    if predicted_gpa >= 3.5:
        g_pill, g_txt = "pill-green", "Excellent 🎉"
    elif predicted_gpa >= 2.5:
        g_pill, g_txt = "pill-yellow", "On Track 📈"
    else:
        g_pill, g_txt = "pill-red", "Needs Attention ⚠️"
 
    st.markdown(f"""
    <div class="glass-card">
      <div class="card-label">✦ Your Results</div>
      <div class="result-grid">
        <div class="result-box">
          <div class="result-label">Predicted GPA</div>
          <div class="result-value">{predicted_gpa:.2f}</div>
          <span class="pill {g_pill}">{g_txt}</span>
        </div>
        <div class="result-box">
          <div class="result-label">Burnout Risk</div>
          <div class="result-value {b_color}">{burnout_label}</div>
          <span class="pill {b_pill}">{b_emoji} {burnout_label} Risk</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)
 
    # Tips
    tips = []
    if anxiety >= 7:
        tips.append(("🧘", "Your anxiety is high — try structured exam prep and mindfulness to stay grounded."))
    if weekly_ai > 20:
        tips.append(("🤖", "Heavy GenAI usage detected — make sure you're building real skills, not just shortcuts."))
    if trad_hrs < 10:
        tips.append(("📚", "Low traditional study hours — balancing AI with deep reading improves retention a lot."))
    if skill_ret < 50:
        tips.append(("🔁", "Low skill retention — try active recall and spaced repetition to lock in what you learn."))
    if predicted_gpa < 2.5:
        tips.append(("📈", "GPA projection is below average — consider reaching out to your academic advisor early."))
    if burnout_label == "High":
        tips.append(("💆", "High burnout risk — rest is productive too. Schedule breaks and protect your sleep."))
    if not tips:
        tips.append(("✅", "You're on a great track! Keep balancing AI tools with active learning and self-care."))
 
    st.markdown('<div class="glass-card"><div class="card-label">✦ Personal Insights</div>', unsafe_allow_html=True)
    for icon, text in tips:
        st.markdown(f'<div class="tip-row"><span class="tip-icon">{icon}</span><span>{text}</span></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
 
# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<p class="footer-txt">StudyPulse · BSAI Semester 2 · AI130 Programming for AI<br>'
    'Zainab Qasim · Eeman Arif · Khizran Fatima</p>',
    unsafe_allow_html=True,
)
 

