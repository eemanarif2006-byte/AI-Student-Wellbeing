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
 
BG_URL = "https://i.pinimg.com/1200x/a5/46/76/a546769fcdf6fc7cf863f09804606357.jpg"
 
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,500;0,600;1,400;1,500&family=DM+Sans:wght@300;400;500&display=swap');
 
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
 
html, body, [class*="css"] {{
    font-family: 'DM Sans', sans-serif;
    color: #3a2a1a;
}}
 
/* Background */
[data-testid="stAppViewContainer"] {{
    background-image: url('{BG_URL}');
    background-size: cover;
    background-position: center top;
    background-attachment: fixed;
    min-height: 100vh;
}}
[data-testid="stAppViewContainer"]::before {{
    content: "";
    position: fixed;
    inset: 0;
    background: rgba(245, 235, 220, 0.72);
    backdrop-filter: blur(6px);
    -webkit-backdrop-filter: blur(6px);
    z-index: 0;
}}
[data-testid="stAppViewContainer"] > section {{
    position: relative;
    z-index: 1;
}}
[data-testid="stHeader"] {{ background: transparent !important; display: none; }}
[data-testid="stToolbar"] {{ display: none !important; }}
#MainMenu, footer {{ visibility: hidden; }}
 
/* Main content width */
.block-container {{
    max-width: 580px !important;
    padding: 0 1.5rem 3rem !important;
    margin: 0 auto !important;
}}
 
/* ── HERO ── */
.hero {{
    text-align: center;
    padding: 3.5rem 0 2rem;
}}
.hero-tag {{
    display: inline-block;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.65rem;
    font-weight: 500;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #b07850;
    background: rgba(210,175,140,0.2);
    border: 1px solid rgba(180,140,100,0.3);
    padding: 0.3rem 1rem;
    border-radius: 999px;
    margin-bottom: 1.2rem;
}}
.hero-title {{
    font-family: 'Cormorant Garamond', Georgia, serif;
    font-size: clamp(2.2rem, 6vw, 3.2rem);
    font-weight: 600;
    line-height: 1.12;
    color: #2a1a0a;
    letter-spacing: -0.01em;
    margin-bottom: 0.9rem;
}}
.hero-title em {{
    font-style: italic;
    color: #c4785a;
}}
.hero-sub {{
    font-size: 0.9rem;
    color: #7a6050;
    font-weight: 300;
    line-height: 1.75;
    max-width: 380px;
    margin: 0 auto;
}}
.hero-rule {{
    width: 40px;
    height: 1.5px;
    background: linear-gradient(90deg, #d4a882, #e8c4a8);
    margin: 1.6rem auto 0;
    border-radius: 2px;
}}
 
/* ── CARD ── */
.card {{
    background: rgba(255, 250, 243, 0.78);
    border: 1px solid rgba(200, 170, 130, 0.28);
    border-radius: 18px;
    padding: 1.6rem 1.6rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    transition: box-shadow 0.25s ease;
}}
.card:hover {{
    box-shadow: 0 6px 28px rgba(180,130,90,0.12);
}}
.card-title {{
    font-family: 'DM Sans', sans-serif;
    font-size: 0.6rem;
    font-weight: 500;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #c4a07a;
    margin-bottom: 1.4rem;
}}
 
/* ── FIELD ROW ── */
.field-label {{
    font-family: 'Cormorant Garamond', serif;
    font-size: 1rem;
    font-weight: 500;
    color: #2a1a0a;
    line-height: 1;
    margin-bottom: 1px;
}}
.field-hint {{
    font-family: 'DM Sans', sans-serif;
    font-size: 0.7rem;
    color: #b09070;
    font-weight: 300;
    margin-bottom: 0.55rem;
}}
.field-sep {{
    height: 1px;
    background: rgba(200,170,130,0.18);
    margin: 0.9rem 0;
}}
 
/* Override Streamlit number input */
[data-testid="stNumberInput"] {{
    width: 100% !important;
}}
[data-testid="stNumberInput"] > div {{
    width: 100% !important;
}}
[data-testid="stNumberInput"] input {{
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 1.05rem !important;
    font-weight: 500 !important;
    color: #2a1a0a !important;
    background: rgba(255,248,238,0.7) !important;
    border: 1px solid rgba(200,170,130,0.35) !important;
    border-radius: 10px !important;
    padding: 0.5rem 0.75rem !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
    width: 100% !important;
    text-align: center !important;
}}
[data-testid="stNumberInput"] input:focus,
[data-testid="stNumberInput"] input:hover {{
    border-color: rgba(180,120,80,0.55) !important;
    box-shadow: 0 0 0 3px rgba(200,150,100,0.12) !important;
    outline: none !important;
}}
[data-testid="stNumberInput"] button {{
    background: rgba(220,190,155,0.25) !important;
    border: 1px solid rgba(200,170,130,0.3) !important;
    color: #b07850 !important;
    border-radius: 8px !important;
    transition: background 0.18s, transform 0.12s !important;
    font-size: 1rem !important;
}}
[data-testid="stNumberInput"] button:hover {{
    background: rgba(200,160,110,0.38) !important;
    transform: scale(1.08) !important;
}}
[data-testid="stNumberInput"] label {{ display: none !important; }}
 
/* ── PREDICT BUTTON ── */
.stButton > button {{
    width: 100% !important;
    background: linear-gradient(135deg, #d4956a, #e8b490) !important;
    color: #fff8f0 !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    padding: 0.75rem 1rem !important;
    box-shadow: 0 3px 16px rgba(180,110,70,0.25) !important;
    transition: transform 0.18s, box-shadow 0.18s, opacity 0.18s !important;
    cursor: pointer !important;
}}
.stButton > button:hover {{
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 22px rgba(180,110,70,0.32) !important;
    opacity: 0.95 !important;
}}
.stButton > button:active {{
    transform: translateY(0px) !important;
}}
 
/* ── RESULTS ── */
.results-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.85rem;
    margin-top: 0.4rem;
}}
.res-box {{
    background: rgba(255,248,238,0.85);
    border: 1px solid rgba(200,170,130,0.25);
    border-radius: 14px;
    padding: 1.1rem 0.8rem 0.9rem;
    text-align: center;
    transition: transform 0.2s, box-shadow 0.2s;
}}
.res-box:hover {{
    transform: translateY(-2px);
    box-shadow: 0 5px 18px rgba(180,130,90,0.13);
}}
.res-lbl {{
    font-family: 'DM Sans', sans-serif;
    font-size: 0.58rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #c4a07a;
    font-weight: 500;
    margin-bottom: 0.4rem;
}}
.res-val {{
    font-family: 'Cormorant Garamond', serif;
    font-size: 2.2rem;
    font-weight: 600;
    color: #2a1a0a;
    line-height: 1;
    margin-bottom: 0.4rem;
}}
.res-val.high   {{ color: #b84040; }}
.res-val.medium {{ color: #b07830; }}
.res-val.low    {{ color: #3a8060; }}
 
.pill {{
    display: inline-block;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.65rem;
    font-weight: 500;
    letter-spacing: 0.06em;
    padding: 0.18rem 0.7rem;
    border-radius: 999px;
}}
.pill-rose   {{ background: #fceef0; color: #8e3040; border: 1px solid #f0c8cc; }}
.pill-amber  {{ background: #fef6e8; color: #7a5010; border: 1px solid #f0ddb0; }}
.pill-sage   {{ background: #eef5f0; color: #2a6845; border: 1px solid #b8ddc8; }}
 
/* ── FOOTER ── */
.footer {{
    text-align: center;
    padding: 1rem 0 2.5rem;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.62rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #c4a888;
}}
.footer-rule {{
    height: 1px;
    background: rgba(200,170,130,0.2);
    margin-bottom: 1.2rem;
}}
</style>
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
 
# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-tag">BSAI &nbsp;·&nbsp; AI130 &nbsp;·&nbsp; Semester 2</div>
  <h1 class="hero-title">Know Your <em>Academic</em><br>Future, Now.</h1>
  <p class="hero-sub">Enter your study habits and let our machine learning models predict your GPA and burnout risk — trained on 50,000 student records.</p>
  <div class="hero-rule"></div>
</div>
""", unsafe_allow_html=True)
 
if not models_ok:
    st.error(f"Could not load models. Make sure gpa_model.pkl, burnout_model.pkl, scaler.pkl and cleaned_students_data.csv are in the same folder as app.py.\n\n{load_err}")
    st.stop()
 
# ── Input helper ──────────────────────────────────────────────────────────────
def input_field(label, hint, key, mn, mx, step, default):
    if key not in st.session_state:
        st.session_state[key] = float(default)
    st.markdown(f'<div class="field-label">{label}</div><div class="field-hint">{hint}</div>', unsafe_allow_html=True)
    fmt = "%.1f" if step < 1 else "%.0f"
    val = st.number_input(
        label, label_visibility="collapsed",
        min_value=float(mn), max_value=float(mx),
        value=float(st.session_state[key]),
        step=float(step), key=f"{key}_num", format=fmt,
    )
    st.session_state[key] = val
    return val
 
# ── Profile card ──────────────────────────────────────────────────────────────
st.markdown('<div class="card"><div class="card-title">Student Profile</div>', unsafe_allow_html=True)
 
pre_gpa   = input_field("Pre-Semester GPA",        "Scale: 0.0 – 4.0",              "pre_gpa",   0.0,  4.0,  0.1, round(defaults.get("Pre_Semester_GPA", 2.5), 1))
st.markdown('<div class="field-sep"></div>', unsafe_allow_html=True)
weekly_ai = input_field("Weekly GenAI Hours",       "Hours per week using AI tools", "weekly_ai", 0.0, 40.0,  0.5, round(defaults.get("Weekly_GenAI_Hours", 5.0), 1))
st.markdown('<div class="field-sep"></div>', unsafe_allow_html=True)
trad_hrs  = input_field("Traditional Study Hours",  "Hours per week without AI",     "trad_hrs",  0.0, 60.0,  0.5, round(defaults.get("Traditional_Study_Hours", 15.0), 1))
st.markdown('<div class="field-sep"></div>', unsafe_allow_html=True)
anxiety   = input_field("Exam Anxiety Level",       "1 = very calm   —   10 = extreme", "anxiety", 1.0, 10.0, 1.0, float(int(round(defaults.get("Anxiety_Level_During_Exams", 5)))))
st.markdown('<div class="field-sep"></div>', unsafe_allow_html=True)
skill_ret = input_field("Skill Retention Score",    "0 = poor   —   100 = excellent","skill_ret", 0.0,100.0,  1.0, round(defaults.get("Skill_Retention_Score", 60.0), 1))
 
st.markdown('</div>', unsafe_allow_html=True)
 
# ── Predict ───────────────────────────────────────────────────────────────────
BURNOUT_MAP   = {0: "High",   1: "Low",    2: "Medium"}
BURNOUT_CLASS = {"High": "high", "Medium": "medium", "Low": "low"}
BURNOUT_PILL  = {"High": "pill-rose", "Medium": "pill-amber", "Low": "pill-sage"}
BURNOUT_TEXT  = {"High": "High Risk", "Medium": "Moderate", "Low": "Low Risk"}
 
if st.button("Predict GPA & Burnout Risk"):
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
    b_label      = BURNOUT_MAP[burnout_code]
    b_class      = BURNOUT_CLASS[b_label]
    b_pill       = BURNOUT_PILL[b_label]
    b_text       = BURNOUT_TEXT[b_label]
 
    if pred_gpa >= 3.5:
        g_pill, g_txt = "pill-sage",  "Excellent"
    elif pred_gpa >= 2.5:
        g_pill, g_txt = "pill-amber", "On Track"
    else:
        g_pill, g_txt = "pill-rose",  "Needs Attention"
 
    st.markdown(f"""
    <div class="card" style="margin-top:0.6rem;">
      <div class="card-title">Results</div>
      <div class="results-grid">
        <div class="res-box">
          <div class="res-lbl">Predicted GPA</div>
          <div class="res-val">{pred_gpa:.2f}</div>
          <span class="pill {g_pill}">{g_txt}</span>
        </div>
        <div class="res-box">
          <div class="res-lbl">Burnout Risk</div>
          <div class="res-val {b_class}">{b_label}</div>
          <span class="pill {b_pill}">{b_text}</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)
 
# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  <div class="footer-rule"></div>
  StudyPulse &nbsp;·&nbsp; BSAI Semester 2 &nbsp;·&nbsp; AI130 Programming for AI<br>
  Zainab Qasim &nbsp;·&nbsp; Eeman Arif &nbsp;·&nbsp; Khizran Fatima
</div>
""", unsafe_allow_html=True)
