import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
 
st.set_page_config(
    page_title="StudyPulse · AI Student Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)
 
BG_URL = "https://i.pinimg.com/1200x/a5/46/76/a546769fcdf6fc7cf863f09804606357.jpg"
 
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,500;0,600;1,400;1,500&family=DM+Sans:wght@300;400;500&display=swap');
 
*, *::before, *::after {{ box-sizing: border-box; }}
 
html, body, [class*="css"] {{
    font-family: 'DM Sans', sans-serif;
    color: #2a1a0e;
}}
 
/* Full-viewport wallpaper, zero blur */
[data-testid="stAppViewContainer"] {{
    background-image: url('{BG_URL}');
    background-size: cover;
    background-position: center top;
    background-attachment: fixed;
    min-height: 100vh;
}}
[data-testid="stAppViewContainer"]::before {{ display: none !important; }}
[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] {{ display: none !important; }}
#MainMenu, footer {{ visibility: hidden; }}
 
/* Force the block-container to be a flex centering wrapper */
.block-container {{
    max-width: 100% !important;
    width: 100% !important;
    padding: 3rem 1rem 4rem !important;
    margin: 0 !important;
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
}}
 
/* Every direct Streamlit child goes inside the sheet indirectly —
   we wrap everything in .sheet via st.markdown open/close */
.sheet {{
    width: min(560px, 94vw);
    background: rgba(255, 245, 230, 0.62);
    border-radius: 22px;
    padding: clamp(1.4rem, 5vw, 2.2rem) clamp(1.2rem, 5vw, 2rem);
    box-shadow: 0 8px 48px rgba(120, 70, 20, 0.18);
    border: 1px solid rgba(200, 160, 110, 0.25);
}}
 
/* ── HERO ── */
.hero {{
    text-align: center;
    padding-bottom: 1.6rem;
}}
.hero-tag {{
    display: inline-block;
    font-size: 0.62rem;
    font-weight: 500;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #8a5c30;
    background: rgba(180,130,70,0.1);
    border: 1px solid rgba(180,130,70,0.25);
    padding: 0.28rem 0.9rem;
    border-radius: 999px;
    margin-bottom: 1.1rem;
}}
.hero-title {{
    font-family: 'Cormorant Garamond', Georgia, serif;
    font-size: clamp(2rem, 5.5vw, 3rem);
    font-weight: 600;
    line-height: 1.12;
    color: #2a1505;
    letter-spacing: -0.01em;
    margin-bottom: 0.8rem;
}}
.hero-title em {{
    font-style: italic;
    color: #b85a30;
}}
.hero-sub {{
    font-size: 0.88rem;
    color: #6a4828;
    font-weight: 300;
    line-height: 1.75;
    max-width: 360px;
    margin: 0 auto;
}}
.hero-rule {{
    width: 36px;
    height: 1.5px;
    background: linear-gradient(90deg, #c8a060, #e8c090);
    margin: 1.3rem auto 0;
    border-radius: 2px;
}}
 
/* ── SECTION TITLE ── */
.section-title {{
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.5rem;
    font-weight: 600;
    color: #2a1505;
    text-align: center;
    letter-spacing: 0.02em;
    margin-bottom: 1.3rem;
}}
 
/* ── FIELD LABELS ── */
.field-label {{
    font-family: 'Cormorant Garamond', serif;
    font-size: 1rem;
    font-weight: 500;
    color: #2a1505;
    margin-bottom: 1px;
    display: block;
}}
.field-hint {{
    font-size: 0.68rem;
    color: #9a7050;
    font-weight: 300;
    margin-bottom: 0.35rem;
    display: block;
}}
.field-sep {{
    height: 1px;
    background: rgba(180,130,70,0.15);
    margin: 0.8rem 0;
}}
.inner-sep {{
    height: 1px;
    background: rgba(180,130,70,0.15);
    margin: 1.3rem 0;
}}
 
/* ── NUMBER INPUT ── */
[data-testid="stNumberInput"] {{ width: 100% !important; }}
[data-testid="stNumberInput"] > div {{ width: 100% !important; }}
[data-testid="stNumberInput"] input {{
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 1.05rem !important;
    font-weight: 500 !important;
    color: #2a1505 !important;
    background: rgba(255, 240, 210, 0.45) !important;
    border: 1px solid rgba(180,130,70,0.3) !important;
    border-radius: 10px !important;
    padding: 0.46rem 0.7rem !important;
    text-align: center !important;
    width: 100% !important;
    transition: border-color 0.2s, box-shadow 0.2s, background 0.2s !important;
}}
[data-testid="stNumberInput"] input:hover {{
    border-color: rgba(180,110,50,0.55) !important;
    background: rgba(255,235,200,0.6) !important;
    box-shadow: 0 0 0 3px rgba(180,120,60,0.1) !important;
}}
[data-testid="stNumberInput"] input:focus {{
    border-color: rgba(180,100,40,0.75) !important;
    background: rgba(255,235,200,0.65) !important;
    box-shadow: 0 0 0 3px rgba(180,110,50,0.14) !important;
    outline: none !important;
}}
[data-testid="stNumberInput"] button {{
    background: rgba(180,130,70,0.12) !important;
    border: 1px solid rgba(180,130,70,0.22) !important;
    color: #8a5c30 !important;
    border-radius: 8px !important;
    font-size: 1rem !important;
    transition: background 0.16s, transform 0.13s !important;
}}
[data-testid="stNumberInput"] button:hover {{
    background: rgba(180,130,70,0.25) !important;
    transform: scale(1.1) !important;
}}
[data-testid="stNumberInput"] label {{ display: none !important; }}
 
/* ── PREDICT BUTTON ── */
.stButton > button {{
    width: 100% !important;
    background: linear-gradient(135deg, #b86838, #d08a58) !important;
    color: #fff5e8 !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
    padding: 0.76rem 1rem !important;
    box-shadow: 0 4px 18px rgba(140,70,20,0.28) !important;
    transition: transform 0.18s, box-shadow 0.2s !important;
    margin-top: 0.3rem !important;
}}
.stButton > button:hover {{
    transform: translateY(-2px) !important;
    box-shadow: 0 7px 24px rgba(140,70,20,0.38) !important;
    filter: brightness(1.05) !important;
}}
.stButton > button:active {{ transform: translateY(0) !important; }}
 
/* ── RESULT BOXES ── */
.results-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.8rem;
    margin-top: 0.5rem;
}}
.res-box {{
    background: rgba(255, 240, 210, 0.5);
    border: 1px solid rgba(180,130,70,0.22);
    border-radius: 14px;
    padding: 1rem 0.7rem 0.85rem;
    text-align: center;
    transition: transform 0.2s, box-shadow 0.2s, background 0.2s;
    cursor: default;
}}
.res-box:hover {{
    transform: translateY(-3px);
    background: rgba(255,235,200,0.65);
    box-shadow: 0 6px 22px rgba(140,90,30,0.15);
}}
.res-lbl {{
    font-family: 'DM Sans', sans-serif;
    font-size: 0.57rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #9a7050;
    font-weight: 500;
    margin-bottom: 0.38rem;
}}
.res-val {{
    font-family: 'Cormorant Garamond', serif;
    font-size: 2.3rem;
    font-weight: 600;
    line-height: 1;
    margin-bottom: 0.38rem;
    color: #2a1505;
}}
.res-val.high   {{ color: #b83030; }}
.res-val.medium {{ color: #9a6010; }}
.res-val.low    {{ color: #2a7848; }}
 
.pill {{
    display: inline-block;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.63rem;
    font-weight: 500;
    letter-spacing: 0.05em;
    padding: 0.18rem 0.72rem;
    border-radius: 999px;
}}
.pill-rose  {{ background: rgba(200,80,80,0.1);  color: #882020; border: 1px solid rgba(200,80,80,0.25); }}
.pill-amber {{ background: rgba(180,130,20,0.1); color: #7a4e00; border: 1px solid rgba(180,130,20,0.25); }}
.pill-sage  {{ background: rgba(40,150,90,0.1);  color: #1a5e38; border: 1px solid rgba(40,150,90,0.25); }}
 
/* ── FOOTER ── */
.footer {{
    text-align: center;
    padding-top: 1.5rem;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.6rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #9a7050;
    line-height: 1.9;
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
    path = os.path.join(base, "cleaned_students_data.csv")
    try:
        df = pd.read_csv(path)
        if df.empty or len(df.columns) == 0:
            raise ValueError("empty")
        return df.drop(columns=["Post_Semester_GPA","Burnout_Risk_Level"], errors="ignore").mean().to_dict()
    except Exception:
        # Safe fallback defaults if CSV is broken/missing
        return {
            "Pre_Semester_GPA": 2.5,
            "Weekly_GenAI_Hours": 5.0,
            "Traditional_Study_Hours": 15.0,
            "Anxiety_Level_During_Exams": 5.0,
            "Skill_Retention_Score": 60.0,
        }
 
try:
    gpa_model, burnout_model, scaler = load_models()
    defaults = load_defaults()
    models_ok = True
except Exception as e:
    models_ok = False
    load_err = str(e)
 
# ── Render: open sheet ────────────────────────────────────────────────────────
st.markdown('<div class="sheet">', unsafe_allow_html=True)
 
st.markdown("""
<div class="hero">
  <div class="hero-tag">BSAI &nbsp;·&nbsp; AI130 &nbsp;·&nbsp; Semester 2</div>
  <h1 class="hero-title">Know Your <em>Academic</em><br>Future, Now.</h1>
  <p class="hero-sub">Enter your study habits and let our machine learning models predict your GPA and burnout risk — trained on 50,000 student records.</p>
  <div class="hero-rule"></div>
</div>
""", unsafe_allow_html=True)
 
if not models_ok:
    st.error(f"Could not load models: {load_err}\n\nMake sure gpa_model.pkl, burnout_model.pkl, scaler.pkl are in your repo.")
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()
 
st.markdown('<div class="section-title">Student Profile</div>', unsafe_allow_html=True)
 
# ── Field helper ──────────────────────────────────────────────────────────────
def field(label, hint, key, mn, mx, step, default):
    if key not in st.session_state:
        st.session_state[key] = float(default)
    st.markdown(f'<span class="field-label">{label}</span><span class="field-hint">{hint}</span>', unsafe_allow_html=True)
    val = st.number_input(
        label, label_visibility="collapsed",
        min_value=float(mn), max_value=float(mx),
        value=float(st.session_state[key]),
        step=float(step), key=f"{key}_num",
        format="%.1f" if step < 1 else "%.0f",
    )
    st.session_state[key] = val
    return val
 
pre_gpa   = field("Pre-Semester GPA",       "0.0 – 4.0",                     "pre_gpa",   0.0,  4.0, 0.1, round(defaults.get("Pre_Semester_GPA", 2.5), 1))
st.markdown('<div class="field-sep"></div>', unsafe_allow_html=True)
weekly_ai = field("Weekly GenAI Hours",      "Hours per week using AI tools",  "weekly_ai", 0.0, 40.0, 0.5, round(defaults.get("Weekly_GenAI_Hours", 5.0), 1))
st.markdown('<div class="field-sep"></div>', unsafe_allow_html=True)
trad_hrs  = field("Traditional Study Hours", "Hours per week without AI",      "trad_hrs",  0.0, 60.0, 0.5, round(defaults.get("Traditional_Study_Hours", 15.0), 1))
st.markdown('<div class="field-sep"></div>', unsafe_allow_html=True)
anxiety   = field("Exam Anxiety Level",      "1 = very calm  —  10 = extreme", "anxiety",   1.0, 10.0, 1.0, float(int(round(defaults.get("Anxiety_Level_During_Exams", 5)))))
st.markdown('<div class="field-sep"></div>', unsafe_allow_html=True)
skill_ret = field("Skill Retention Score",   "0 = poor  —  100 = excellent",   "skill_ret", 0.0,100.0, 1.0, round(defaults.get("Skill_Retention_Score", 60.0), 1))
 
st.markdown('<div class="inner-sep"></div>', unsafe_allow_html=True)
 
# ── Predict ───────────────────────────────────────────────────────────────────
BURNOUT_MAP   = {0: "High",   1: "Low",    2: "Medium"}
BURNOUT_CLASS = {"High": "high", "Medium": "medium", "Low": "low"}
BURNOUT_PILL  = {"High": "pill-rose", "Medium": "pill-amber", "Low": "pill-sage"}
BURNOUT_TEXT  = {"High": "High Risk", "Medium": "Moderate",   "Low": "Low Risk"}
 
if st.button("Predict GPA & Burnout Risk"):
    inp = dict(defaults)
    inp.pop("Post_Semester_GPA", None)
    inp.pop("Burnout_Risk_Level", None)
    inp.update({
        "Pre_Semester_GPA":          pre_gpa,
        "Weekly_GenAI_Hours":         weekly_ai,
        "Traditional_Study_Hours":    trad_hrs,
        "Anxiety_Level_During_Exams": float(anxiety),
        "Skill_Retention_Score":      skill_ret,
    })
    df_in        = pd.DataFrame([inp])
    pred_gpa     = gpa_model.predict(df_in)[0]
    burnout_code = int(burnout_model.predict(df_in)[0])
    b_label      = BURNOUT_MAP[burnout_code]
 
    g_pill, g_txt = (
        ("pill-sage",  "Excellent")       if pred_gpa >= 3.5 else
        ("pill-amber", "On Track")        if pred_gpa >= 2.5 else
        ("pill-rose",  "Needs Attention")
    )
 
    st.markdown(f"""
    <div style="margin-top:0.2rem;">
      <div class="section-title" style="margin-bottom:0.9rem;">Results</div>
      <div class="results-grid">
        <div class="res-box">
          <div class="res-lbl">Predicted GPA</div>
          <div class="res-val">{pred_gpa:.2f}</div>
          <span class="pill {g_pill}">{g_txt}</span>
        </div>
        <div class="res-box">
          <div class="res-lbl">Burnout Risk</div>
          <div class="res-val {BURNOUT_CLASS[b_label]}">{b_label}</div>
          <span class="pill {BURNOUT_PILL[b_label]}">{BURNOUT_TEXT[b_label]}</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)
 
st.markdown("""
<div class="footer">
  StudyPulse &nbsp;·&nbsp; BSAI Semester 2 &nbsp;·&nbsp; AI130 Programming for AI<br>
  Zainab Qasim &nbsp;·&nbsp; Eeman Arif &nbsp;·&nbsp; Khizran Fatima
</div>
""", unsafe_allow_html=True)
 
st.markdown('</div>', unsafe_allow_html=True)
