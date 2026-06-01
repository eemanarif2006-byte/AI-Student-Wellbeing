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

*, *::before, *::after {{ box-sizing: border-box; }}

html, body, [class*="css"] {{
    font-family: 'DM Sans', sans-serif;
}}

/* Wallpaper on the full page — no blur, no overlay */
[data-testid="stAppViewContainer"] {{
    background-image: url('{BG_URL}');
    background-size: cover;
    background-position: center top;
    background-attachment: fixed;
}}

/* The centered column IS the sheet — style it directly */
[data-testid="stAppViewContainer"] > section > div > div > div > div {{
    background: rgba(255, 246, 232, 0.68) !important;
    border-radius: 22px !important;
    border: 1px solid rgba(190, 145, 90, 0.22) !important;
    box-shadow: 0 8px 48px rgba(100, 55, 10, 0.16) !important;
    padding: 2rem 2rem 2.5rem !important;
}}

/* block-container controls the outer width and centering */
.block-container {{
    max-width: 560px !important;
    padding-top: 3rem !important;
    padding-bottom: 4rem !important;
    padding-left: 1rem !important;
    padding-right: 1rem !important;
}}

[data-testid="stHeader"] {{ background: transparent !important; }}
[data-testid="stToolbar"] {{ display: none !important; }}
[data-testid="stDecoration"] {{ display: none !important; }}
#MainMenu, footer {{ visibility: hidden; }}

/* ── HERO ── */
.hero {{
    background: rgba(255,255,255,0.12);
    border-radius: 18px;
    padding: 1.3rem;
}}

.hero-tag {{
    display: inline-block;
    font-size: 0.62rem;
    font-weight: 500;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #7a4e24;
    background: rgba(170, 120, 60, 0.12);
    border: 1px solid rgba(170, 120, 60, 0.28);
    padding: 0.28rem 0.95rem;
    border-radius: 999px;
    margin-bottom: 1.1rem;
}}
.hero-title {{
    font-family: 'Cormorant Garamond', Georgia, serif;
    font-size: clamp(1.9rem, 5vw, 2.9rem);
    font-weight: 600;
    line-height: 1.13;
    color: #201005;
    letter-spacing: -0.01em;
    margin-bottom: 0.8rem;
}}
.hero-title em {{
    font-style: italic;
    color: #aa4e28;
}}
.hero-sub {{
    font-size: 0.88rem;
    color: #6a4020;
    font-weight: 300;
    line-height: 1.75;
    max-width: 360px;
    margin: 0 auto;
}}
.hero-rule {{
    width: 36px;
    height: 1.5px;
    background: linear-gradient(90deg, #c89858, #e8be88);
    margin: 1.3rem auto 0;
    border-radius: 2px;
}}

.soft-panel {{
    background: rgba(122, 78, 36, 0.08);
    border: 1px solid rgba(122, 78, 36, 0.15);
    border-radius: 18px;
    padding: 1.2rem;
    margin-bottom: 1rem;
}}

/* ── SECTION TITLE ── */
.section-title {{
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.5rem;
    font-weight: 600;
    color: #201005;
    text-align: center;
    letter-spacing: 0.02em;
    margin-bottom: 1.3rem;
}}

/* ── FIELD LABELS ── */
.field-label {{
    font-family: 'Cormorant Garamond', serif;
    font-size: 1rem;
    font-weight: 500;
    color: #201005;
    margin-bottom: 1px;
    display: block;
}}
.field-hint {{
    font-size: 0.68rem;
    color: #907050;
    font-weight: 300;
    margin-bottom: 0.35rem;
    display: block;
}}
.field-sep {{
    height: 1px;
    background: rgba(170, 120, 60, 0.15);
    margin: 0.8rem 0;
}}
.inner-sep {{
    height: 1px;
    background: rgba(170, 120, 60, 0.15);
    margin: 1.3rem 0;
}}

/* ── NUMBER INPUT ── */
[data-testid="stNumberInput"] {{ width: 100% !important; }}
[data-testid="stNumberInput"] > div {{ width: 100% !important; }}
[data-testid="stNumberInput"] input {{
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 1.05rem !important;
    font-weight: 500 !important;
    color: #201005 !important;
    background: rgba(255, 238, 210, 0.55) !important;
    border: 1px solid rgba(170, 120, 60, 0.32) !important;
    border-radius: 10px !important;
    padding: 0.46rem 0.7rem !important;
    text-align: center !important;
    width: 100% !important;
    transition: border-color 0.2s,
                box-shadow 0.2s,
                background 0.2s,
                transform 0.15s !important;
}}

[data-testid="stNumberInput"] input:hover {{
    transform: translateY(-1px);
}}
[data-testid="stNumberInput"] input:hover {{
    border-color: rgba(160, 100, 40, 0.6) !important;
    background: rgba(255, 232, 198, 0.7) !important;
    box-shadow: 0 0 0 3px rgba(170, 110, 50, 0.1) !important;
}}
[data-testid="stNumberInput"] input:focus {{
    border-color: rgba(160, 90, 30, 0.8) !important;
    background: rgba(255, 230, 195, 0.75) !important;
    box-shadow: 0 0 0 3px rgba(170, 100, 40, 0.14) !important;
    outline: none !important;
}}
[data-testid="stNumberInput"] button {{
    background: rgba(170, 120, 60, 0.13) !important;
    border: 1px solid rgba(170, 120, 60, 0.24) !important;
    color: #7a4e24 !important;
    border-radius: 8px !important;
    font-size: 1rem !important;
    transition: background 0.16s, transform 0.13s !important;
}}
[data-testid="stNumberInput"] button:hover {{
    background: rgba(170, 120, 60, 0.26) !important;
    transform: scale(1.1) !important;
}}
[data-testid="stNumberInput"] label {{ display: none !important; }}

/* ── PREDICT BUTTON ── */
.stButton > button {{
    width: 100% !important;
    background: linear-gradient(135deg, #ae6030, #cc8050) !important;
    color: #fff5e5 !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
    padding: 0.76rem 1rem !important;
    box-shadow: 0 4px 18px rgba(130, 60, 10, 0.28) !important;
    transition: transform 0.18s, box-shadow 0.2s, filter 0.18s !important;
    margin-top: 0.3rem !important;
}}
.stButton > button:hover {{
    transform: translateY(-2px) !important;
    box-shadow: 0 7px 24px rgba(130, 60, 10, 0.38) !important;
    filter: brightness(1.06) !important;
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
    background: rgba(255, 238, 210, 0.6);
    border: 1px solid rgba(170, 120, 60, 0.22);
    border-radius: 14px;
    padding: 1rem 0.7rem 0.85rem;
    text-align: center;
    transition: transform 0.2s, box-shadow 0.2s, background 0.2s;
    cursor: default;
}}
.res-box:hover {{
    transform: translateY(-3px);
    background: rgba(255, 232, 198, 0.75);
    box-shadow: 0 6px 22px rgba(130, 80, 20, 0.15);
}}
.res-lbl {{
    font-family: 'DM Sans', sans-serif;
    font-size: 0.57rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #907050;
    font-weight: 500;
    margin-bottom: 0.38rem;
}}
.res-val {{
    font-family: 'Cormorant Garamond', serif;
    font-size: 2.3rem;
    font-weight: 600;
    line-height: 1;
    margin-bottom: 0.38rem;
    color: #201005;
}}
.res-val.high   {{ color: #a02828; }}
.res-val.medium {{ color: #885808; }}
.res-val.low    {{ color: #246040; }}

.pill {{
    display: inline-block;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.63rem;
    font-weight: 500;
    letter-spacing: 0.05em;
    padding: 0.18rem 0.72rem;
    border-radius: 999px;
}}
.pill-rose  {{ background: rgba(180,60,60,0.1);  color: #7a1818; border: 1px solid rgba(180,60,60,0.25); }}
.pill-amber {{ background: rgba(160,110,10,0.1); color: #6a4000; border: 1px solid rgba(160,110,10,0.25); }}
.pill-sage  {{ background: rgba(30,130,70,0.1);  color: #144a2c; border: 1px solid rgba(30,130,70,0.25); }}

@media (max-width: 640px) {{
    .results-grid {{
        grid-template-columns: 1fr;
    }}

    .hero-title {{
        font-size: 2rem !important;
    }}

    .res-val {{
        font-size: 2rem !important;
    }}
}}
/* ── FOOTER ── */
.footer {{
    text-align: center;
    padding-top: 1.5rem;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.6rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #907050;
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
        return {
            "Pre_Semester_GPA": 2.5,
            "Weekly_GenAI_Hours": 5.0,
            "Traditional_Study_Hours": 15.0,
            "Anxiety_Level_During_Exams": 5.0,
            "Skill_Retention_Score": 60.0,
        }

try:
    gpa_model, burnout_model = load_models()
    defaults = load_defaults()
    models_ok = True
except Exception as e:
    models_ok = False
    load_err = str(e)

# ── HERO ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-tag">BSAI &nbsp;·&nbsp; AI130 &nbsp;·&nbsp; Semester 2</div>
  <h1 class="hero-title">Know Your <em>Academic</em><br>Future, Now.</h1>
  <p class="hero-sub">Enter your study habits and let our machine learning models predict your GPA and burnout risk — trained on 50,000 student records.</p>
  <div class="hero-rule"></div>
</div>
""", unsafe_allow_html=True)

if not models_ok:
    st.error(f"Could not load models: {load_err}\n\nMake sure gpa_model.pkl and burnout_model.pkl are in your repo root.")
    st.stop()

st.markdown("""
<div class="soft-panel">
<div class="section-title">Student Profile</div>
</div>
""", unsafe_allow_html=True)

# ── Field helper ──────────────────────────────────────────────────────────────
def field(label, hint, key, mn, mx, step, default):
    if key not in st.session_state:
        st.session_state[key] = float(default)
    st.markdown(
        f'<span class="field-label">{label}</span>'
        f'<span class="field-hint">{hint}</span>',
        unsafe_allow_html=True,
    )
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
        "Pre_Semester_GPA":           pre_gpa,
        "Weekly_GenAI_Hours":          weekly_ai,
        "Traditional_Study_Hours":     trad_hrs,
        "Anxiety_Level_During_Exams":  float(anxiety),
        "Skill_Retention_Score":       skill_ret,
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
