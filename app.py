import gc
import html
import hashlib
import json
import os
import tempfile
import time
import uuid

import streamlit as st

from api_utils import build_interview, create_session_summary, evaluate_answer
from document_utils import read_cv


st.set_page_config(page_title="Candidate360", page_icon="✦", layout="wide")


ROLE_OPTIONS = [
    "Accountant",
    "Accounting",
    "Actuary",
    "Administration",
    "Administrative Assistant",
    "Advertising",
    "Aerospace Engineer",
    "Aerospace Engineering",
    "Agricultural Engineer",
    "Agriculture",
    "Airline Cabin Crew",
    "Animation",
    "Anthropology",
    "App Developer",
    "Archaeology",
    "Architect",
    "Architecture",
    "Art and Design",
    "Artificial Intelligence",
    "Artificial Intelligence Specialist",
    "Auditor",
    "Automotive Engineer",
    "Banking",
    "Biochemistry",
    "Biology",
    "Biomedical Engineer",
    "Biomedical Engineering",
    "Biotechnology",
    "Brand Manager",
    "Business",
    "Business Administration",
    "Business Analyst",
    "Business Development Specialist",
    "Business Management",
    "Chemical Engineer",
    "Chemical Engineering",
    "Chemistry",
    "Civil Engineer",
    "Civil Engineering",
    "Clinical Psychologist",
    "Communications",
    "Communications Specialist",
    "Computer Engineer",
    "Computer Engineering",
    "Computer Programmer",
    "Computer Science",
    "Construction Manager",
    "Content Creator",
    "Content Writer",
    "Corporate Communications",
    "Criminology",
    "Customer Service",
    "Customer Service Representative",
    "Data Analyst",
    "Data Engineer",
    "Data Science",
    "Data Scientist",
    "Database Administrator",
    "Dentist",
    "Dentistry",
    "Design",
    "Dietitian",
    "Digital Marketing Specialist",
    "Doctor",
    "E-commerce Specialist",
    "Economics",
    "Education",
    "Electrical Engineer",
    "Electrical Engineering",
    "Electronics Engineer",
    "Emergency Medical Technician",
    "Engineering",
    "English Language and Literature",
    "Environmental Engineer",
    "Environmental Science",
    "Event Coordinator",
    "Fashion Design",
    "Fashion Designer",
    "Film Production",
    "Finance",
    "Finance Officer",
    "Financial Analyst",
    "Flight Attendant",
    "Food Science",
    "Forensic Science",
    "General Manager",
    "Geology",
    "Graphic Design",
    "Graphic Designer",
    "Healthcare",
    "Healthcare Administration",
    "Health and Safety Officer",
    "Hospitality",
    "Hospitality Management",
    "Hotel Manager",
    "Hotel Receptionist",
    "HR Operations Executive",
    "Human Resources",
    "Human Resources Specialist",
    "Industrial Engineer",
    "Industrial Engineering",
    "Information Systems",
    "Information Technology",
    "Insurance Specialist",
    "Interior Design",
    "Interior Designer",
    "International Business",
    "International Relations",
    "IT",
    "IT Support Specialist",
    "Journalism",
    "Journalist",
    "Laboratory Technician",
    "Law",
    "Lawyer",
    "Legal Assistant",
    "Librarian",
    "Logistics",
    "Logistics Coordinator",
    "Management",
    "Marine Engineer",
    "Marine Engineering",
    "Marketing",
    "Marketing Executive",
    "Mathematics",
    "Mechanical Engineer",
    "Mechanical Engineering",
    "Media Studies",
    "Medicine",
    "Mobile App Developer",
    "Network Engineer",
    "Nurse",
    "Nursing",
    "Nutrition",
    "Nutritionist",
    "Office Administrator",
    "Operations Coordinator",
    "Operations Manager",
    "Paralegal",
    "Petroleum Engineer",
    "Petroleum Engineering",
    "Pharmacist",
    "Pharmacy",
    "Photography",
    "Photographer",
    "Physics",
    "Physiotherapist",
    "Physiotherapy",
    "Political Science",
    "Procurement Specialist",
    "Product Designer",
    "Product Manager",
    "Production Engineer",
    "Project Coordinator",
    "Project Manager",
    "Psychology",
    "Public Administration",
    "Public Health",
    "Public Relations",
    "Public Relations Specialist",
    "Quality Assurance Specialist",
    "Radiographer",
    "Radiography",
    "Real Estate Agent",
    "Receptionist",
    "Research Assistant",
    "Risk Analyst",
    "Safety Engineer",
    "Sales",
    "Sales Executive",
    "Science",
    "Secretary",
    "Social Media Specialist",
    "Social Work",
    "Social Worker",
    "Sociology",
    "Software Developer",
    "Software Engineer",
    "Statistics",
    "Supply Chain Management",
    "Supply Chain Specialist",
    "Systems Analyst",
    "Teacher",
    "Teaching",
    "Telecommunications Engineer",
    "Tourism",
    "Translation",
    "Translator",
    "UI/UX Designer",
    "Veterinary Doctor",
    "Veterinary Medicine",
    "Web Developer"
]


def report_list(title, values):
    items = "".join(f"<li>{html.escape(str(value))}</li>" for value in values)
    if not items:
        items = "<li>Not available</li>"
    return f"<h3>{html.escape(title)}</h3><ul>{items}</ul>"


def create_html_report(report):
    target_role = html.escape(str(report.get("target_role", "Not specified")))
    cv_analysis = report.get("cv_analysis", {})
    answers = report.get("answers", [])
    summary = report.get("summary", {})
    visual = report.get("visual_observations", {})

    answer_sections = ""
    for number, item in enumerate(answers, 1):
        evaluation = item.get("evaluation", {})
        voice = item.get("voice_evaluation", {})
        voice_text = "Not completed"
        if voice:
            voice_text = (
                f"Duration: {voice.get('duration_seconds', 0)} seconds; "
                f"pace: {voice.get('words_per_minute', 0)} words per minute "
                f"({voice.get('pace_label', 'Not available')}); "
                f"listed filler words: {voice.get('filler_count', 0)} "
                f"({voice.get('filler_label', 'Not available')})."
            )
        answer_sections += f"""
        <section class="card">
            <div class="number">ANSWER {number}</div>
            <h2>{html.escape(str(item.get("question", "Interview question")))}</h2>
            <p><strong>Candidate answer</strong></p>
            <p>{html.escape(str(item.get("answer", "")))}</p>
            <div class="score">Overall practice score: {html.escape(str(evaluation.get("overall", 0)))}%</div>
            {report_list("What worked", evaluation.get("strengths", []))}
            {report_list("Next improvements", evaluation.get("improvements", []))}
            <h3>Recorded-voice delivery</h3>
            <p>{html.escape(voice_text)}</p>
        </section>
        """

    visual_text = "Not completed"
    if visual:
        visual_parts = []
        if "person_visibility" in visual:
            visual_parts.append(
                f"Person visible: {visual.get('person_visibility', 0)}%"
            )
        elif "people" in visual:
            visual_parts.append(
                f"People detected: {visual.get('people', 0)}"
            )
        if "face_visibility" in visual:
            visual_parts.append(
                f"Face visible: {visual.get('face_visibility', 0)}%"
            )

        expressions = visual.get("expressions", [])
        if expressions:
            expression_text = ", ".join(
                f"{item.get('label', 'Unknown').title()} "
                f"{round(item.get('score', 0) * 100, 1)}%"
                for item in expressions
            )
            visual_parts.append("Expression outputs: " + expression_text)

        confidence_outputs = visual.get("confidence_outputs", [])
        if confidence_outputs:
            confidence_text = ", ".join(
                f"{item.get('label', 'Unknown')} "
                f"{round(item.get('score', 0) * 100, 1)}%"
                for item in confidence_outputs
            )
            visual_parts.append(
                "Confidence-dataset labels: " + confidence_text
            )

        gestures = visual.get("gestures", [])
        if gestures:
            gesture_text = ", ".join(
                f"{item.get('label', 'Unknown').title()} "
                f"{round(item.get('score', 0) * 100, 1)}%"
                for item in gestures
            )
            visual_parts.append("Hand gestures: " + gesture_text)

        visual_text = "; ".join(visual_parts) or "Not available"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Candidate360 Session Report</title>
<style>
body{{font-family:Arial,sans-serif;background:#eef4f8;color:#17263a;margin:0;padding:35px;line-height:1.6}}
.report{{max-width:950px;margin:auto}}
.hero{{background:linear-gradient(120deg,#102441,#17606b);color:white;padding:36px;border-radius:22px}}
.hero h1{{margin:4px 0;font-size:34px}}.hero p{{color:#d7eaf1}}
.card{{background:white;border:1px solid #d8e3ec;border-radius:18px;padding:28px;margin-top:20px;box-shadow:0 8px 24px #17324d14}}
.number{{color:#087f75;font-weight:bold;letter-spacing:.12em;font-size:12px}}
.score{{display:inline-block;background:#dff7f1;color:#075e55;font-weight:bold;padding:8px 14px;border-radius:20px}}
h2{{color:#173a5e}}h3{{margin-bottom:5px;color:#225174}}li{{margin-bottom:6px}}
.notice{{background:#fff7df;border-left:5px solid #e7a928;padding:16px;margin-top:22px;border-radius:8px}}
</style>
</head>
<body><main class="report">
<header class="hero"><div>CANDIDATE360</div><h1>Interview Practice Report</h1><p>Target role: {target_role}</p></header>
<section class="card"><div class="number">OPPORTUNITY MAP</div><h2>{html.escape(str(cv_analysis.get("match_summary", "CV and job analysis")))}</h2>
<div class="score">Practice match: {html.escape(str(cv_analysis.get("match_score", 0)))}%</div>
{report_list("Skills and experience found in the CV", cv_analysis.get("cv_strengths", []))}
{report_list("Skills to prove", cv_analysis.get("skills_to_prove", []))}</section>
<section class="card"><div class="number">FINAL COACH SUMMARY</div><h2>{html.escape(str(summary.get("headline", "Interview debrief")))}</h2>
<p>{html.escape(str(summary.get("overall_progress", "")))}</p>
{report_list("Top strengths", summary.get("top_strengths", []))}
{report_list("Priority actions", summary.get("priority_actions", []))}
{report_list("Next practice plan", summary.get("next_practice_plan", []))}</section>
{answer_sections}
<section class="card"><div class="number">VISUAL DELIVERY EVALUATION</div><p>{html.escape(visual_text)}</p></section>
<div class="notice"><strong>Practice-use notice:</strong> This report supports interview preparation only. Answer, recorded-voice and visual feedback remain separate. Visual outputs do not prove emotion, confidence, nervousness, honesty or personality.</div>
</main></body></html>"""


RECORDING_MAX_AGE_SECONDS = 2 * 60 * 60


@st.cache_resource
def cleanup_stale_recordings():
    """Remove abandoned temporary Candidate360 recordings after two hours."""
    current_time = time.time()
    try:
        file_names = os.listdir(tempfile.gettempdir())
    except OSError:
        return True

    for file_name in file_names:
        if not file_name.startswith("candidate360_"):
            continue
        file_path = os.path.join(tempfile.gettempdir(), file_name)
        try:
            if current_time - os.path.getmtime(file_path) > RECORDING_MAX_AGE_SECONDS:
                os.remove(file_path)
        except OSError:
            continue
    return True


cleanup_stale_recordings()


@st.cache_resource
def get_speech_model():
    from speech_utils import load_speech_model
    return load_speech_model()


@st.cache_resource
def get_pose_model():
    from vision_utils import load_pose_model
    return load_pose_model()


@st.cache_resource
def get_live_person_model():
    from vision_utils import load_live_person_model
    return load_live_person_model()


@st.cache_resource
def get_expression_model():
    from vision_utils import load_expression_model
    return load_expression_model()


@st.cache_resource
def get_confidence_model():
    from vision_utils import load_confidence_model
    return load_confidence_model()


@st.cache_resource
def get_gesture_recognizer():
    from vision_utils import load_gesture_recognizer
    return load_gesture_recognizer()


def get_vision_models():
    return (
        get_pose_model(),
        get_expression_model(),
        get_confidence_model()
    )


def release_visual_models():
    """Release cached visual models before loading the speech model."""
    for cached_loader in (
        get_pose_model,
        get_live_person_model,
        get_expression_model,
        get_confidence_model,
        get_gesture_recognizer
    ):
        clear_method = getattr(cached_loader, "clear", None)
        if callable(clear_method):
            clear_method()

    try:
        from tensorflow.keras import backend as keras_backend
        keras_backend.clear_session()
    except Exception:
        pass

    gc.collect()


def get_face_model_status():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(project_dir, "models", "face_expression_model.keras")
    class_path = os.path.join(project_dir, "models", "class_names.json")

    missing_files = []
    if not os.path.exists(model_path):
        missing_files.append("models/face_expression_model.keras")
    if not os.path.exists(class_path):
        missing_files.append("models/class_names.json")

    if missing_files:
        return {
            "ready": False,
            "message": "Facial-expression output is unavailable. Missing: " + ", ".join(missing_files)
        }

    return {"ready": True, "message": "Facial-expression model files are available."}


def get_confidence_model_status():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(project_dir, "models", "confidence_model.keras")
    class_path = os.path.join(
        project_dir,
        "models",
        "confidence_class_names.json"
    )

    missing_files = []
    if not os.path.exists(model_path):
        missing_files.append("models/confidence_model.keras")
    if not os.path.exists(class_path):
        missing_files.append("models/confidence_class_names.json")

    if missing_files:
        return {
            "ready": False,
            "message": (
                "Confidence-dataset output is unavailable. Missing: "
                + ", ".join(missing_files)
            )
        }
    return {
        "ready": True,
        "message": "Confidence-dataset model files are available."
    }


def get_rtc_configuration():
    """Return browser ICE servers without exposing credentials in the UI."""
    ice_servers = [
        {
            "urls": [
                "stun:stun.l.google.com:19302",
                "stun:stun1.l.google.com:19302",
                "stun:stun2.l.google.com:19302"
            ]
        }
    ]

    try:
        turn_settings = st.secrets.get("turn", {})
        turn_urls = turn_settings.get("urls", [])
        turn_username = str(turn_settings.get("username", "")).strip()
        turn_credential = str(turn_settings.get("credential", "")).strip()

        if isinstance(turn_urls, str):
            turn_urls = [turn_urls]
        else:
            turn_urls = list(turn_urls)

        turn_urls = [str(url).strip() for url in turn_urls if str(url).strip()]
        if turn_urls and turn_username and turn_credential:
            ice_servers.append(
                {
                    "urls": turn_urls,
                    "username": turn_username,
                    "credential": turn_credential
                }
            )
    except Exception:
        # Public STUN remains available when TURN has not been configured.
        pass

    # Also accept the single JSON secret used by claudeapp.py.
    try:
        turn_servers_raw = st.secrets.get("TURN_ICE_SERVERS", "")
        if turn_servers_raw:
            turn_servers = (
                json.loads(turn_servers_raw)
                if isinstance(turn_servers_raw, str)
                else turn_servers_raw
            )
            if isinstance(turn_servers, list):
                ice_servers.extend(turn_servers)
    except (TypeError, ValueError):
        pass

    return {"iceServers": ice_servers}

st.markdown("""
<style>
html,body,[class*="css"]{font-family:'Segoe UI',Arial,sans-serif}
.stApp{background:radial-gradient(circle at 85% 0%,#123758 0,transparent 25%),#07111f;color:#eef6ff}
.block-container{max-width:1320px;padding:1.5rem 2rem 4rem}

/* Readable text throughout the dark interface */
p,li,span,label,[data-testid="stMarkdownContainer"]{color:#dce9f7}
h1,h2,h3,h4{color:#ffffff!important;letter-spacing:-.02em}
h2{font-size:1.9rem!important}h3{font-size:1.25rem!important}
[data-testid="stCaptionContainer"],.stCaption{color:#b9cce0!important;font-size:.95rem!important;line-height:1.55}
[data-testid="stWidgetLabel"] p{color:#f2f7ff!important;font-size:1rem!important;font-weight:700!important}
small{color:#bdd0e3!important}

/* Sidebar */
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0d1b2f 0%,#091424 100%);border-right:1px solid #29425f}
[data-testid="stSidebar"] .block-container{padding:1.6rem 1.15rem}
[data-testid="stSidebar"] h1{font-size:1.55rem!important;margin-bottom:1.25rem}
[data-testid="stSidebar"] p,[data-testid="stSidebar"] label{color:#eaf3fc!important}

/* Hero and feature cards */
.hero{position:relative;overflow:hidden;padding:46px;border-radius:28px;background:radial-gradient(circle at 86% 8%,#35e6c950,transparent 29%),linear-gradient(125deg,#102441,#173b61 60%,#0f5f69);border:1px solid #4cebd16b;box-shadow:0 24px 70px #0008;margin-bottom:26px}
.hero h1{font-size:3.25rem!important;line-height:1.06;margin:10px 0 14px}
.hero p{color:#dceaff!important;font-size:1.18rem;line-height:1.65;max-width:860px;margin:0}
.eyebrow{color:#65f5dc!important;font-weight:800;letter-spacing:.14em;font-size:.82rem}
.glass{box-sizing:border-box;height:188px;background:linear-gradient(145deg,#142941e8,#0d1c2ee8);border:1px solid #345777;border-radius:20px;padding:22px;box-shadow:0 14px 40px #0005;margin-bottom:12px}
.glass h3{margin:.7rem 0 .55rem;font-size:1.22rem!important}
.glass p{color:#c4d4e5!important;font-size:.96rem;line-height:1.5;margin:0}

/* Questions, notices and result areas */
.question{min-height:112px;display:flex;align-items:center;padding:26px;background:linear-gradient(120deg,#123252,#10243b);border:1px solid #3b7893;border-left:6px solid #55edcf;border-radius:18px;color:#ffffff;font-size:1.28rem;font-weight:700;line-height:1.55}
.notice{padding:18px 20px;border-radius:15px;background:#14283b;border:1px solid #3b5b76;color:#e0ecf8;font-size:1rem;line-height:1.6;margin-bottom:16px}
.notice b{color:#6cf2dc}

/* Consistent widgets */
div[data-baseweb="input"]>div,div[data-baseweb="select"]>div{background:#ffffff!important;border:1px solid #90a9c1!important;border-radius:12px!important;min-height:48px}
div[data-baseweb="input"] input,div[data-baseweb="select"] span{color:#14243a!important;font-size:1rem!important}
textarea{background:#ffffff!important;color:#14243a!important;border:1px solid #90a9c1!important;border-radius:12px!important;font-size:1rem!important;line-height:1.55!important}
textarea::placeholder,input::placeholder{color:#667b91!important;opacity:1!important}
[data-testid="stFileUploaderDropzone"]{background:#0f2034;border:1px dashed #4f7697;border-radius:16px;min-height:150px}
[data-testid="stFileUploaderDropzone"] p,[data-testid="stFileUploaderDropzone"] small{color:#dce9f7!important}
.stButton>button,.stDownloadButton>button{background:linear-gradient(90deg,#20bfa7,#2779d8);color:white!important;border:0;border-radius:12px;font-size:1rem;font-weight:800;min-height:48px;box-shadow:0 8px 24px #1f9f9d45}
.stButton>button:hover,.stDownloadButton>button:hover{filter:brightness(1.12);transform:translateY(-1px)}

/* Equal metrics and highly visible stage navigation */
div[data-testid="stMetric"]{box-sizing:border-box;min-height:126px;background:#101f32;border:1px solid #34516e;padding:18px;border-radius:16px}
[data-testid="stMetricLabel"] p{color:#bfd1e3!important;font-size:.95rem!important;font-weight:700!important}
[data-testid="stMetricValue"]{color:#ffffff!important;font-size:1.85rem!important}
[data-testid="stTabs"]{margin-top:24px}
[data-testid="stTabs"] [role="tablist"]{display:flex;gap:10px;background:#0b1a2d;border:1px solid #294866;border-radius:16px;padding:9px;box-shadow:0 12px 34px #0005}
[data-testid="stTabs"] button[role="tab"]{flex:1;min-height:58px;padding:10px 16px;background:#122740;border:1px solid #315777;border-radius:12px;color:#dcecff!important;font-size:1rem;font-weight:800;white-space:normal}
[data-testid="stTabs"] button[role="tab"] p{color:#dcecff!important;font-size:1rem!important;font-weight:800!important;text-align:center}
[data-testid="stTabs"] button[role="tab"]:hover{background:#193452;border-color:#4d789c}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"]{background:linear-gradient(120deg,#14aF9f,#167fd0);border-color:#64ead6;box-shadow:0 8px 22px #0ea5a544;color:#ffffff!important}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] p{color:#ffffff!important}
[data-testid="stTabs"] [data-baseweb="tab-highlight"]{display:none}
.stProgress>div>div{background:linear-gradient(90deg,#25c8ad,#3288ec)}
hr{border-color:#2c4661!important;margin:2rem 0!important}

@media(max-width:800px){
  .block-container{padding:1rem 1rem 3rem}.hero{padding:30px 24px}.hero h1{font-size:2.25rem!important}.hero p{font-size:1.02rem}.glass{height:auto;min-height:160px}
  [data-testid="stTabs"] [role="tablist"]{flex-wrap:wrap}
  [data-testid="stTabs"] button[role="tab"]{flex:1 1 46%;min-width:145px}
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero"><div class="eyebrow">CANDIDATE360 • AI INTERVIEW COACH</div>
<h1>See your complete story.<br>Practise your strongest interview.</h1>
<p>Turn your CV and a real job description into a tailored interview, answer by text or recorded voice,
use live or uploaded visual practice, and leave with separate answer, voice-delivery and visual feedback.</p></div>
""", unsafe_allow_html=True)

for column, item in zip(st.columns(4), [
    ("01", "CV Analyst", "Finds relevant skills, experience and gaps in the CV and role."),
    ("02", "Interviewer", "Creates fair questions tailored to the target position."),
    ("03", "Answer Coach", "Reviews relevance, structure, specificity and alignment."),
    ("04", "Summary Coach", "Turns the session into clear next-practice actions.")
]):
    with column:
        st.markdown(f'<div class="glass"><span class="eyebrow">{item[0]}</span><h3>{item[1]}</h3><p>{item[2]}</p></div>', unsafe_allow_html=True)

for key, value in {
    "cv_text": "", "job_description": "", "questions": [], "answer_results": [],
    "question_index": 0, "transcript": ""
}.items():
    if key not in st.session_state:
        st.session_state[key] = value

try:
    api_key = (
        st.secrets.get("OPENROUTER_API_KEY", "")
        or st.secrets.get("OPENAI_API_KEY", "")
    )
except Exception:
    api_key = ""

with st.sidebar:
    st.title("✦ Studio controls")
    st.markdown("**Set the role before building your interview.**")
    target_role = st.selectbox(
        "Target role or major",
        ROLE_OPTIONS,
        index=None,
        placeholder="Search or type a role or major...",
        accept_new_options=True,
        help="Start typing to filter the suggestions. You can also enter a role, major or study field that is not listed."
    )
    st.divider()
    if api_key:
        st.success("AI coach is ready")
    else:
        st.warning(
            "Add OPENROUTER_API_KEY or OPENAI_API_KEY to Streamlit Secrets "
            "to enable the AI coach."
        )
    st.caption("Your CV, answers and recordings are used only to provide practice feedback during the session.")

setup_tab, practice_tab, delivery_tab, summary_tab = st.tabs([
    "01 · Role Lab", "02 · Interview Room", "03 · Delivery Lab", "04 · Debrief"
])

with setup_tab:
    st.subheader("Build the interview around a real opportunity")
    left, right = st.columns([1, 1])
    with left:
        cv_file = st.file_uploader("Upload CV", type=["pdf", "docx", "txt"])
        if cv_file is not None:
            try:
                cv_fingerprint = hashlib.sha256(cv_file.getvalue()).hexdigest()
                if st.session_state.get("cv_file_fingerprint") != cv_fingerprint:
                    cv_file.seek(0)
                    st.session_state.cv_text = read_cv(cv_file)
                    st.session_state.cv_file_fingerprint = cv_fingerprint
                st.success(f"Read {len(st.session_state.cv_text):,} characters from the CV.")
                with st.expander("Check extracted CV text"):
                    edited_cv_text = st.text_area(
                        "CV text",
                        value=st.session_state.cv_text,
                        height=240,
                        help="Correct any PDF extraction mistakes here before running the CV Analyst."
                    )
                    st.session_state.cv_text = edited_cv_text
            except Exception as error:
                st.error(str(error))
    with right:
        st.session_state.job_description = st.text_area(
            "Paste job description", st.session_state.job_description, height=255,
            placeholder="Paste the responsibilities, requirements and preferred skills..."
        )

    if st.button("Run CV Analyst + Build Interview", type="primary", use_container_width=True):
        if not target_role or not st.session_state.cv_text or not st.session_state.job_description:
            st.warning("Add the target role, CV and job description first.")
        else:
            try:
                with st.spinner("The CV Analyst and Interviewer are preparing your studio..."):
                    interview_setup = build_interview(
                        st.session_state.cv_text, st.session_state.job_description, target_role, api_key
                    )
                    st.session_state.cv_analysis = interview_setup.get(
                        "cv_analysis",
                        {}
                    )
                    st.session_state.questions = interview_setup.get(
                        "questions",
                        []
                    )
                    st.session_state.answer_results = []
                    st.session_state.question_index = 0
                st.success("Your tailored interview is ready.")
            except Exception as error:
                st.error(str(error))

    if "cv_analysis" in st.session_state:
        result = st.session_state.cv_analysis
        st.markdown("### Opportunity map")
        a, b, c = st.columns(3)
        a.metric("Practice match", f'{result.get("match_score", 0)}%')
        b.metric("CV strengths found", len(result.get("cv_strengths", [])))
        c.metric("Skills to prove", len(result.get("skills_to_prove", [])))
        st.write(result.get("match_summary", ""))
        col1, col2, col3 = st.columns(3)
        for column, title, values in [
            (col1, "Skills and experience found in your CV", result.get("cv_strengths", [])),
            (col2, "Missing or unclear", result.get("missing_or_unclear", [])),
            (col3, "Prove in the interview", result.get("skills_to_prove", []))
        ]:
            with column:
                st.markdown(f"**{title}**")
                for value in values:
                    st.write("• " + value)

with practice_tab:
    questions = st.session_state.questions
    if not questions:
        st.info("Complete Role Lab first to generate your tailored interview.")
    else:
        question_index = st.selectbox(
            "Interview question", range(len(questions)),
            index=min(st.session_state.question_index, len(questions) - 1),
            format_func=lambda number: f'{number + 1}. {questions[number].get("type", "Interview")}'
        )
        st.session_state.question_index = question_index
        selected = questions[question_index]
        question = selected.get("question", "")
        st.markdown(f'<div class="question">{html.escape(question)}</div>', unsafe_allow_html=True)
        st.caption("Why it was selected: " + selected.get("purpose", "Tailored interview practice"))

        answer_method = st.radio(
            "Answer method",
            ["Type my answer", "Record with microphone", "Upload a recording"],
            horizontal=True
        )
        audio_file = None
        if answer_method == "Record with microphone":
            st.caption("Allow microphone access when your browser asks. If recording fails, refresh the page or choose Upload a recording.")
            audio_file = st.audio_input(
                "Record your answer",
                sample_rate=16000,
                key="microphone_answer"
            )
        elif answer_method == "Upload a recording":
            audio_file = st.file_uploader("Upload audio", type=["wav", "mp3", "m4a", "flac"], key="answer_audio")

        if audio_file is not None:
            st.audio(audio_file)
            if st.button("Create multilingual transcript"):
                try:
                    with st.spinner("Loading Whisper and transcribing the recording..."):
                        from speech_utils import transcribe_audio
                        speech_model = get_speech_model()
                        st.session_state.transcript = transcribe_audio(audio_file, speech_model)
                    st.success("Transcript created. Correct any mistakes before evaluation.")
                except Exception as error:
                    st.error(str(error))

        answer = st.text_area(
            "Candidate answer / corrected transcript",
            value=st.session_state.transcript if answer_method != "Type my answer" else "",
            height=220,
            help="Correction is important for accents, names and technical terms."
        )

        if audio_file is not None:
            if st.button(
                "Evaluate recorded-voice delivery",
                use_container_width=True
            ):
                try:
                    from speech_utils import evaluate_voice_delivery
                    st.session_state.last_voice_evaluation = (
                        evaluate_voice_delivery(audio_file, answer)
                    )
                    st.session_state.last_voice_question = question
                    for saved_answer in st.session_state.answer_results:
                        if saved_answer.get("question") == question:
                            saved_answer["voice_evaluation"] = (
                                st.session_state.last_voice_evaluation
                            )
                except Exception as error:
                    st.error(str(error))

        if (
            st.session_state.get("last_voice_question") == question
            and "last_voice_evaluation" in st.session_state
        ):
            voice_result = st.session_state.last_voice_evaluation
            st.markdown("### Recorded-voice delivery evaluation")
            voice_columns = st.columns(3)
            voice_columns[0].metric(
                "Recording duration",
                f'{voice_result.get("duration_seconds", 0)} sec'
            )
            voice_columns[1].metric(
                "Speaking pace",
                f'{voice_result.get("words_per_minute", 0)} words/min'
            )
            voice_columns[2].metric(
                "Listed filler words",
                voice_result.get("filler_count", 0)
            )
            st.write(
                "**Pace observation:** "
                + voice_result.get("pace_label", "Not available")
            )
            st.write(
                "**Filler-word observation:** "
                + voice_result.get("filler_label", "Not available")
            )
            for feedback_item in voice_result.get("feedback", []):
                st.write("→ " + feedback_item)
            st.caption(
                "This delivery evaluation uses recording duration and the "
                "corrected transcript. It does not infer accent quality, "
                "emotion, confidence or nervousness. Filler detection covers "
                "the listed English and Arabic markers only."
            )

        if st.button("Ask the Answer Coach", type="primary", use_container_width=True):
            if not answer.strip():
                st.warning("Type, record or upload an answer first.")
            else:
                try:
                    with st.spinner("The Answer Coach is reviewing your answer and structure..."):
                        result = evaluate_answer(
                            question, answer, st.session_state.cv_text,
                            st.session_state.job_description, api_key
                        )
                    voice_evaluation = {}
                    if st.session_state.get("last_voice_question") == question:
                        voice_evaluation = st.session_state.get(
                            "last_voice_evaluation",
                            {}
                        )
                    record = {
                        "question": question,
                        "answer": answer,
                        "evaluation": result,
                        "voice_evaluation": voice_evaluation
                    }
                    st.session_state.answer_results = [
                        item for item in st.session_state.answer_results if item["question"] != question
                    ] + [record]
                    st.session_state.last_evaluation = result
                except Exception as error:
                    st.error(str(error))

        if "last_evaluation" in st.session_state:
            result = st.session_state.last_evaluation
            st.markdown("### Coach board")
            score_columns = st.columns(5)
            for column, label, key in zip(score_columns,
                ["Relevance", "Specificity", "Structure", "Role fit", "Overall"],
                ["relevance", "specificity", "structure", "job_alignment", "overall"]):
                column.metric(label, f'{result.get(key, 0)}%')
            st.progress(result.get("overall", 0) / 100)
            left, right = st.columns(2)
            with left:
                st.markdown("**What worked**")
                for value in result.get("strengths", []): st.write("✓ " + value)
            with right:
                st.markdown("**Next improvement**")
                for value in result.get("improvements", []): st.write("→ " + value)
            with st.expander("See an improved answer example"):
                st.write(result.get("improved_answer", ""))
            st.info("Follow-up: " + result.get("follow_up_question", ""))

with delivery_tab:
    st.subheader("Live and recorded visual delivery evaluation")
    st.markdown('<div class="notice"><b>Important:</b> These models describe visible dataset-label outputs only. They cannot determine true emotion, nervousness, confidence, honesty, personality or job suitability. Visual results remain separate from the answer score.</div>', unsafe_allow_html=True)
    face_model_status = get_face_model_status()
    if not face_model_status["ready"]:
        st.warning(face_model_status["message"])
    confidence_model_status = get_confidence_model_status()
    if not confidence_model_status["ready"]:
        st.warning(confidence_model_status["message"])

    st.markdown("### Record an interview video")
    st.write(
        "Record your video and voice first. To keep the camera smooth, no AI "
        "models run while you are recording. After you stop, the app analyses "
        "the saved video and then provides transcript, voice and answer feedback."
    )

    if (
        not st.session_state.get("simple_recording_prepared", False)
        and not st.session_state.get("live_recording_finalized", False)
    ):
        if st.button(
            "Prepare camera and microphone",
            type="primary",
            use_container_width=True,
            key="prepare_simple_recording"
        ):
            old_recording = st.session_state.get("live_recording_path", "")
            old_audio = st.session_state.get("live_recording_audio_path", "")
            for old_path in (old_recording, old_audio):
                if old_path and os.path.exists(old_path):
                    os.remove(old_path)

            recording_id = uuid.uuid4().hex
            st.session_state.live_recording_path = os.path.join(
                tempfile.gettempdir(),
                "candidate360_" + recording_id + ".flv"
            )
            st.session_state.simple_recording_prepared = True
            st.session_state.live_recording_finalized = False
            st.session_state.live_video_transcript = ""
            st.session_state.pop("live_video_corrected_transcript", None)
            st.session_state.pop("live_video_voice_evaluation", None)
            st.session_state.pop("live_video_answer_evaluation", None)
            st.rerun()

    if (
        st.session_state.get("simple_recording_prepared", False)
        and not st.session_state.get("live_recording_finalized", False)
    ):
        try:
            from aiortc.contrib.media import MediaRecorder
            from streamlit_webrtc import WebRtcMode, webrtc_streamer

            live_recording_path = st.session_state.live_recording_path

            def simple_recorder_factory():
                return MediaRecorder(live_recording_path, format="flv")

            simple_context = webrtc_streamer(
                key="candidate360_recording_camera",
                mode=WebRtcMode.SENDONLY,
                media_stream_constraints={
                    "video": {
                        "width": {"ideal": 640},
                        "height": {"ideal": 480},
                        "frameRate": {"ideal": 15, "max": 20}
                    },
                    "audio": {
                        "echoCancellation": True,
                        "noiseSuppression": True,
                        "autoGainControl": True
                    }
                },
                rtc_configuration=get_rtc_configuration(),
                in_recorder_factory=simple_recorder_factory,
                async_processing=False
            )

            recording_ready = (
                os.path.exists(live_recording_path)
                and os.path.getsize(live_recording_path) > 1024
                and not simple_context.state.playing
            )

            if recording_ready:
                st.success(
                    "Recording complete. Preview it, then analyse the saved "
                    "video. The analysis does not run while the camera is open."
                )
                st.video(live_recording_path)

                if st.button(
                    "Analyse recorded interview",
                    type="primary",
                    use_container_width=True,
                    key="analyse_simple_recording"
                ):
                    try:
                        from vision_utils import analyse_video

                        with st.spinner(
                            "Sampling the saved video and creating visual "
                            "observations..."
                        ):
                            pose_model = get_pose_model()
                            expression_model = get_expression_model()
                            confidence_model = get_confidence_model()
                            gesture_recognizer = get_gesture_recognizer()
                            visual_result = analyse_video(
                                live_recording_path,
                                pose_model,
                                expression_model,
                                confidence_model,
                                gesture_recognizer
                            )

                        st.session_state.delivery_result = visual_result
                        st.session_state.live_recording_finalized = True
                        st.session_state.simple_recording_prepared = False
                        release_visual_models()
                        st.rerun()
                    except Exception as error:
                        release_visual_models()
                        st.error(str(error))
        except ImportError as error:
            st.session_state.simple_recording_prepared = False
            st.error(
                "Video recording is unavailable. Check streamlit-webrtc in "
                "requirements.txt. Details: " + str(error)
            )
        except Exception as error:
            st.session_state.simple_recording_prepared = False
            st.error(
                "The recorder could not start. Allow camera and microphone "
                "permission, then try again. Details: " + str(error)
            )

    if st.session_state.get("live_recording_finalized", False):
        live_recording_path = st.session_state.get("live_recording_path", "")
        if live_recording_path and os.path.exists(live_recording_path):
            st.markdown("### Recorded-video feedback")
            st.video(live_recording_path)

            if st.button(
                "Create transcript from this video",
                use_container_width=True,
                key="transcribe_simple_recording"
            ):
                try:
                    from speech_utils import (
                        extract_audio_from_video,
                        transcribe_audio_path
                    )

                    with st.spinner(
                        "Extracting the recorded voice and running Whisper..."
                    ):
                        audio_path = extract_audio_from_video(
                            live_recording_path
                        )
                        transcript = transcribe_audio_path(
                            audio_path,
                            get_speech_model()
                        )

                    old_audio = st.session_state.get(
                        "live_recording_audio_path",
                        ""
                    )
                    if (
                        old_audio
                        and old_audio != audio_path
                        and os.path.exists(old_audio)
                    ):
                        os.remove(old_audio)
                    st.session_state.live_recording_audio_path = audio_path
                    st.session_state.live_video_transcript = transcript
                    st.session_state.live_video_corrected_transcript = transcript
                    st.success(
                        "Transcript created. Correct any mistakes before "
                        "requesting feedback."
                    )
                except Exception as error:
                    st.error(str(error))

            if "live_video_corrected_transcript" not in st.session_state:
                st.session_state.live_video_corrected_transcript = (
                    st.session_state.get("live_video_transcript", "")
                )

            corrected_live_transcript = st.text_area(
                "Recorded-video answer / corrected transcript",
                key="live_video_corrected_transcript",
                height=190
            )

            voice_column, answer_column = st.columns(2)
            with voice_column:
                if st.button(
                    "Evaluate voice delivery",
                    use_container_width=True,
                    key="evaluate_simple_video_voice"
                ):
                    try:
                        from speech_utils import evaluate_voice_delivery_path

                        audio_path = st.session_state.get(
                            "live_recording_audio_path",
                            ""
                        )
                        if not audio_path or not os.path.exists(audio_path):
                            st.warning(
                                "Create the transcript first so the recorded "
                                "audio is ready."
                            )
                        else:
                            st.session_state.live_video_voice_evaluation = (
                                evaluate_voice_delivery_path(
                                    audio_path,
                                    corrected_live_transcript
                                )
                            )
                    except Exception as error:
                        st.error(str(error))

            with answer_column:
                if st.button(
                    "Ask the Answer Coach",
                    type="primary",
                    use_container_width=True,
                    key="evaluate_simple_video_answer"
                ):
                    questions = st.session_state.get("questions", [])
                    if not questions:
                        st.warning("Complete Role Lab first.")
                    elif not corrected_live_transcript.strip():
                        st.warning("Create and check the transcript first.")
                    else:
                        try:
                            question_index = min(
                                st.session_state.get("question_index", 0),
                                len(questions) - 1
                            )
                            video_question = questions[question_index].get(
                                "question",
                                ""
                            )
                            with st.spinner(
                                "The Answer Coach is reviewing the answer..."
                            ):
                                video_evaluation = evaluate_answer(
                                    video_question,
                                    corrected_live_transcript,
                                    st.session_state.cv_text,
                                    st.session_state.job_description,
                                    api_key
                                )
                            st.session_state.live_video_answer_evaluation = (
                                video_evaluation
                            )
                        except Exception as error:
                            st.error(str(error))

            live_voice_result = st.session_state.get(
                "live_video_voice_evaluation",
                {}
            )
            if live_voice_result:
                st.markdown("#### Voice-delivery observations")
                voice_metrics = st.columns(3)
                voice_metrics[0].metric(
                    "Duration",
                    f'{live_voice_result.get("duration_seconds", 0)} sec'
                )
                voice_metrics[1].metric(
                    "Speaking pace",
                    f'{live_voice_result.get("words_per_minute", 0)} words/min'
                )
                voice_metrics[2].metric(
                    "Listed filler words",
                    live_voice_result.get("filler_count", 0)
                )
                for feedback_item in live_voice_result.get("feedback", []):
                    st.write("→ " + feedback_item)

            live_answer_result = st.session_state.get(
                "live_video_answer_evaluation",
                {}
            )
            if live_answer_result:
                st.markdown("#### Recorded-video Answer Coach")
                answer_metrics = st.columns(5)
                for column, label, key in zip(
                    answer_metrics,
                    [
                        "Relevance",
                        "Specificity",
                        "Structure",
                        "Role fit",
                        "Overall"
                    ],
                    [
                        "relevance",
                        "specificity",
                        "structure",
                        "job_alignment",
                        "overall"
                    ]
                ):
                    column.metric(label, f'{live_answer_result.get(key, 0)}%')
                for value in live_answer_result.get("strengths", []):
                    st.write("✓ " + value)
                for value in live_answer_result.get("improvements", []):
                    st.write("→ " + value)

            if st.button(
                "Record another interview",
                use_container_width=True,
                key="reset_simple_recording"
            ):
                st.session_state.live_recording_finalized = False
                st.session_state.simple_recording_prepared = False
                st.session_state.pop("live_video_corrected_transcript", None)
                st.rerun()

    st.divider()
    st.markdown("### Uploaded image or recorded video")
    media_file = st.file_uploader("Upload an interview image or video", type=["jpg", "jpeg", "png", "mp4", "mov", "avi"], key="delivery_media")
    if media_file is not None:
        if media_file.type.startswith("video"):
            st.video(media_file)
        else:
            st.image(media_file, width=520)

        if st.button("Run person + face-model observation"):
            try:
                media_fingerprint = hashlib.sha256(
                    media_file.getvalue()
                ).hexdigest()

                if (
                    st.session_state.get("delivery_media_fingerprint")
                    == media_fingerprint
                    and "delivery_result" in st.session_state
                ):
                    st.info("Reusing the completed observation for this file.")
                else:
                    with st.spinner(
                        "Sampling the video and running fast batched analysis..."
                    ):
                        from vision_utils import analyse_image, analyse_video, save_upload
                        media_path = save_upload(media_file)
                        try:
                            (
                                pose_model,
                                expression_model_data,
                                confidence_model_data
                            ) = get_vision_models()
                            if media_file.type.startswith("video"):
                                delivery_result = analyse_video(
                                    media_path,
                                    pose_model,
                                    expression_model_data,
                                    confidence_model_data
                                )
                            else:
                                delivery_result = analyse_image(
                                    media_path,
                                    pose_model,
                                    expression_model_data,
                                    confidence_model_data
                                )
                            st.session_state.delivery_result = delivery_result
                            st.session_state.delivery_media_fingerprint = (
                                media_fingerprint
                            )
                        finally:
                            if os.path.exists(media_path):
                                os.remove(media_path)
                    st.success("Observation complete.")
            except Exception as error:
                st.error(str(error))

    if "delivery_result" in st.session_state:
        result = st.session_state.delivery_result
        st.markdown("### Observation summary")
        expressions = result.get("expressions", [])

        if result.get("source") == "live_camera":
            first, second, third, fourth = st.columns(4)
            first.metric(
                "YOLO person visible",
                f'{result.get("person_visibility", 0)}%'
            )
            second.metric(
                "Face visible",
                f'{result.get("face_visibility", 0)}%'
            )
            third.metric(
                "Most frequent expression output",
                result.get("main_expression", "Not available").title()
            )
            fourth.metric(
                "Most frequent confidence-dataset output",
                result.get(
                    "main_confidence_output",
                    "Not available"
                ).title()
            )
        elif "analysed_frames" in result:
            first, second = st.columns(2)
            first.metric("Person visible", f'{result.get("person_visibility", 0)}%')
            second.metric(
                "Most frequent expression output",
                result.get("main_expression", "Not available").title()
            )
        else:
            first, second = st.columns(2)
            first.metric("People detected", result.get("people", 0))

            main_expression = "Not available"
            if expressions:
                main_expression = expressions[0].get("label", "Not available").title()

            second.metric("Top expression output", main_expression)

        if expressions:
            if result.get("source") == "live_camera":
                st.markdown("#### Live facial-expression distribution")
                st.caption(
                    "Percentage of classified live-camera samples in which "
                    "each label was the model's top output."
                )
            elif "analysed_frames" in result:
                st.markdown("#### Facial-expression distribution")
                st.caption(
                    "Percentage of successfully classified sampled video "
                    "moments in which each label was the model's top output."
                )
            else:
                st.markdown("#### Ranked expression-model outputs")
                st.caption(
                    "Model confidence scores for the uploaded image."
                )
            for item in expressions:
                score = round(item.get("score", 0) * 100, 1)
                st.write(f'{item.get("label", "Unknown").title()}: {score}%')
                st.progress(min(max(item.get("score", 0), 0), 1))
        else:
            expression_error = result.get("expression_error", "")
            if expression_error:
                st.error("Expression model error: " + expression_error)
            elif result.get("face_error"):
                st.info(result["face_error"])
            else:
                st.info("Expression output is unavailable. Add face_expression_model.keras and class_names.json to the models folder.")

        if result.get("evaluation_feedback"):
            st.info(result["evaluation_feedback"])

        confidence_outputs = result.get("confidence_outputs", [])
        if confidence_outputs:
            st.markdown("#### Confidence-dataset label distribution")
            if result.get("source") == "live_camera":
                st.caption(
                    "Percentage of classified live-camera samples assigned "
                    "to each confidence-dataset label."
                )
            elif "analysed_frames" in result:
                st.caption(
                    "Percentage of successfully classified face-cropped "
                    "video moments assigned to each dataset label."
                )
            else:
                st.caption(
                    "Classifier scores for the dataset labels in the "
                    "face-cropped image."
                )
            for item in confidence_outputs:
                score = round(item.get("score", 0) * 100, 1)
                st.write(f'{item.get("label", "Unknown").title()}: {score}%')
                st.progress(min(max(item.get("score", 0), 0), 1))
        else:
            confidence_error = result.get("confidence_error", "")
            if confidence_error:
                st.error("Confidence-dataset model error: " + confidence_error)
            elif result.get("face_error"):
                st.info(result["face_error"])
            else:
                st.info(
                    "Confidence-dataset output is unavailable. Add "
                    "confidence_model.keras and "
                    "confidence_class_names.json to the models folder."
                )

        st.caption(
            "The confident/unconfident labels were manually assigned in "
            "the source dataset. They are not verified measurements of "
            "confidence or nervousness and are never used in answer scores."
        )

        gestures = result.get("gestures", [])
        if gestures:
            st.markdown("#### Hand-gesture distribution")
            st.caption(
                "Percentage of sampled moments containing a detected hand "
                "that were assigned to each supported gesture label."
            )
            for item in gestures:
                score = round(item.get("score", 0) * 100, 1)
                st.write(
                    f'{item.get("label", "Unknown").title()}: {score}%'
                )
                st.progress(min(max(item.get("score", 0), 0), 1))
        elif result.get("source") == "live_camera":
            gesture_error = result.get("gesture_error", "")
            if gesture_error:
                st.error("Hand-gesture model error: " + gesture_error)
            else:
                st.info("No supported hand gesture was detected.")

with summary_tab:
    st.subheader("Turn practice into a plan")
    answer_results = st.session_state.answer_results
    if not answer_results:
        st.info("Evaluate at least one answer to unlock the debrief.")
    else:
        scores = [item["evaluation"].get("overall", 0) for item in answer_results]
        a, b, c = st.columns(3)
        a.metric("Answers completed", len(answer_results))
        b.metric("Average practice score", f'{round(sum(scores) / len(scores))}%')
        c.metric("Best answer score", f'{max(scores)}%')

        if st.button("Create final coach summary", type="primary", use_container_width=True):
            try:
                with st.spinner("The Summary Coach is preparing your next practice plan..."):
                    st.session_state.session_summary = create_session_summary(
                        target_role, st.session_state.get("cv_analysis", {}), answer_results, api_key
                    )
            except Exception as error:
                st.error(str(error))

        if "session_summary" in st.session_state:
            result = st.session_state.session_summary
            st.markdown("### " + result.get("headline", "Your interview debrief"))
            st.write(result.get("overall_progress", ""))
            left, right = st.columns(2)
            with left:
                st.markdown("**Top strengths**")
                for value in result.get("top_strengths", []): st.write("✓ " + value)
                st.markdown("**Priority actions**")
                for value in result.get("priority_actions", []): st.write("→ " + value)
            with right:
                st.markdown("**Next practice plan**")
                for number, value in enumerate(result.get("next_practice_plan", []), 1):
                    st.write(f"{number}. {value}")

            report = {
                "target_role": target_role,
                "cv_analysis": st.session_state.get("cv_analysis", {}),
                "answers": answer_results,
                "summary": result,
                "visual_observations": st.session_state.get("delivery_result", {})
            }
            html_report = create_html_report(report)
            st.download_button(
                "Download readable session report",
                html_report,
                file_name="candidate360_session_report.html",
                mime="text/html",
                use_container_width=True
            )
            with st.expander("Technical data export"):
                st.caption("JSON is included for technical review or future application development.")
                st.download_button(
                    "Download JSON data",
                    json.dumps(report, indent=2, ensure_ascii=False),
                    file_name="candidate360_session_data.json",
                    mime="application/json",
                    use_container_width=True
                )

st.markdown("---")
st.caption("Interview practice support only. Feedback must not be used as an automated hiring decision or as evidence of protected traits, personality, honesty, confidence or nervousness.")


