# 💼 Candidate360

Candidate360 is an AI-powered interview-practice application that adapts each session to the candidate’s CV, selected role and job description. It combines tailored questions with answer coaching, multilingual speech transcription and separate visual-delivery observations.

🌐 **Live app:** [Candidate360 on Streamlit](https://candidate360-c7jluz3jvtx39hw5pcunvv.streamlit.app/)

## 💡 The problem

Many interview-practice tools provide generic questions and feedback. They may not consider the candidate’s experience, the requirements of the target job or how the answer is delivered.

## 🚀 The solution

Candidate360 connects these areas in one practice workflow:

1. 📄 **Role Lab** reads the CV and job description, checks role alignment and creates tailored interview questions.
2. 🎙️ **Interview Room** accepts a typed answer, microphone recording or uploaded audio file.
3. 🎥 **Delivery Lab** processes a recorded or uploaded interview video through separate visual and speech routes.
4. 📊 **Debrief** summarises strengths, improvements and suggested next-practice actions.

## ✨ Main features

### 📄 Role and CV preparation

- Upload CV files in PDF, DOCX or TXT format.
- Paste a real job description and select or enter a target role or major.
- Display a role/job match, partial-match or mismatch notice.
- Generate six role-aware interview questions through a language model.

### 🎙️ Answer and speech coaching

- Answer by text, microphone recording or desktop audio upload.
- Accept WAV, MP3, M4A and FLAC audio recordings.
- Create an editable multilingual transcript before answer evaluation.
- Evaluate relevance, specificity, structure and job alignment.
- Provide strengths, improvements, a stronger example and a follow-up question.

### 🎥 Video and visual observations

- Record or upload interview videos in common formats.
- Extract and transcribe speech from an uploaded video.
- Report speaking duration, words per minute and listed filler words.
- Estimate person visibility using YOLO pose detection.
- Display facial-expression model outputs from the FER-2013 classifier.
- Display confident/unconfident appearance-label outputs from the second classifier.
- Recognise supported hand gestures using MediaPipe.

### 📊 Session report

- Generate a downloadable practice report.

## 🔗 How OpenRouter is used

OpenRouter connects Candidate360 to the selected large language model. The app sends the candidate’s CV, target role, job description, interview question and answer through the OpenRouter API. The language model then generates tailored questions and coaching feedback.

OpenRouter provides the connection; the selected language model generates the output.

## 🗣️ Speech transcription

Candidate360 supports two transcription routes:

- **Recommended hosted route:** Groq Whisper Large V3 Turbo, enabled with `GROQ_API_KEY`.
- **Local fallback:** Hugging Face `openai/whisper-base` when a Groq key is not configured.

The candidate can select the spoken language or use automatic detection. The transcript remains editable because speech recognition can mishear accents, names, technical vocabulary or unclear recordings.

For better transcription:

- record in a quiet location;
- speak clearly and close to the microphone;
- select the correct spoken language when known;
- check and correct the transcript before evaluation;
- use the Groq transcription route for Streamlit deployment when available.

## 👁️ Visual models

| Component | Purpose |
|---|---|
| YOLO pose model | Checks whether a person is visible in sampled video moments. |
| FER-2013 Xception classifier | Produces seven facial-expression dataset-label scores. |
| Confident/Unconfident classifier | Produces the two appearance labels used in its source dataset. |
| MediaPipe Gesture Recognizer | Recognises supported hand-gesture categories. |

Visual labels are shown separately from the answer evaluation. They do not change the candidate’s answer score and should not be treated as verified emotions, confidence, personality or employability.

## 📁 Required project structure

```text
Candidate360/
├── app.py
├── candidate360_api.py
├── document_utils.py
├── speech_utils.py
├── vision_utils.py
├── requirements.txt
├── packages.txt
├── README.md
└── models/
    ├── face_expression_model.keras
    ├── class_names.json
    ├── confidence_model.keras
    ├── confidence_class_names.json
    └── gesture_recognizer.task
```

The YOLO pose weights are loaded through Ultralytics. The MediaPipe task file can also be downloaded automatically by `vision_utils.py` if it is missing.

## 🔐 Streamlit secrets

Open **Manage app → Settings → Secrets** and add:

```toml
OPENROUTER_API_KEY = "your_openrouter_api_key"
GROQ_API_KEY = "your_groq_api_key"
```

- `OPENROUTER_API_KEY` enables tailored questions and answer feedback.
- `GROQ_API_KEY` enables the recommended hosted multilingual transcription.

Do not place secret keys directly in `app.py`, `candidate360_api.py`, `README.md` or any GitHub file.

## 💻 Run locally

1. Download or clone the repository.
2. Open a terminal inside the project folder.
3. Install the Python dependencies:

```bash
pip install -r requirements.txt
```

4. Install FFmpeg on the computer. Streamlit Cloud reads this requirement from `packages.txt`.
5. Create `.streamlit/secrets.toml` and add the API keys.
6. Start the application:

```bash
streamlit run app.py
```

## ☁️ Deploy on Streamlit Community Cloud

1. Upload all project files to the GitHub repository using the structure above.
2. Keep `requirements.txt` and `packages.txt` in the same main folder as `app.py`.
3. Store large `.keras` files with Git LFS if normal GitHub upload limits prevent uploading them.
4. Connect the repository to Streamlit Community Cloud.
5. Set the main file path to `app.py`.
6. Add the OpenRouter and Groq keys to Streamlit Secrets.
7. Reboot the app after changing dependencies, model files or secrets.

## ⚠️ Important limitations

- Language-model feedback can be incomplete or incorrect and should be reviewed by the candidate.
- Speech accuracy varies with language, accent, audio quality and background noise.
- Facial-expression and confidence-dataset labels are imperfect visual classifications, not psychological measurements.
- MediaPipe recognises only the gestures supported by its task model.
- Video analysis samples selected moments rather than processing every frame.
- Candidate360 is designed for practice and self-reflection. It should not be used to make hiring decisions.

## 🧰 Technology used

- Python
- Streamlit
- OpenRouter with an OpenAI-compatible client
- Groq speech-to-text API
- Hugging Face Transformers and Whisper
- TensorFlow/Keras and Xception transfer learning
- Ultralytics YOLO
- OpenCV
- MediaPipe
- FFmpeg

## 👤 Project author

**Nadia Mohamed Fawzi**  
General Assembly Data Science Bootcamp · 2026
