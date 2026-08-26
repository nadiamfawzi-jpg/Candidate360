import os
import tempfile

from transformers import pipeline


def load_speech_model():
    speech_model = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-small"
    )
    return speech_model


def save_audio(uploaded_audio):
    suffix = os.path.splitext(uploaded_audio.name)[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as audio_file:
        audio_file.write(uploaded_audio.getbuffer())
        audio_path = audio_file.name
    return audio_path


def transcribe_audio(uploaded_audio, speech_model):
    audio_path = save_audio(uploaded_audio)

    try:
        transcript = speech_model(
            audio_path,
            chunk_length_s=30,
            stride_length_s=5,
            generate_kwargs={"task": "transcribe"}
        )
        return transcript.get("text", "").strip()
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)
