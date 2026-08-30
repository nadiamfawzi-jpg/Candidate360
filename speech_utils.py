import os
import re
import tempfile

from mutagen import File as MutagenFile
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


def get_audio_duration(uploaded_audio):
    """Read the recording duration without changing or resampling the audio."""
    audio_path = save_audio(uploaded_audio)

    try:
        audio_information = MutagenFile(audio_path)
        if audio_information is None or audio_information.info is None:
            raise ValueError(
                "The recording duration could not be read. Try WAV, MP3, "
                "M4A or FLAC audio."
            )

        duration_seconds = float(audio_information.info.length)
        if duration_seconds <= 0:
            raise ValueError("The recording does not contain readable audio.")
        return duration_seconds
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)


def evaluate_voice_delivery(uploaded_audio, corrected_transcript):
    """Return simple delivery observations from duration and transcript text."""
    transcript = corrected_transcript.strip()
    if not transcript:
        raise ValueError("Create and check the transcript before evaluating delivery.")

    duration_seconds = get_audio_duration(uploaded_audio)
    words = re.findall(r"\b[\w'-]+\b", transcript, flags=re.UNICODE)
    word_count = len(words)
    words_per_minute = round(word_count / duration_seconds * 60)

    filler_patterns = {
        "um": r"\b(?:um+|umm+)\b",
        "uh": r"\b(?:uh+|uhh+)\b",
        "you know": r"\byou\s+know\b",
        "basically": r"\bbasically\b",
        "actually": r"\bactually\b",
        "يعني": r"\bيعني\b",
        "امم": r"\b(?:ام+|أم+)\b",
    }

    filler_details = {}
    for label, pattern in filler_patterns.items():
        matches = re.findall(pattern, transcript, flags=re.IGNORECASE)
        if matches:
            filler_details[label] = len(matches)

    filler_count = sum(filler_details.values())
    filler_rate = round(filler_count / max(word_count, 1) * 100, 1)

    if words_per_minute < 100:
        pace_label = "Slow"
        pace_feedback = (
            "The pace was slower than the broad practice range. Rehearse once "
            "more while keeping short, natural pauses."
        )
    elif words_per_minute <= 170:
        pace_label = "Balanced"
        pace_feedback = (
            "The speaking pace was within the broad interview-practice range."
        )
    else:
        pace_label = "Fast"
        pace_feedback = (
            "The pace was faster than the broad practice range. Slow slightly "
            "and separate the main points."
        )

    if filler_rate <= 2:
        filler_label = "Few detected"
        filler_feedback = "Few listed filler words were detected in the transcript."
    elif filler_rate <= 5:
        filler_label = "Some detected"
        filler_feedback = (
            "Some listed filler words were detected. Replace them with a short pause."
        )
    else:
        filler_label = "Frequent"
        filler_feedback = (
            "Listed filler words appeared frequently. Practise the answer in "
            "shorter sections and pause between ideas."
        )

    feedback = [pace_feedback, filler_feedback]
    if duration_seconds < 15:
        feedback.append(
            "The recording was brief. Check that the answer includes an example "
            "and a clear result where the question requires them."
        )

    return {
        "duration_seconds": round(duration_seconds, 1),
        "word_count": word_count,
        "words_per_minute": words_per_minute,
        "pace_label": pace_label,
        "filler_count": filler_count,
        "filler_rate": filler_rate,
        "filler_label": filler_label,
        "filler_details": filler_details,
        "feedback": feedback,
    }
