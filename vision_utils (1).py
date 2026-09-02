import json
import math
import os
import tempfile
from functools import lru_cache

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from ultralytics import YOLO


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(PROJECT_DIR, "models")

EXPRESSION_MODEL_PATH = os.path.join(MODEL_DIR, "face_expression_model.keras")
EXPRESSION_CLASS_NAMES_PATH = os.path.join(MODEL_DIR, "class_names.json")
CONFIDENCE_MODEL_PATH = os.path.join(MODEL_DIR, "confidence_model.keras")
CONFIDENCE_CLASS_NAMES_PATH = os.path.join(
    MODEL_DIR,
    "confidence_class_names.json"
)

IMAGE_SIZE = (128, 128)
POSE_IMAGE_SIZE = 320
VIDEO_SAMPLES_PER_SECOND = 1
VIDEO_MAX_SAMPLES = 24
VIDEO_MIN_SAMPLES = 6
FACE_CASCADE_PATH = os.path.join(
    cv2.data.haarcascades,
    "haarcascade_frontalface_default.xml"
)


def load_pose_model():
    return YOLO("yolo26n-pose.pt")


def _get_model_status(model_path, class_path, display_name):
    missing_files = []
    for path in (model_path, class_path):
        if not os.path.exists(path):
            missing_files.append(os.path.join("models", os.path.basename(path)))

    if missing_files:
        return {
            "ready": False,
            "message": (
                f"{display_name} output is unavailable. Missing: "
                + ", ".join(missing_files)
            )
        }

    return {
        "ready": True,
        "message": f"{display_name} model files are available."
    }


def get_expression_model_status():
    return _get_model_status(
        EXPRESSION_MODEL_PATH,
        EXPRESSION_CLASS_NAMES_PATH,
        "Facial-expression"
    )


def get_confidence_model_status():
    return _get_model_status(
        CONFIDENCE_MODEL_PATH,
        CONFIDENCE_CLASS_NAMES_PATH,
        "Confidence-dataset"
    )


def _load_classifier(model_path, class_path, status):
    if not status["ready"]:
        return None, []

    model = load_model(model_path, compile=False)
    with open(class_path, "r", encoding="utf-8") as class_file:
        class_names = json.load(class_file)

    output_classes = int(model.output_shape[-1])
    if len(class_names) != output_classes:
        raise ValueError(
            f"The model has {output_classes} outputs but "
            f"{os.path.basename(class_path)} has {len(class_names)} labels."
        )

    return model, class_names


def load_expression_model():
    return _load_classifier(
        EXPRESSION_MODEL_PATH,
        EXPRESSION_CLASS_NAMES_PATH,
        get_expression_model_status()
    )


def load_confidence_model():
    return _load_classifier(
        CONFIDENCE_MODEL_PATH,
        CONFIDENCE_CLASS_NAMES_PATH,
        get_confidence_model_status()
    )


@lru_cache(maxsize=1)
def load_face_detector():
    detector = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    if detector.empty():
        raise RuntimeError("OpenCV could not load its frontal-face detector.")
    return detector


def extract_largest_face(frame, face_detector, padding_ratio=0.15):
    """Return the largest detected face with a small context margin."""
    if frame is None or face_detector is None:
        return None

    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(
        grayscale,
        scaleFactor=1.15,
        minNeighbors=5,
        minSize=(48, 48)
    )
    if len(faces) == 0:
        return None

    x, y, width, height = max(faces, key=lambda box: box[2] * box[3])
    pad_x = int(width * padding_ratio)
    pad_y = int(height * padding_ratio)
    image_height, image_width = frame.shape[:2]

    x1 = max(x - pad_x, 0)
    y1 = max(y - pad_y, 0)
    x2 = min(x + width + pad_x, image_width)
    y2 = min(y + height + pad_y, image_height)
    return frame[y1:y2, x1:x2]


def prepare_face_batch(face_crops):
    prepared_faces = []
    for face_crop in face_crops:
        if face_crop is None or face_crop.size == 0:
            continue
        rgb_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        resized_face = cv2.resize(
            rgb_face,
            IMAGE_SIZE,
            interpolation=cv2.INTER_AREA
        )
        prepared_faces.append(
            np.asarray(resized_face, dtype=np.float32)
        )

    if not prepared_faces:
        return np.empty((0, *IMAGE_SIZE, 3), dtype=np.float32)

    return np.stack(prepared_faces)


def predict_face_batch(prepared_faces, model_data):
    model, class_names = model_data
    number_of_faces = len(prepared_faces)
    if model is None or number_of_faces == 0:
        return [[] for _ in range(number_of_faces)]

    prediction_batch = np.asarray(
        model(prepared_faces, training=False)
    )

    batch_results = []
    for prediction in prediction_batch:
        results = [
            {"label": class_name, "score": float(score)}
            for class_name, score in zip(class_names, prediction)
        ]
        batch_results.append(
            sorted(results, key=lambda item: item["score"], reverse=True)
        )
    return batch_results


def save_upload(uploaded_file):
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as saved_file:
        saved_file.write(uploaded_file.getbuffer())
        return saved_file.name


def _summarise_prediction_batch(prediction_batch):
    label_counts = {}
    for predictions in prediction_batch:
        if not predictions:
            continue
        label = predictions[0]["label"]
        label_counts[label] = label_counts.get(label, 0) + 1

    classified_frames = sum(label_counts.values())
    if not classified_frames:
        return []

    results = [
        {
            "label": label,
            "score": count / classified_frames,
            "observations": count
        }
        for label, count in label_counts.items()
    ]
    return sorted(results, key=lambda item: item["score"], reverse=True)


def _read_video_samples(
    video_path,
    samples_per_second=VIDEO_SAMPLES_PER_SECOND,
    max_samples=VIDEO_MAX_SAMPLES,
    min_samples=VIDEO_MIN_SAMPLES
):
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError("The uploaded video could not be opened.")

    try:
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frames_per_second = capture.get(cv2.CAP_PROP_FPS)
        if not frames_per_second or frames_per_second <= 0:
            frames_per_second = 30.0

        if total_frames > 0:
            duration_seconds = total_frames / frames_per_second
            target_samples = max(
                min_samples,
                int(math.ceil(duration_seconds * samples_per_second))
            )
            target_samples = min(max_samples, total_frames, target_samples)
            sample_positions = np.linspace(
                0,
                total_frames - 1,
                num=target_samples,
                dtype=int
            )

            frames = []
            for position in sample_positions:
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(position))
                success, frame = capture.read()
                if success:
                    frames.append(frame)
            return frames

        sample_every = max(
            1,
            int(frames_per_second / samples_per_second)
        )
        frames = []
        frame_number = 0
        while capture.isOpened() and len(frames) < max_samples:
            success, frame = capture.read()
            if not success:
                break
            if frame_number % sample_every == 0:
                frames.append(frame)
            frame_number += 1
        return frames
    finally:
        capture.release()


def _count_person_frames(pose_model, frames):
    if not frames:
        return 0

    pose_results = pose_model(
        frames,
        verbose=False,
        imgsz=POSE_IMAGE_SIZE
    )
    return sum(
        1
        for result in pose_results
        if result.boxes is not None and len(result.boxes) > 0
    )


def analyse_image(
    image_path,
    pose_model,
    expression_model_data,
    confidence_model_data
):
    pose_result = pose_model(
        image_path,
        verbose=False,
        imgsz=POSE_IMAGE_SIZE
    )[0]
    people = len(pose_result.boxes) if pose_result.boxes is not None else 0

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("The uploaded image could not be read.")

    face_crop = extract_largest_face(image, load_face_detector())
    face_detected = face_crop is not None
    prepared_faces = prepare_face_batch([face_crop])

    expression_batch = predict_face_batch(
        prepared_faces,
        expression_model_data
    )
    confidence_batch = predict_face_batch(
        prepared_faces,
        confidence_model_data
    )
    expressions = expression_batch[0][:5] if expression_batch else []
    confidence_outputs = confidence_batch[0][:2] if confidence_batch else []

    face_error = ""
    if not face_detected:
        face_error = (
            "No clear frontal face was detected. Try a brighter image "
            "with the face larger and facing the camera."
        )

    return {
        "people": people,
        "face_detected": face_detected,
        "face_error": face_error,
        "expressions": expressions,
        "main_expression": (
            expressions[0]["label"] if expressions else "Not available"
        ),
        "confidence_outputs": confidence_outputs,
        "main_confidence_output": (
            confidence_outputs[0]["label"]
            if confidence_outputs else "Not available"
        )
    }


def analyse_video(
    video_path,
    pose_model,
    expression_model_data,
    confidence_model_data,
    samples_per_second=VIDEO_SAMPLES_PER_SECOND,
    max_frames=VIDEO_MAX_SAMPLES
):
    frames = _read_video_samples(
        video_path,
        samples_per_second=samples_per_second,
        max_samples=max_frames
    )
    analysed_frames = len(frames)
    if not analysed_frames:
        raise ValueError("No readable frames were found in the video.")

    person_frames = _count_person_frames(pose_model, frames)

    face_detector = load_face_detector()
    face_crops = []
    for frame in frames:
        face_crop = extract_largest_face(frame, face_detector)
        if face_crop is not None:
            face_crops.append(face_crop)

    prepared_faces = prepare_face_batch(face_crops)
    face_frames = len(prepared_faces)

    expression_error = ""
    confidence_error = ""

    try:
        expression_batch = predict_face_batch(
            prepared_faces,
            expression_model_data
        )
    except Exception as error:
        expression_batch = []
        expression_error = str(error)

    try:
        confidence_batch = predict_face_batch(
            prepared_faces,
            confidence_model_data
        )
    except Exception as error:
        confidence_batch = []
        confidence_error = str(error)

    expressions = _summarise_prediction_batch(expression_batch)
    confidence_outputs = _summarise_prediction_batch(confidence_batch)

    face_error = ""
    if not face_frames:
        face_error = (
            "No clear frontal face was detected in the sampled moments. "
            "Use brighter lighting and keep the face larger and facing "
            "the camera."
        )

    return {
        "analysed_frames": analysed_frames,
        "person_visibility": round(person_frames / analysed_frames * 100),
        "face_visibility": round(face_frames / analysed_frames * 100),
        "face_error": face_error,
        "main_expression": (
            expressions[0]["label"] if expressions else "Not available"
        ),
        "expressions": expressions,
        "expression_error": expression_error,
        "main_confidence_output": (
            confidence_outputs[0]["label"]
            if confidence_outputs else "Not available"
        ),
        "confidence_outputs": confidence_outputs,
        "confidence_error": confidence_error
    }
