import json
import math
import os
import tempfile
import threading
import time
import urllib.request
from functools import lru_cache

import cv2
import numpy as np
from tensorflow.keras.models import load_model


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(PROJECT_DIR, "models")

EXPRESSION_MODEL_PATH = os.path.join(MODEL_DIR, "face_expression_model.keras")
EXPRESSION_CLASS_NAMES_PATH = os.path.join(MODEL_DIR, "class_names.json")
CONFIDENCE_MODEL_PATH = os.path.join(MODEL_DIR, "confidence_model.keras")
CONFIDENCE_CLASS_NAMES_PATH = os.path.join(
    MODEL_DIR,
    "confidence_class_names.json"
)
GESTURE_MODEL_PATH = os.path.join(MODEL_DIR, "gesture_recognizer.task")
GESTURE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
    "gesture_recognizer/float16/latest/gesture_recognizer.task"
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
    from ultralytics import YOLO
    return YOLO("yolo26n-pose.pt")


def load_live_person_model():
    from ultralytics import YOLO
    return YOLO("yolo26n.pt")


def ensure_gesture_model():
    if os.path.exists(GESTURE_MODEL_PATH):
        return GESTURE_MODEL_PATH

    os.makedirs(MODEL_DIR, exist_ok=True)
    temporary_path = GESTURE_MODEL_PATH + ".download"
    try:
        urllib.request.urlretrieve(GESTURE_MODEL_URL, temporary_path)
        os.replace(temporary_path, GESTURE_MODEL_PATH)
    except Exception as error:
        raise RuntimeError(
            "The hand-gesture model could not be downloaded. Add "
            "models/gesture_recognizer.task to the project manually or "
            "check network access."
        ) from error
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)
    return GESTURE_MODEL_PATH


def load_gesture_recognizer():
    import mediapipe as mp

    model_path = ensure_gesture_model()
    options = mp.tasks.vision.GestureRecognizerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    return mp.tasks.vision.GestureRecognizer.create_from_options(options)


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


class LiveExpressionAnalyser:
    """Create tutor-style live observations without building a frame queue."""

    def __init__(
        self,
        person_model,
        expression_model_data,
        confidence_model_data,
        gesture_recognizer,
        sample_interval=2.0,
        analysis_width=480
    ):
        self.person_model = person_model
        self.expression_model_data = expression_model_data
        self.confidence_model_data = confidence_model_data
        self.gesture_recognizer = gesture_recognizer
        self.sample_interval = sample_interval
        self.analysis_width = analysis_width
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        with getattr(self, "lock", threading.Lock()):
            self.total_samples = 0
            self.person_samples = 0
            self.face_samples = 0
            self.expression_counts = {}
            self.confidence_counts = {}
            self.gesture_counts = {}
            self.latest_person = "Not detected"
            self.latest_person_score = 0.0
            self.latest_expression = ""
            self.latest_confidence = ""
            self.latest_gesture = "No hand detected"
            self.latest_person_box = None
            self.person_error = ""
            self.expression_error = ""
            self.confidence_error = ""
            self.gesture_error = ""
            self.last_sample_time = 0.0
            self.analysis_running = False

    def _prepare_analysis_frame(self, image):
        image_height, image_width = image.shape[:2]
        if image_width <= self.analysis_width:
            return image

        scale = self.analysis_width / image_width
        resized_height = max(int(image_height * scale), 1)
        return cv2.resize(
            image,
            (self.analysis_width, resized_height),
            interpolation=cv2.INTER_AREA
        )

    def _record_prediction(self, predictions, count_name, latest_name):
        if not predictions:
            return

        top_label = predictions[0]["label"]
        with self.lock:
            label_counts = getattr(self, count_name)
            label_counts[top_label] = label_counts.get(top_label, 0) + 1
            setattr(self, latest_name, top_label)

    def _analyse_person(self, analysis_frame):
        if self.person_model is None:
            return

        try:
            results = self.person_model.predict(
                analysis_frame,
                imgsz=320,
                classes=[0],
                conf=0.35,
                max_det=1,
                verbose=False
            )
            result = results[0]
            boxes = result.boxes
            person_detected = boxes is not None and len(boxes) > 0

            person_box = None
            person_score = 0.0
            if person_detected:
                coordinates = boxes.xyxy[0].cpu().numpy().tolist()
                if getattr(boxes, "conf", None) is not None:
                    person_score = float(boxes.conf[0].cpu().item())
                height, width = analysis_frame.shape[:2]
                person_box = (
                    coordinates[0] / width,
                    coordinates[1] / height,
                    coordinates[2] / width,
                    coordinates[3] / height
                )

            with self.lock:
                self.latest_person = (
                    "Detected" if person_detected else "Not detected"
                )
                self.latest_person_score = person_score
                self.latest_person_box = person_box
                if person_detected:
                    self.person_samples += 1
        except Exception as error:
            with self.lock:
                self.person_error = str(error)

    def _analyse_gesture(self, analysis_frame):
        if self.gesture_recognizer is None:
            return

        try:
            import mediapipe as mp

            rgb_frame = cv2.cvtColor(analysis_frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=np.ascontiguousarray(rgb_frame)
            )
            result = self.gesture_recognizer.recognize(mp_image)

            hand_detected = bool(result.hand_landmarks)
            gesture_label = "No hand detected"
            if hand_detected:
                gesture_label = "Unrecognized hand gesture"
                if result.gestures and result.gestures[0]:
                    category = result.gestures[0][0]
                    category_name = category.category_name or ""
                    if category_name and category_name != "None":
                        gesture_label = category_name.replace("_", " ")

            with self.lock:
                self.latest_gesture = gesture_label
                if hand_detected:
                    self.gesture_counts[gesture_label] = (
                        self.gesture_counts.get(gesture_label, 0) + 1
                    )
        except Exception as error:
            with self.lock:
                self.gesture_error = str(error)

    def _analyse_sample(self, image):
        analysis_frame = self._prepare_analysis_frame(image)
        self._analyse_person(analysis_frame)
        self._analyse_gesture(analysis_frame)
        face_crop = extract_largest_face(
            analysis_frame,
            load_face_detector()
        )

        with self.lock:
            self.total_samples += 1

        if face_crop is None:
            return

        prepared_faces = prepare_face_batch([face_crop])

        with self.lock:
            self.face_samples += 1

        try:
            expression_batch = predict_face_batch(
                prepared_faces,
                self.expression_model_data
            )
            expression_predictions = (
                expression_batch[0] if expression_batch else []
            )
            self._record_prediction(
                expression_predictions,
                "expression_counts",
                "latest_expression"
            )
        except Exception as error:
            with self.lock:
                self.expression_error = str(error)

        try:
            confidence_batch = predict_face_batch(
                prepared_faces,
                self.confidence_model_data
            )
            confidence_predictions = (
                confidence_batch[0] if confidence_batch else []
            )
            self._record_prediction(
                confidence_predictions,
                "confidence_counts",
                "latest_confidence"
            )
        except Exception as error:
            with self.lock:
                self.confidence_error = str(error)

    def process_frame(self, frame):
        """Receive one browser-camera frame and return the displayed frame."""
        import av

        image = frame.to_ndarray(format="bgr24")
        current_time = time.monotonic()

        with self.lock:
            should_analyse = (
                not self.analysis_running
                and current_time - self.last_sample_time
                >= self.sample_interval
            )
            if should_analyse:
                self.analysis_running = True
                self.last_sample_time = current_time

        if should_analyse:
            try:
                self._analyse_sample(image)
            finally:
                with self.lock:
                    self.analysis_running = False

        with self.lock:
            latest_person = self.latest_person
            latest_person_score = self.latest_person_score
            latest_expression = self.latest_expression
            latest_confidence = self.latest_confidence
            latest_gesture = self.latest_gesture
            latest_person_box = self.latest_person_box

        image_height, image_width = image.shape[:2]
        if latest_person_box:
            x1 = int(latest_person_box[0] * image_width)
            y1 = int(latest_person_box[1] * image_height)
            x2 = int(latest_person_box[2] * image_width)
            y2 = int(latest_person_box[3] * image_height)
            cv2.rectangle(image, (x1, y1), (x2, y2), (60, 110, 255), 2)
            cv2.putText(
                image,
                "person " + f"{latest_person_score:.2f}",
                (x1, max(y1 - 8, 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (60, 110, 255),
                2,
                cv2.LINE_AA
            )

        overlay_height = min(112, image_height)
        overlay = image.copy()
        cv2.rectangle(
            overlay,
            (0, 0),
            (image_width, overlay_height),
            (18, 28, 39),
            -1
        )
        cv2.addWeighted(overlay, 0.82, image, 0.18, 0, image)

        expression_text = (
            latest_expression.title() if latest_expression else "Waiting"
        )
        confidence_text = (
            latest_confidence.title() if latest_confidence else "Waiting"
        )
        gesture_text = latest_gesture or "No hand detected"

        person_status_text = "YOLO person: " + latest_person
        if latest_person == "Detected":
            person_status_text += " " + f"({latest_person_score:.0%})"

        cv2.putText(
            image,
            person_status_text,
            (16, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (50, 230, 255),
            2,
            cv2.LINE_AA
        )
        cv2.putText(
            image,
            "Expression: " + expression_text + " / " + confidence_text,
            (16, 64),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.63,
            (245, 245, 245),
            2,
            cv2.LINE_AA
        )
        cv2.putText(
            image,
            "Gesture: " + gesture_text.title(),
            (16, 98),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.63,
            (120, 240, 150),
            2,
            cv2.LINE_AA
        )

        return av.VideoFrame.from_ndarray(image, format="bgr24")

    def recv(self, frame):
        """WebRTC adapter required by the live video processor."""
        return self.process_frame(frame)

    def get_results(self):
        with self.lock:
            total_samples = self.total_samples
            person_samples = self.person_samples
            face_samples = self.face_samples
            expression_counts = dict(self.expression_counts)
            confidence_counts = dict(self.confidence_counts)
            gesture_counts = dict(self.gesture_counts)
            person_error = self.person_error
            expression_error = self.expression_error
            confidence_error = self.confidence_error
            gesture_error = self.gesture_error

        def create_distribution(label_counts):
            classified_samples = sum(label_counts.values())
            if not classified_samples:
                return []
            results = [
                {
                    "label": label,
                    "score": count / classified_samples,
                    "observations": count
                }
                for label, count in label_counts.items()
            ]
            return sorted(
                results,
                key=lambda item: item["score"],
                reverse=True
            )

        expressions = create_distribution(expression_counts)
        confidence_outputs = create_distribution(confidence_counts)
        gestures = create_distribution(gesture_counts)

        person_visibility = round(
            person_samples / total_samples * 100
        ) if total_samples else 0

        face_visibility = round(
            face_samples / total_samples * 100
        ) if total_samples else 0

        if total_samples < 3:
            evaluation_feedback = (
                "Keep the camera running for longer before creating the "
                "evaluation."
            )
        elif face_visibility < 60:
            evaluation_feedback = (
                "Facial analysis was limited because a clear frontal face was "
                "not visible in many sampled moments. Improve lighting and "
                "camera position."
            )
        else:
            evaluation_feedback = (
                "The face was visible in enough sampled moments for a facial-"
                "expression distribution."
            )

        return {
            "source": "live_camera",
            "analysed_frames": total_samples,
            "person_visibility": person_visibility,
            "person_error": person_error,
            "face_visibility": face_visibility,
            "face_error": "" if face_samples else (
                "No clear frontal face was detected during the live session."
            ),
            "main_expression": (
                expressions[0]["label"]
                if expressions else "Not available"
            ),
            "expressions": expressions,
            "expression_error": expression_error,
            "main_confidence_output": (
                confidence_outputs[0]["label"]
                if confidence_outputs else "Not available"
            ),
            "confidence_outputs": confidence_outputs,
            "confidence_error": confidence_error,
            "main_gesture": (
                gestures[0]["label"] if gestures else "No hand detected"
            ),
            "gestures": gestures,
            "gesture_error": gesture_error,
            "evaluation_feedback": evaluation_feedback,
        }

    def release_models(self):
        """Release per-camera references before loading the speech model."""
        with self.lock:
            gesture_recognizer = self.gesture_recognizer
            self.person_model = None
            self.expression_model_data = (None, [])
            self.confidence_model_data = (None, [])
            self.gesture_recognizer = None

        if gesture_recognizer is not None:
            close_method = getattr(gesture_recognizer, "close", None)
            if callable(close_method):
                close_method()


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


