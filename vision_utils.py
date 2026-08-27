import os
import tempfile

import cv2
import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array, load_img
from ultralytics import YOLO


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(PROJECT_DIR, "models")
FACE_MODEL_PATH = os.path.join(MODEL_DIR, "face_expression_model.keras")
CLASS_NAMES_PATH = os.path.join(MODEL_DIR, "class_names.json")
GESTURE_MODEL_PATH = os.path.join(MODEL_DIR, "gesture_recognizer.task")

GESTURE_LABELS = {
    "Closed_Fist": "Closed fist",
    "Open_Palm": "Open palm",
    "Pointing_Up": "Pointing up",
    "Thumb_Down": "Thumbs down",
    "Thumb_Up": "Thumbs up",
    "Victory": "Peace sign",
    "ILoveYou": "I love you sign"
}


def load_pose_model():
    return YOLO("yolo26n-pose.pt")


def get_face_model_status():
    missing_files = []
    if not os.path.exists(FACE_MODEL_PATH):
        missing_files.append("models/face_expression_model.keras")
    if not os.path.exists(CLASS_NAMES_PATH):
        missing_files.append("models/class_names.json")

    if missing_files:
        return {
            "ready": False,
            "message": "Facial-expression output is unavailable. Missing: " + ", ".join(missing_files)
        }
    return {"ready": True, "message": "Facial-expression model files are available."}


def load_face_model():
    status = get_face_model_status()
    if not status["ready"]:
        return None, []

    face_model = load_model(FACE_MODEL_PATH)
    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as class_file:
        class_names = json.load(class_file)

    output_classes = int(face_model.output_shape[-1])
    if len(class_names) != output_classes:
        raise ValueError(
            f"The model has {output_classes} outputs but class_names.json has {len(class_names)} labels."
        )
    return face_model, class_names


def load_gesture_recognizer():
    if not os.path.exists(GESTURE_MODEL_PATH):
        return None, "Missing models/gesture_recognizer.task"

    try:
        import mediapipe as mp

        options = mp.tasks.vision.GestureRecognizerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=GESTURE_MODEL_PATH
            ),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        recognizer = mp.tasks.vision.GestureRecognizer.create_from_options(
            options
        )
        return recognizer, ""
    except Exception as error:
        return None, str(error)


def predict_expression(image_path, face_model, class_names):
    if face_model is None:
        return []

    image = load_img(image_path, target_size=(128, 128))
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    prediction = face_model.predict(image_array, verbose=0)[0]

    results = []
    for class_name, score in zip(class_names, prediction):
        results.append({"label": class_name, "score": float(score)})
    results = sorted(results, key=lambda item: item["score"], reverse=True)
    return results


def predict_gestures(frame, gesture_recognizer):
    if gesture_recognizer is None:
        return []

    import mediapipe as mp

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame = np.ascontiguousarray(rgb_frame)
    media_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )
    recognition = gesture_recognizer.recognize(media_image)

    results = []
    for hand_gestures in recognition.gestures:
        if not hand_gestures:
            continue

        top_gesture = hand_gestures[0]
        category_name = top_gesture.category_name
        if category_name == "None":
            continue

        results.append({
            "label": GESTURE_LABELS.get(
                category_name,
                category_name.replace("_", " ").title()
            ),
            "score": float(top_gesture.score)
        })

    return results


def save_upload(uploaded_file):
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as saved_file:
        saved_file.write(uploaded_file.getbuffer())
        return saved_file.name


def analyse_image(
    image_path,
    pose_model,
    face_model_data,
    gesture_model_data
):
    pose_results = pose_model(image_path, verbose=False)
    pose_result = pose_results[0]
    people = len(pose_result.boxes) if pose_result.boxes is not None else 0
    keypoints_visible = 0

    if pose_result.keypoints is not None and pose_result.keypoints.conf is not None:
        keypoints_visible = int((pose_result.keypoints.conf.cpu().numpy() > 0.5).sum())

    face_model, class_names = face_model_data
    expressions = predict_expression(image_path, face_model, class_names)[:5]
    gesture_recognizer, gesture_error = gesture_model_data
    image = cv2.imread(image_path)
    gestures = []
    if image is not None:
        try:
            gestures = predict_gestures(image, gesture_recognizer)
        except Exception as error:
            gesture_error = str(error)

    return {
        "people": people,
        "keypoints_visible": keypoints_visible,
        "expressions": expressions,
        "main_expression": expressions[0]["label"] if expressions else "Not available",
        "gestures": gestures,
        "main_gesture": gestures[0]["label"] if gestures else "Not detected",
        "gesture_error": gesture_error
    }


def analyse_video(
    video_path,
    pose_model,
    face_model_data,
    gesture_model_data,
    sample_every=45,
    max_frames=40
):
    capture = cv2.VideoCapture(video_path)
    frame_number = 0
    analysed_frames = 0
    person_frames = 0
    visible_keypoints = []
    expression_counts = {}
    expression_frames = 0
    expression_error = ""
    gesture_counts = {}
    gesture_observations = 0
    gesture_recognizer, gesture_error = gesture_model_data

    while capture.isOpened() and analysed_frames < max_frames:
        success, frame = capture.read()
        if not success:
            break

        if frame_number % sample_every == 0:
            pose_result = pose_model(frame, verbose=False)[0]
            people = len(pose_result.boxes) if pose_result.boxes is not None else 0
            if people > 0:
                person_frames += 1

            if pose_result.keypoints is not None and pose_result.keypoints.conf is not None:
                visible = int((pose_result.keypoints.conf.cpu().numpy() > 0.5).sum())
                visible_keypoints.append(visible)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as frame_file:
                frame_path = frame_file.name
            cv2.imwrite(frame_path, frame)
            try:
                face_model, class_names = face_model_data
                frame_expressions = predict_expression(frame_path, face_model, class_names)[:5]
                if frame_expressions:
                    expression_frames += 1
                    main_label = frame_expressions[0]["label"]
                    expression_counts[main_label] = (
                        expression_counts.get(main_label, 0) + 1
                    )
            except Exception as error:
                expression_error = str(error)
            finally:
                if os.path.exists(frame_path):
                    os.remove(frame_path)

            try:
                frame_gestures = predict_gestures(
                    frame,
                    gesture_recognizer
                )
                for item in frame_gestures:
                    label = item["label"]
                    gesture_counts[label] = gesture_counts.get(label, 0) + 1
                    gesture_observations += 1
            except Exception as error:
                gesture_error = str(error)

            analysed_frames += 1
        frame_number += 1

    capture.release()
    main_expression = "Not available"
    if expression_counts:
        main_expression = max(expression_counts, key=expression_counts.get)

    expressions = []
    if expression_counts and expression_frames:
        expressions = [
            {
                "label": label,
                "score": count / expression_frames,
                "observations": count
            }
            for label, count in expression_counts.items()
        ]
        expressions = sorted(
            expressions,
            key=lambda item: item["score"],
            reverse=True
        )

    gestures = []
    if gesture_counts and gesture_observations:
        gestures = [
            {
                "label": label,
                "score": count / gesture_observations,
                "observations": count
            }
            for label, count in gesture_counts.items()
        ]
        gestures = sorted(
            gestures,
            key=lambda item: item["score"],
            reverse=True
        )

    main_gesture = "Not detected"
    if gesture_counts:
        main_gesture = max(gesture_counts, key=gesture_counts.get)

    person_visibility = round(person_frames / analysed_frames * 100) if analysed_frames else 0
    average_keypoints = round(sum(visible_keypoints) / len(visible_keypoints), 1) if visible_keypoints else 0

    return {
        "analysed_frames": analysed_frames,
        "person_visibility": person_visibility,
        "average_keypoints": average_keypoints,
        "main_expression": main_expression,
        "expressions": expressions,
        "expression_error": expression_error,
        "main_gesture": main_gesture,
        "gestures": gestures,
        "gesture_error": gesture_error
    }
