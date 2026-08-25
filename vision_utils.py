import os
import tempfile

import cv2
import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array, load_img
from ultralytics import YOLO


def load_pose_model():
    return YOLO("yolo26n-pose.pt")


def load_face_model():
    model_path = "models/face_expression_model.keras"
    class_path = "models/class_names.json"
    if not os.path.exists(model_path) or not os.path.exists(class_path):
        return None, []

    face_model = load_model(model_path)
    with open(class_path, "r", encoding="utf-8") as class_file:
        class_names = json.load(class_file)
    return face_model, class_names


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


def save_upload(uploaded_file):
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as saved_file:
        saved_file.write(uploaded_file.getbuffer())
        return saved_file.name


def analyse_image(image_path, pose_model, face_model_data):
    pose_results = pose_model(image_path, verbose=False)
    pose_result = pose_results[0]
    people = len(pose_result.boxes) if pose_result.boxes is not None else 0
    keypoints_visible = 0

    if pose_result.keypoints is not None and pose_result.keypoints.conf is not None:
        keypoints_visible = int((pose_result.keypoints.conf.cpu().numpy() > 0.5).sum())

    face_model, class_names = face_model_data
    expressions = predict_expression(image_path, face_model, class_names)[:5]
    return {
        "people": people,
        "keypoints_visible": keypoints_visible,
        "expressions": expressions
    }


def analyse_video(video_path, pose_model, face_model_data, sample_every=45, max_frames=40):
    capture = cv2.VideoCapture(video_path)
    frame_number = 0
    analysed_frames = 0
    person_frames = 0
    visible_keypoints = []
    expression_totals = {}

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
                for item in frame_expressions:
                    label = item["label"]
                    expression_totals[label] = expression_totals.get(label, 0) + item["score"]
            except Exception:
                pass
            finally:
                if os.path.exists(frame_path):
                    os.remove(frame_path)

            analysed_frames += 1
        frame_number += 1

    capture.release()
    main_expression = "Not available"
    if expression_totals:
        main_expression = max(expression_totals, key=expression_totals.get)

    person_visibility = round(person_frames / analysed_frames * 100) if analysed_frames else 0
    average_keypoints = round(sum(visible_keypoints) / len(visible_keypoints), 1) if visible_keypoints else 0

    return {
        "analysed_frames": analysed_frames,
        "person_visibility": person_visibility,
        "average_keypoints": average_keypoints,
        "main_expression": main_expression
    }
