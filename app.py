import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from deepface import DeepFace
from yolov5 import detect
from streamlit_webrtc import webrtc_streamer
import av

# Load the pre-trained models
mask_model = load_model("last.pt")  # YOLOv5 model for mask detection
age_model = load_model("age.h5")     # Pre-trained model for age prediction
gender_model = load_model("gender.h5") # Pre-trained model for gender prediction

# Load the pre-trained face detection model for YOLOv5
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function for real-time mask detection, age, and gender prediction
def detect_mask_age_gender(frame):
    # Perform face detection using YOLOv5
    results = detect(frame, weights="path/to/your/pretrained_model.pt")

    for result in results:
        label = result["label"]
        confidence = result["confidence"]
        box = result["box"]
        x, y, w, h = box

        # If a face is detected
        if label == "face":
            # Crop the face region
            face_roi = frame[y:y+h, x:x+w]

            # Perform mask detection using YOLOv5
            mask_label = detect_mask_in_roi(face_roi)

            # Display mask prediction
            cv2.putText(frame, f'Mask: {mask_label}', (x, y-50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if mask_label == "No Mask":  # If no mask is detected, perform age and gender prediction
                # Preprocess the face for age and gender prediction
                face_for_prediction = cv2.resize(face_roi, (224, 224))
                face_for_prediction = img_to_array(face_for_prediction)
                face_for_prediction = preprocess_input(face_for_prediction)
                face_for_prediction = np.expand_dims(face_for_prediction, axis=0)

                # Perform age prediction
                age_prediction = age_model.predict(face_for_prediction)
                age_class = np.argmax(age_prediction)
                age_classes = ['(0-2)', '(3-9)', '(10-19)', '(20-29)', '(30-39)', '(40-49)', '(50-59)', '(60-69)', '(70+)']
                predicted_age = age_classes[age_class]

                # Perform gender prediction
                gender_prediction = gender_model.predict(face_for_prediction)
                gender = "Male" if gender_prediction[0][0] > 0.5 else "Female"

                # Display age and gender prediction
                cv2.putText(frame, f'Age: {predicted_age}', (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f'Gender: {gender}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame

# Placeholder function for mask detection within a region of interest (ROI)
def detect_mask_in_roi(face_roi):
    # Convert the face ROI to grayscale for better processing
    gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

    # Apply face mask detection logic (example logic)
    # For demonstration purposes, let's assume if there's a grayscale pixel intensity above a certain threshold, we consider it as a mask
    # You can replace this with your actual mask detection model or algorithm
    mask_threshold = 100  # Adjust this threshold as needed
    avg_intensity = np.mean(gray_face)

    if avg_intensity > mask_threshold:
        return "Mask"
    else:
        return "No Mask"

# Streamlit web app
st.title("Face Mask Detection")

# Start the webcam stream
webrtc_streamer(
    key="example",
    video_processor_factory=detect_mask_age_gender,
    async_processing=True,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)