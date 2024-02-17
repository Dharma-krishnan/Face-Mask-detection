import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from deepface import DeepFace
from streamlit_webrtc import webrtc_streamer
import av

# Load the trained mask detection model
model = load_model("mask_detection_model.h5")

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define the video frame callback function
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")

    # Perform face detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    # Iterate over detected faces
    for (x, y, w, h) in faces:
        face_img = img[y:y+h, x:x+w]

        # Preprocess the face image for mask detection
        resized_face = cv2.resize(face_img, (128, 128))
        resized_face = img_to_array(resized_face)
        resized_face = preprocess_input(resized_face)
        resized_face = np.expand_dims(resized_face, axis=0)

        # Perform mask detection
        predictions = model.predict(resized_face)
        label = "Mask" if np.argmax(predictions) == 1 else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        # Display the label and bounding box on the original frame
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    return av.VideoFrame.from_ndarray(img, format="bgr24")


# Streamlit web app
st.title("Real-time Face Mask Detection")

# Start the webcam stream with WebRTC
webrtc_streamer(key="example", video_processor_factory=video_frame_callback)
