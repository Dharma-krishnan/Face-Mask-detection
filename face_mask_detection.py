import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the pre-trained model
model = load_model("mask_detection_model.h5")

class FaceMaskTransformer(VideoTransformerBase):
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    def transform(self, frame):
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Process each face
        for (x, y, w, h) in faces:
            # Extract the face ROI
            face_roi = frame[y:y+h, x:x+w]

            # Resize the face ROI to 128x128
            face_roi = cv2.resize(face_roi, (128, 128))

            # Normalize the face ROI
            face_roi = face_roi.astype("float32") / 255.0

            # Expand dimensions of the face ROI to match the model input shape
            face_roi = np.expand_dims(face_roi, axis=0)

            # Make a prediction using the pre-trained model
            prediction = model.predict(face_roi)

            # Get the predicted label
            label = np.argmax(prediction)

            # Get the color for the label
            if label == 0:
                color = (0, 0, 255)  # Red for "without
