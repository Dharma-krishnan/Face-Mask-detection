import cv2
import numpy as np
import tensorflow as tf
from deepface import DeepFace
from tensorflow.keras.models import load_model
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, VideoTransformerBase

# Load the model
model = load_model("mask_detection_model.h5")

# Initialize the face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define a custom VideoTransformer
class FaceMaskTransformer(VideoTransformerBase):
    def transform(self, frame):
        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Perform face detection
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

        for (x, y, w, h) in faces:
            face = frame[y:y + h, x:x + w]

            # Perform mask detection
            face = cv2.resize(face, (128, 128))
            face = np.expand_dims(face, axis=0)
            face = np.expand_dims(face, axis=3)
            face = tf.convert_to_tensor(face, dtype=tf.float32)
            predictions = model.predict(face)
            label = "Mask" if np.argmax(predictions) == 1 else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            # If no mask is detected, estimate age and gender
            if label == "No Mask":
                results = DeepFace.analyze(face, actions=['age', 'gender'], enforce_detection=False)
                results = results[0]
                age = int(results['age'])
                gender = results['dominant_gender']

                # Display age and gender estimation
                cv2.putText(frame, f'Age: {age} years', (x, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(frame, f'Gender: {gender}', (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Display mask detection result
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        return frame

# Streamlit WebRTC configuration
config = RTCConfiguration({
  "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

# Streamlit WebRTC app
st.title("Real-time Face Mask Detection")
webrtc_streamer(key="example", rtc_configuration=config, video_transformer_factory=FaceMaskTransformer)
