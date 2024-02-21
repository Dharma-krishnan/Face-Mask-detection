import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from deepface import DeepFace
from streamlit_webrtc import webrtc_streamer
import av
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the trained mask detection model
model = load_model("mask_detection_model.h5")

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function for real-time mask detection


# def detect_mask(frame):
# Preprocess the frame
# resized_frame = cv2.resize(frame, (128, 128))
# resized_frame = img_to_array(resized_frame)
# resized_frame = preprocess_input(resized_frame)
# resized_frame = np.expand_dims(resized_frame, axis=0)

# # Perform prediction
# predictions = model.predict(resized_frame)
# return predictions

# Define the video frame callback function

@st.cache
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    resized_frame = cv2.resize(img, (128, 128))  # Corrected resize function
    resized_frame = img_to_array(resized_frame)
    resized_frame = preprocess_input(resized_frame)
    resized_frame = np.expand_dims(resized_frame, axis=0)

    # Perform prediction
    predictions = model.predict(resized_frame)
    # Perform mask detection
    label = "Mask" if np.argmax(predictions) == 1 else "No Mask"
    color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    # Display the frame with the label
    for (x, y, w, h) in faces:
        # If no mask is detected, estimate age and gender
        if label == "No Mask":
            results = DeepFace.analyze(
                img[y:y+h, x:x+w], actions=['age', 'gender'], enforce_detection=False)
            results = results[0]
            age = results['age']
            gender = results['dominant_gender']
            cv2.putText(img, f'Age: {age:.1f} years', (x, y - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(img, f'Gender: {gender}', (x, y - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Display mask detection result
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    return av.VideoFrame.from_ndarray(img, format="bgr24")


# Streamlit web app
st.title("Face Mask Detection")

# Start the webcam stream
webrtc_streamer(
    key="example",
    video_frame_callback=video_frame_callback,  # Pass the function itself
    async_processing=True,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)

st.title("Upload a video")
uploaded_file = st.file_uploader("Upload a video", type=["mp4"])

if uploaded_file is not None:
    video_bytes = uploaded_file.read()
    video_np_array = np.frombuffer(video_bytes, np.uint8)
    video_frame_generator = cv2.VideoCapture(video_np_array)

    while True:
        ret, frame = video_frame_generator.read()
        if not ret:
            break

        av_frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
        processed_frame = video_frame_callback(av_frame)

        st.image(processed_frame.to_ndarray(format="bgr24"), channels="BGR")


# Define the path to the validation data directory
# validation_data_directory = "data"

# # Load validation data
# validation_datagen = ImageDataGenerator(
#     rescale=1.0/255.0
# )

# validation_generator = validation_datagen.flow_from_directory(
#     validation_data_directory,
#     target_size=(128, 128),
#     batch_size=32,
#     class_mode="categorical",
#     classes=["without_mask", "with_mask"]
# )

# Evaluate model
# loss, accuracy = model.evaluate(validation_generator)

# # Streamlit web app
# st.title("Mask Detection Model Evaluation")
# st.write("Validation Loss:", loss)  #237/237 [==============================] - 26s 107ms/step - loss: 0.0137 - accuracy: 0.996
# st.write("Validation Accuracy:", accuracy)



footer="""<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: white;
color: black;
text-align: center;
}
</style>
<div class="footer">
<p>Developed by <a style='display: block; text-align: center;' href="https://dharmakrishnan.me" target="_blank">Dharma Krishnan R</a></p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)
