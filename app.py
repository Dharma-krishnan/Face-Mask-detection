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
import tempfile
import os
from yolov5 import detect
# Load the trained mask detection model
model = load_model("mask_detection_model.h5")

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function for real-time mask detection

def detect_faces_yolov5(frame):
    # Detect objects using YOLOv5
    results = detect(frame)
    faces = []

    # Filter detected objects to only include faces
    for detection in results.pred:
        if detection['label'] == 'face':
            x1, y1, x2, y2 = detection['box']
            faces.append((x1, y1, x2 - x1, y2 - y1))  # Convert box to (x, y, w, h) format

    return faces

def detect_mask(frame):
    # Preprocess the frame
    resized_frame = cv2.resize(frame, (128, 128))
    resized_frame = img_to_array(resized_frame)
    resized_frame = preprocess_input(resized_frame)
    resized_frame = np.expand_dims(resized_frame, axis=0)

# Perform prediction
    predictions = model.predict(resized_frame)
    return predictions

# Define the video frame callback function


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

    # Detect faces using YOLOv5
    faces = detect_faces_yolov5(img)

    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # faces = face_cascade.detectMultiScale(
    #     gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

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

# #uploading file from the user 
# def process_video(input_path, output_folder):
#     video_capture = cv2.VideoCapture(input_path)
#     frame_count = 0

#     while True:
#         ret, frame = video_capture.read()
#         if not ret:
#             break

#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = face_cascade.detectMultiScale(
#             gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
#         predictions = detect_mask(frame)
#         processed_frame = draw_rectangles(frame, faces, predictions)

#         # Save the processed frame as an image
#         output_path = os.path.join(output_folder, f"frame_{frame_count}.jpg")
#         cv2.imwrite(output_path, processed_frame)

#         frame_count += 1

#     video_capture.release()


# Upload a video file
# st.title("Upload A Video for Detection")

# # Upload a video file
# uploaded_file = st.file_uploader("Upload a video", type=["mp4"])

# if uploaded_file is not None:
#     # Save the uploaded video to a temporary file
#     with tempfile.NamedTemporaryFile(delete=False) as temp_file:
#         temp_file.write(uploaded_file.read())
#         temp_file_path = temp_file.name

#     # Create a temporary folder to store processed frames
#     temp_folder = tempfile.mkdtemp()

#     # Process the video and save frames with detections
#     process_video(temp_file_path, temp_folder)

#     # Display the processed frames as images
#     for filename in sorted(os.listdir(temp_folder)):
#         st.image(os.path.join(temp_folder, filename))

#     # Cleanup: delete temporary video file and folder
#     os.remove(temp_file_path)
#     for filename in os.listdir(temp_folder):
#         os.remove(os.path.join(temp_folder, filename))
#     os.rmdir(temp_folder)



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

import numpy as np
from tensorflow.keras.models import load_model, model_from_json
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from streamlit_webrtc import webrtc_streamer
import av

# Load the trained mask detection model
model = load_model("mask_detection_model.h5")

# Load Levi-Hassner age model architecture from the .json file
with open("age.json", "r") as json_file:
    age_model_json = json_file.read()
age_model = model_from_json(age_model_json)

# Load the pretrained age model weights from the .h5 file
age_model.load_weights("age.h5")

# Load Levi-Hassner gender model architecture from the .json file
with open("gender.json", "r") as json_file:
    gender_model_json = json_file.read()
gender_model = model_from_json(gender_model_json)

# Load the pretrained gender model weights from the .h5 file
gender_model.load_weights("gender.h5")

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function for real-time mask detection
def detect_mask(frame):
    # Preprocess the frame
    resized_frame = cv2.resize(frame, (128, 128))
    resized_frame = img_to_array(resized_frame)
    resized_frame = preprocess_input(resized_frame)
    resized_frame = np.expand_dims(resized_frame, axis=0)

    # Perform prediction
    predictions = model.predict(resized_frame)
    return predictions

# Function to predict age and gender for video
# def predict_age_gender(face):
#     face = cv2.resize(face, (64, 64))
#     face = face.astype("float") / 255.0
#     face = np.expand_dims(face, axis=0)

#     # Predict age
#     age_prediction = age_model.predict(face)[0]
#     age_classes = ['(0-2)', '(3-9)', '(10-19)', '(20-29)', '(30-39)', '(40-49)', '(50-59)', '(60-69)', '(70+)']
#     age = age_classes[np.argmax(age_prediction)]

#     # Predict gender
#     gender_prediction = gender_model.predict(face)[0]
#     gender = "Male" if np.argmax(gender_prediction) == 0 else "Female"

#     return age, gender
def predict_age_gender(face):
    # Resize the input image to match the expected input size of the age model
    resized_face = cv2.resize(face, (227, 227))
    
    # Preprocess the resized face image
    resized_face = resized_face.astype("float") / 255.0
    resized_face = np.expand_dims(resized_face, axis=0)

    # Predict age
    age_prediction = age_model.predict(resized_face)[0]
    age_classes = ['(0-2)', '(3-9)', '(10-19)', '(20-29)', '(30-39)', '(40-49)', '(50-59)', '(60-69)', '(70+)']
    age = age_classes[np.argmax(age_prediction)]

    # Predict gender
    gender_prediction = gender_model.predict(face)[0]
    gender = "Male" if np.argmax(gender_prediction) == 0 else "Female"

    return age, gender

# Function to process uploaded image
def process_image(uploaded_image):
    # Read the uploaded image
    img = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), 1)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    # Perform mask detection and age-gender prediction
    for (x, y, w, h) in faces:
        face = img[y:y+h, x:x+w]
        
        # Perform mask detection
        mask_prediction = detect_mask(face)
        label = "Mask" if np.argmax(mask_prediction) == 1 else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        
        # If no mask, predict age and gender
        if label == "No Mask":
            age, gender = predict_age_gender(face)

            # Display age and gender prediction results
            cv2.putText(img, f'Age: {age}', (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(img, f'Gender: {gender}', (x, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Display mask detection result
        cv2.putText(img, f'Mask: {label}', (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

    return img
# Streamlit web app
st.title("Upload a JPG Image")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg"])

if uploaded_file is not None:
    # Process the uploaded image
    processed_image = process_image(uploaded_file)

    # Display the processed image
    st.image(processed_image, channels="BGR")

# Define the video frame callback function
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")

    # Detect faces
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    # Perform mask detection and age-gender prediction
    for (x, y, w, h) in faces:
        face = img[y:y+h, x:x+w]
        
        # Perform mask detection
        mask_prediction = detect_mask(face)
        label = "Mask" if np.argmax(mask_prediction) == 1 else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        
        # Predict age and gender
        age, gender = predict_age_gender(face)

        # Display mask detection and age-gender prediction results
        cv2.putText(img, f'Mask: {label}', (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(img, f'Age: {age}', (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(img, f'Gender: {gender}', (x, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

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

# #uploading file from the user 
# def process_video(input_path, output_folder):
#     video_capture = cv2.VideoCapture(input_path)
#     frame_count = 0

#     while True:
#         ret, frame = video_capture.read()
#         if not ret:
#             break

#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = face_cascade.detectMultiScale(
#             gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
#         predictions = detect_mask(frame)
#         processed_frame = draw_rectangles(frame, faces, predictions)

#         # Save the processed frame as an image
#         output_path = os.path.join(output_folder, f"frame_{frame_count}.jpg")
#         cv2.imwrite(output_path, processed_frame)

#         frame_count += 1

#     video_capture.release()


# Upload a video file
# st.title("Upload A Video for Detection")

# # Upload a video file
# uploaded_file = st.file_uploader("Upload a video", type=["mp4"])

# if uploaded_file is not None:
#     # Save the uploaded video to a temporary file
#     with tempfile.NamedTemporaryFile(delete=False) as temp_file:
#         temp_file.write(uploaded_file.read())
#         temp_file_path = temp_file.name

#     # Create a temporary folder to store processed frames
#     temp_folder = tempfile.mkdtemp()

#     # Process the video and save frames with detections
#     process_video(temp_file_path, temp_folder)

#     # Display the processed frames as images
#     for filename in sorted(os.listdir(temp_folder)):
#         st.image(os.path.join(temp_folder, filename))

#     # Cleanup: delete temporary video file and folder
#     os.remove(temp_file_path)
#     for filename in os.listdir(temp_folder):
#         os.remove(os.path.join(temp_folder, filename))
#     os.rmdir(temp_folder)



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
