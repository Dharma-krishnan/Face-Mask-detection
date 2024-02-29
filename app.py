import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from streamlit_webrtc import webrtc_streamer
import av

# Load models (ensure correct file paths)
mask_model = load_model("mask_detection_model.h5")
age_model = load_model("age.h5")  # Use `load_model` directly for .h5 files
gender_model = load_model("gender.h5")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Corrected preprocessing for age-gender models
def preprocess_face(face):
    face = cv2.resize(face, (64, 64))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = face.astype("float") / 255.0
    face = np.expand_dims(face, axis=0)
    return face

# Define video frame callback
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))

    for (x, y, w, h) in faces:
        face = img[y:y+h, x:x+w]

        # Mask detection
        mask_prediction = mask_model.predict(preprocess_input(img_to_array(cv2.resize(face, (128, 128)))))
        label = "Mask" if mask_prediction[0][0] > 0.5 else "No Mask"  # Adjust threshold if needed
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        cv2.putText(img, f'Mask: {label}', (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Age-gender prediction
        face = preprocess_face(face)
        age_prediction = age_model.predict(face)[0]
        age = age_classes[np.argmax(age_prediction)]
        gender_prediction = gender_model.predict(face)[0]
        gender = "Male" if np.argmax(gender_prediction) == 0 else "Female"
        cv2.putText(img, f'Age: {age}', (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(img, f'Gender: {gender}', (x, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

    return av.VideoFrame.from_ndarray(img, format="bgr24")

# Streamlit app
st.title("Face Mask Detection")
webrtc_streamer(key="example", video_frame_callback=video_frame_callback, async_processing=True, rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
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
