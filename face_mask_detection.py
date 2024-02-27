import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Flatten, Dense, Dropout, Input, LSTM, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, LSTM, Concatenate
from tensorflow.keras.optimizers import Adam

# Generate synthetic sequential data
seq_length = 10
seq_features = 50
num_samples = 1000

synthetic_seq_data = np.random.rand(num_samples, seq_length, seq_features)

# Initialize the base MobileNetV2 model
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(128, 128, 3))

# Construct the head of the model that will be placed on top of the base model
head_model = base_model.output
head_model = AveragePooling2D(pool_size=(4, 4))(head_model)
head_model = Flatten(name="flatten")(head_model)
head_model = Dense(128, activation="relu")(head_model)
head_model = Dropout(0.5)(head_model)

# Define LSTM layer for sequential data processing
seq_input = Input(shape=(seq_length, seq_features))
lstm_output = LSTM(units=64)(seq_input)

# Concatenate CNN and LSTM outputs
combined_input = Concatenate()([head_model, lstm_output])

# Additional dense layers for further processing
dense_output = Dense(128, activation="relu")(combined_input)
output = Dense(2, activation="softmax")(dense_output)

# Combined model
model = Model(inputs=[base_model.input, seq_input], outputs=output)

# Freeze the layers in the base model
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
opt = Adam(lr=1e-4)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# Data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

# Define the path to the data directory
data_directory = "data"

# Generate data from the directory structure
train_generator = train_datagen.flow_from_directory(
    data_directory,
    target_size=(128, 128),
    batch_size=32,
    class_mode="categorical",
    classes=["without_mask", "with_mask"]  # Order of classes matters
)

# Training the model with synthetic sequential data
model.fit(
    [train_generator, synthetic_seq_data],  # Pass both image and synthetic sequential data
    steps_per_epoch=len(train_generator),
    epochs=20
)

# Save the model
model.save("mask_detection_model.h5")