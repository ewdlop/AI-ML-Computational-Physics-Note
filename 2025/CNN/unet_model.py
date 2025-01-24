import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
import os

# Configuration
DATA_DIR = "path/to/your/dataset"
IMG_SIZE = (256, 256)
BATCH_SIZE = 8
EPOCHS = 50

# U-Net Architecture
def unet_model(input_size=(256, 256, 3)):
    inputs = Input(input_size)
    
    # Downsample
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Bottleneck
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)

    # Upsample
    up4 = UpSampling2D(size=(2, 2))(conv3)
    up4 = Conv2D(128, 2, activation='relu', padding='same')(up4)
    merge4 = concatenate([conv2, up4], axis=3)
    conv4 = Conv2D(128, 3, activation='relu', padding='same')(merge4)
    conv4 = Conv2D(128, 3, activation='relu', padding='same')(conv4)

    up5 = UpSampling2D(size=(2, 2))(conv4)
    up5 = Conv2D(64, 2, activation='relu', padding='same')(up5)
    merge5 = concatenate([conv1, up5], axis=3)
    conv5 = Conv2D(64, 3, activation='relu', padding='same')(merge5)
    conv5 = Conv2D(64, 3, activation='relu', padding='same')(conv5)

    # Output
    outputs = Conv2D(1, 1, activation='sigmoid')(conv5)

    model = Model(inputs=inputs, outputs=outputs)
    return model

# Data Generator
def create_generator(img_folder, mask_folder, batch_size):
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    
    img_generator = datagen.flow_from_directory(
        img_folder,
        target_size=IMG_SIZE,
        class_mode=None,
        batch_size=batch_size,
        seed=1
    )
    
    mask_generator = datagen.flow_from_directory(
        mask_folder,
        target_size=IMG_SIZE,
        class_mode=None,
        batch_size=batch_size,
        seed=1,
        color_mode='grayscale'
    )
    
    return zip(img_generator, mask_generator)

# Dataset Preparation (Example structure)
"""
dataset/
├── images/
│   ├── class1/
│   │   ├── img1.jpg
│   │   └── img2.jpg
└── masks/
    ├── class1/
        ├── mask1.png
        └── mask2.png
"""

# Create and compile model
model = unet_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model
train_generator = create_generator(
    os.path.join(DATA_DIR, "images"),
    os.path.join(DATA_DIR, "masks"),
    BATCH_SIZE
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5),
    tf.keras.callbacks.ModelCheckpoint(
        'face_segmentation_cnn.h5',
        save_best_only=True
    )
]

history = model.fit(
    train_generator,
    steps_per_epoch=100,
    epochs=EPOCHS,
    callbacks=callbacks
)

print("Model saved as face_segmentation_cnn.h5")
