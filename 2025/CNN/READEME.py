#!/usr/bin/env python3
#!pip install tensorflow opencv-python mediapipe numpy

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# CNN Model Architecture for Face Segmentation
def build_cnn_model(input_shape=(256, 256, 3)):
    model = tf.keras.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.UpSampling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.UpSampling2D((2,2)),
        layers.Conv2D(1, (3,3), activation='sigmoid', padding='same')
    ])
    return model

import mediapipe as mp

def apply_snapchat_filter():
    cap = cv2.VideoCapture(0)
    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue
            
            # Detect faces with MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb_frame)
            
            if results.detections:
                for detection in results.detections:
                    # Extract face region
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    x, y = int(bboxC.xmin * iw), int(bboxC.ymin * ih)
                    w, h = int(bboxC.width * iw), int(bboxC.height * ih)
                    
                    # Preprocess for CNN
                    face_roi = frame[y:y+h, x:x+w]
                    resized_face = cv2.resize(face_roi, (256, 256))
                    mask = model.predict(np.expand_dims(resized_face/255.0, axis=0))[0]
                    
                    # Apply filter (e.g., virtual glasses)
                    mask = cv2.resize(mask.squeeze(), (w, h))
                    filter_overlay = cv2.imread("glasses_filter.png", cv2.IMREAD_UNCHANGED)
                    filter_overlay = cv2.resize(filter_overlay, (w, h))
                    
                    # Blend filter with face
                    alpha = mask[..., None]
                    blended = (1 - alpha) * face_roi + alpha * filter_overlay[..., :3]
                    frame[y:y+h, x:x+w] = blended.astype(np.uint8)

            cv2.imshow('Snapchat-like Filter', frame)
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()



def main():
    model = build_cnn_model()
    model.compile(optimizer='adam', loss='binary_crossentropy')    
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    apply_snapchat_filter()
  
if __name__=="__main__":
    main()
