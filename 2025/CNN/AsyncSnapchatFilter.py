import asyncio
import cv2
import numpy as np
import mediapipe as mp
from vidgear.gears.asyncio import WebGear
from tensorflow.keras.models import load_model
from concurrent.futures import ThreadPoolExecutor

# Configuration
MODEL_PATH = "face_segmentation_cnn.h5"
FILTER_IMAGE = "glasses_filter.png"

class AsyncSnapchatFilter:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.loop = asyncio.get_event_loop()
        
        # Load CNN model async
        self.model = self.loop.run_in_executor(
            self.executor, 
            load_model, MODEL_PATH
        )
        
        # MediaPipe setup
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            min_detection_confidence=0.5
        )
        
        # Load filter image
        self.filter_img = cv2.imread(FILTER_IMAGE, cv2.IMREAD_UNCHANGED)

    async def async_cnn_predict(self, frame):
        # Run CNN inference in executor
        return await self.loop.run_in_executor(
            self.executor,
            self._blocking_cnn_predict,
            frame
        )

    def _blocking_cnn_predict(self, frame):
        # Actual CNN prediction (blocking)
        resized = cv2.resize(frame, (256, 256))
        mask = self.model.predict(np.expand_dims(resized/255.0, axis=0))[0]
        return cv2.resize(mask.squeeze(), (frame.shape[1], frame.shape[0]))

    async def async_face_detection(self, frame):
        # Run MediaPipe in executor
        return await self.loop.run_in_executor(
            self.executor,
            self._blocking_face_detection,
            frame
        )

    def _blocking_face_detection(self, frame):
        # Actual face detection (blocking)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.face_detection.process(rgb)

    async def process_frame(self, frame):
        # Parallel face detection and CNN processing
        detection_task = self.async_face_detection(frame)
        cnn_task = self.async_cnn_predict(frame)
        results = await asyncio.gather(detection_task, cnn_task)
        
        detection_result, mask = results
        return self.apply_filter(frame, detection_result, mask)

    def apply_filter(self, frame, detection_result, mask):
        if detection_result.detections:
            for detection in detection_result.detections:
                bbox = detection.location_data.relative_bounding_box
                ih, iw = frame.shape[:2]
                x, y = int(bbox.xmin * iw), int(bbox.ymin * ih)
                w, h = int(bbox.width * iw), int(bbox.height * ih)
                
                # Apply CNN mask
                alpha = mask[y:y+h, x:x+w, np.newaxis]
                overlay = cv2.resize(self.filter_img, (w, h))
                
                # Blend overlay
                blended = (1 - alpha) * frame[y:y+h, x:x+w] + alpha * overlay[..., :3]
                frame[y:y+h, x:x+w] = blended.astype(np.uint8)
        return frame

async def video_stream():
    filter_processor = AsyncSnapchatFilter()
    webgear = WebGear(source=0, logging=True)
    
    async with webgear.stream() as stream:
        while True:
            frame = await stream.read()
            if frame is None:
                break
                
            processed = await filter_processor.process_frame(frame)
            await stream.send(processed)
            
            # For local display (optional)
            cv2.imshow('Async Snapchat Filter', processed)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cv2.destroyAllWindows()
    await webgear.close()

if __name__ == "__main__":
    try:
        asyncio.run(video_stream())
    except KeyboardInterrupt:
        print("Filter application stopped")
