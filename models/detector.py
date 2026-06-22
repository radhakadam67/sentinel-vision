import numpy as np
from ultralytics import YOLO
from configs.settings import YOLO_MODEL_PATH, YOLO_CONFIDENCE, YOLO_CLASSES


class ObjectDetector:
    def __init__(self):
        self.model = YOLO(YOLO_MODEL_PATH)
        self.confidence = YOLO_CONFIDENCE
        self.classes = YOLO_CLASSES
        print(f"[Detector] YOLOv8 loaded from {YOLO_MODEL_PATH}")

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Run detection on a single frame.

        Args:
            frame: BGR image array (from OpenCV)

        Returns:
            List of detections, each as:
            {
                "bbox": [x1, y1, x2, y2],
                "confidence": float,
                "class_id": int
            }
        """
        results = self.model(
            frame,
            conf=self.confidence,
            classes=self.classes,
            verbose=False
        )[0]

        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append({
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "confidence": float(box.conf[0]),
                "class_id": int(box.cls[0])
            })

        return detections
