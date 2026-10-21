import cv2
import numpy as np
from ultralytics import YOLO

class YoloDetector:
    def __init__(self, model_name="best_with_ball.pt", conf=0.25, target_classes=None):
        self.model = YOLO(model_name, task = "detect")
        self.conf = conf
        self.names = self.model.names
        self.target_classes = target_classes or ["drone", "ball"]

    def _is_valid_bbox(self, bbox, frame_shape):
        x1, y1, x2, y2 = bbox
        h, w = frame_shape[:2]
        box_w = x2 - x1
        box_h = y2 - y1
        if box_w <= 0 or box_h <= 0:
            return False
        frame_area = w * h
        box_area = box_w * box_h
        if box_area / frame_area > 0.5:
            return False
        aspect = box_w / box_h if box_h != 0 else 0
        if aspect < 0.1 or aspect > 10.0:
            return False
        return True

    def detect(self, frame):
        """
        Возвращает словарь лучших детекций по каждому классу:
        {'drone': {...}, 'ball': {...}}
        """
        results = self.model(frame, conf=self.conf, imgsz=640,half=True, verbose=False)[0]
        detections = {}

        if results.boxes is None or len(results.boxes) == 0:
            return detections

        for box in results.boxes:
            class_id = int(box.cls[0].item())
            class_name = self.names.get(class_id)

            if class_name not in self.target_classes:
                continue

            conf = float(box.conf[0].item())
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            bbox = (int(x1), int(y1), int(x2), int(y2))

            if not self._is_valid_bbox(bbox, frame.shape):
                continue

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            # Оставляем объект с максимальным confidence для каждого класса
            if class_name not in detections or conf > detections[class_name]['conf']:
                detections[class_name] = {
                    'bbox': bbox,
                    'center': (cx, cy),
                    'conf': conf,
                    'class_id': class_id,
                    'class_name': class_name
                }

        return detections

    def detect_batch(self, frames):
        return [self.detect(f) for f in frames]