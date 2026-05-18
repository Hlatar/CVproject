from ultralytics import YOLO


class YoloDetector:
    def __init__(self, model_name="best.pt", conf=0.35, target_class_name=None):
        self.model = YOLO(model_name)
        self.conf = conf
        self.names = self.model.names
        self.target_class_name = target_class_name
        self.target_class_id = self._resolve_class_id(target_class_name)

    def _resolve_class_id(self, class_name):
        if class_name is None:
            return None

        for class_id, name in self.names.items():
            if name == class_name:
                return class_id

        raise ValueError(
            f"Класс '{class_name}' не найден в модели. "
            f"Доступные классы: {list(self.names.values())}"
        )

    def detect(self, frame):
        # Для Drone-Detection-YOLOv11x лучше использовать imgsz=640
        # Модель обучена на этом разрешении и лучше детектит маленькие дроны
        results = self.model(frame, conf=self.conf, imgsz=640, verbose=False)
        boxes = results[0].boxes

        if boxes is None or len(boxes) == 0:
            return None

        best = None
        best_conf = -1.0

        for box in boxes:
            class_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())

            if self.target_class_id is not None and class_id != self.target_class_id:
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cx = float((x1 + x2) / 2.0)
            cy = float((y1 + y2) / 2.0)

            if conf > best_conf:
                best_conf = conf
                best = {
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    "center": (cx, cy),
                    "conf": conf,
                    "class_id": class_id,
                    "class_name": self.names[class_id],
                }

        return best