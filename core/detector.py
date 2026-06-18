import cv2
from ultralytics import YOLO
import numpy as np

class YoloDetector:
    def __init__(self, model_name="best.pt", conf=0.35, target_class_name=None,
                 use_tracker=True, tracker_type='CSRT', max_tracker_age=10,
                 tracker_reset_iou=0.3):
        """
        use_tracker: включить трекинг для устойчивости
        tracker_type: 'CSRT' (точный, медленный) или 'KCF' (быстрее)
        max_tracker_age: сколько кадров трекер может жить без подтверждения YOLO
        tracker_reset_iou: если IoU новой YOLO-бокса с треком < порога, не переключаемся (подавление ложных срабатываний)
        """
        self.model = YOLO(model_name)
        self.conf = conf
        self.names = self.model.names
        self.target_class_name = target_class_name
        self.target_class_id = self._resolve_class_id(target_class_name)

        self.use_tracker = use_tracker
        self.tracker_type = tracker_type
        self.max_tracker_age = max_tracker_age
        self.tracker_reset_iou = tracker_reset_iou

        # Для каждого индекса камеры в batch храним отдельный трекер и его возраст
        self.trackers = {}          # key = camera_index, value = cv2.Tracker
        self.tracker_ages = {}      # сколько кадров трекер не подтверждался YOLO

    def _resolve_class_id(self, class_name):
        if class_name is None:
            return None
        for cid, name in self.names.items():
            if name == class_name:
                return cid
        raise ValueError(f"Класс '{class_name}' не найден. Доступные: {list(self.names.values())}")

    def _create_tracker(self):
        # Проверяем доступные варианты для разных версий OpenCV
        if self.tracker_type == 'CSRT':
            # Пробуем современный путь
            if hasattr(cv2, 'TrackerCSRT_create'):
                return cv2.TrackerCSRT_create()
            # Затем legacy
            elif hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerCSRT_create'):
                return cv2.legacy.TrackerCSRT_create()
            else:
                raise RuntimeError("OpenCV установлен без модуля tracking (contrib). Установите opencv-contrib-python")
        elif self.tracker_type == 'KCF':
            if hasattr(cv2, 'TrackerKCF_create'):
                return cv2.TrackerKCF_create()
            elif hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerKCF_create'):
                return cv2.legacy.TrackerKCF_create()
            else:
                raise RuntimeError("OpenCV установлен без модуля tracking (contrib). Установите opencv-contrib-python")
        else:
            # По умолчанию CSRT
            return self._create_tracker()  # рекурсивно с CSRT, но лучше явно вызвать для нужного типа

    def _box_iou(self, boxA, boxB):
        """IoU двух боксов в формате (x1,y1,x2,y2)"""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        if interArea == 0:
            return 0.0
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return interArea / float(boxAArea + boxBArea - interArea)

    def _extract_best_from_result(self, result):
        """Извлекает лучший объект из одного результата YOLO (как раньше, но возвращает None если нет)"""
        boxes = result.boxes
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
            bbox = (int(x1), int(y1), int(x2), int(y2))
            if not self._is_valid_bbox(bbox, result.orig_shape):
                continue
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            h, w = result.orig_shape[:2]
            cx = max(0, min(cx, w - 1))
            cy = max(0, min(cy, h - 1))
            if conf > best_conf:
                best_conf = conf
                best = {
                    'bbox': bbox,
                    'center': (cx, cy),
                    'conf': conf,
                    'class_id': class_id,
                    'class_name': self.names[class_id]
                }
        return best

    def _is_valid_bbox(self, bbox, frame_shape):
        """Проверка бокса (оставлена без изменений)"""
        x1, y1, x2, y2 = bbox
        h, w = frame_shape[:2]
        box_w = x2 - x1
        box_h = y2 - y1
        if box_w <= 0 or box_h <= 0:
            return False
        frame_area = w * h
        box_area = box_w * box_h
        area_ratio = box_area / frame_area
        if area_ratio > 0.35:
            return False
        aspect = box_w / box_h if box_h != 0 else 0
        if aspect < 0.15 or aspect > 6.0:
            return False
        return True

    def detect(self, frame):
        """Одиночный кадр (для совместимости, не используется в main)"""
        # Для одиночного вызова можно вызывать detect_batch с одним кадром
        return self.detect_batch([frame])[0]

    def detect_batch(self, frames):
        # 1. YOLO-прогон
        results = self.model(frames, conf=self.conf, imgsz=640, verbose=False)
        detections = []

        for i, result in enumerate(results):
            yolo_det = self._extract_best_from_result(result)

            # Если трекер не используется – сразу возвращаем YOLO (или None)
            if not self.use_tracker:
                detections.append(yolo_det)
                continue

            # Текущий трекер для этого индекса камеры
            tracker = self.trackers.get(i)
            age = self.tracker_ages.get(i, 0)

            # YOLO что-то нашла
            if yolo_det is not None:
                # Если трекер уже существует, проверяем, не ложное ли это срабатывание
                if tracker is not None and age < self.max_tracker_age:
                    # Обновляем трекер, чтобы получить его текущее положение
                    success, trk_bbox = tracker.update(frames[i])
                    if success:
                        x, y, w, h = [int(v) for v in trk_bbox]
                        trk_box = (x, y, x + w, y + h)
                        iou = self._box_iou(yolo_det['bbox'], trk_box)
                        if iou < self.tracker_reset_iou:
                            # YOLO нашла что-то далеко от трека -> игнорируем, продолжаем с трекером
                            cx_trk = x + w/2
                            cy_trk = y + h/2
                            detections.append({
                                'bbox': (x, y, x+w, y+h),
                                'center': (cx_trk, cy_trk),
                                'conf': None,
                                'class_id': self.target_class_id,
                                'class_name': self.target_class_name,
                                'from_tracker': True
                            })
                            self.tracker_ages[i] = age + 1
                            continue
                # YOLO валидна – переинициализируем трекер на новом боксе
                bbox_yolo = yolo_det['bbox']
                x1, y1, x2, y2 = bbox_yolo
                w_t = x2 - x1
                h_t = y2 - y1
                new_tracker = self._create_tracker()
                new_tracker.init(frames[i], (x1, y1, w_t, h_t))
                self.trackers[i] = new_tracker
                self.tracker_ages[i] = 0
                detections.append(yolo_det)

            else:
                # YOLO не нашла – используем трекер, если он жив
                if tracker is not None and age < self.max_tracker_age:
                    success, trk_bbox = tracker.update(frames[i])
                    if success:
                        x, y, w, h = [int(v) for v in trk_bbox]
                        cx = x + w/2
                        cy = y + h/2
                        h_fr, w_fr = frames[i].shape[:2]
                        cx = max(0, min(cx, w_fr - 1))
                        cy = max(0, min(cy, h_fr - 1))
                        detections.append({
                            'bbox': (x, y, x+w, y+h),
                            'center': (cx, cy),
                            'conf': None,
                            'class_id': self.target_class_id,
                            'class_name': self.target_class_name,
                            'from_tracker': True
                        })
                        self.tracker_ages[i] = age + 1
                    else:
                        # трекер потерялся
                        detections.append(None)
                        if i in self.trackers:
                            del self.trackers[i]
                            del self.tracker_ages[i]
                else:
                    detections.append(None)
                    # очищаем, если был старый трекер
                    if i in self.trackers:
                        del self.trackers[i]
                        del self.tracker_ages[i]

        return detections