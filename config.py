CAM_ID_1 = 4
CAM_ID_2 = 2
CALIB_FILE = "stereo_params.npz"

AXIS_LEN_M = 0.05
TRAJECTORY_PLOT_FILE = "trajectory_3d.png"

# YOLO - Модель для обнаружения дронов
# Скачать модель: https://huggingface.co/doguilmak/Drone-Detection-YOLOv11x/resolve/main/best.pt
# Сохранить в папку проекта как "best_drone.pt"
YOLO_MODEL = "best.pt"     # имя скачанного файла
YOLO_CONF = 0.25                  # порог уверенности (для дронов рекомендуется 0.25-0.3)

# Для Drone-Detection-YOLOv11x модель имеет только один класс: "drone" (индекс 0)
TARGET_CLASS_NAME = "drone"       # имя класса для детекции

# === НАСТРОЙКИ REALSENSE ===
REALSENSE_USE_COLOR = True          # Использовать цветной поток для YOLO (если True) или IR (если False)
REALSENSE_WIDTH = 1280
REALSENSE_HEIGHT = 720
REALSENSE_FPS = 30
REALSENSE_USE_DEPTH = False    