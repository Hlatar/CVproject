BALL_TRAJECTORY_PLOT_FILE = "trajectory_ball.png"
DRONE_TRAJECTORY_PLOT_FILE = "trajectory_drone.png"

# Модель и пороги
YOLO_MODEL = "best_with_ball.pt"
YOLO_CONF = 0.25

# Отслеживаемые классы
TARGET_CLASSES = ["drone", "ball"]

# === Настройки RealSense ===
REALSENSE_USE_COLOR = True
REALSENSE_WIDTH = 640       # 640x480 — разрешение для 60 FPS на RGB-камере
REALSENSE_HEIGHT = 480
REALSENSE_FPS = 60

USE_AUTO_EXPOSURE = True    # Включена автоэкспозиция
REALSENSE_EXPOSURE = 150