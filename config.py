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

# === Параметры фильтра Калмана ===
# Мяч резко меняет направление при ударах/отскоках -> 15.0
# Дрон летит плавнее -> 3.0
KF_BALL_PROCESS_NOISE = 15.0
KF_DRONE_PROCESS_NOISE = 3.0

# Дисперсия шума измерений RealSense
KF_MEASUREMENT_NOISE = 0.0025  # std = 5 см (0.05^2)
KF_Z_NOISE_FACTOR = 4.0  # по Z std = 10 см (0.10^2 = 0.01)