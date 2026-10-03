TRAJECTORY_PLOT_FILE = "trajectory_3d.png" # the image of terajectory

YOLO_MODEL = "best.pt"                     # model name
YOLO_CONF = 0.25                           # confidence threshold (for drone recommended 0.25-0.3)

# model contains only one class: "drone" (index 0)
TARGET_CLASS_NAME = "drone"       # name of clas detection

# === REALSENSE settings ===
REALSENSE_USE_COLOR = True          # using color stream for YOLO (if True) or IR (or False)
REALSENSE_WIDTH = 1280
REALSENSE_HEIGHT = 720
REALSENSE_FPS = 30 