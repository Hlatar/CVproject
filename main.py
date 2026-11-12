import cv2 as cv
import numpy as np
import time
from config import *
from core.camera_realsense import RealSenseCameraSystem
from core.detector import YoloDetector
from core.visualization import draw_detection_overlay, draw_dual_trajectory_panel
from core.target_tracker import TrackedTarget
from core.keyboard_handler import handle_keyboard

def main():
    cameras = RealSenseCameraSystem(REALSENSE_WIDTH, REALSENSE_HEIGHT, REALSENSE_FPS)
    K_color = cameras.get_color_intrinsics()
    detector = YoloDetector(model_name=YOLO_MODEL, conf=YOLO_CONF, target_classes=TARGET_CLASSES)

    # Создаем 2 независимых объекта-трекера вместо разрозненных переменных
    ball = TrackedTarget("Ball", (0, 165, 255), {
        'dt': 1.0/REALSENSE_FPS, 'process_noise': KF_BALL_PROCESS_NOISE,
        'measurement_noise': KF_MEASUREMENT_NOISE, 'z_noise_factor': KF_Z_NOISE_FACTOR, 'enable_gravity': True
    })
    drone = TrackedTarget("Drone", (0, 255, 0), {
        'dt': 1.0/REALSENSE_FPS, 'process_noise': KF_DRONE_PROCESS_NOISE,
        'measurement_noise': KF_MEASUREMENT_NOISE, 'z_noise_factor': KF_Z_NOISE_FACTOR, 'enable_gravity': False
    })

    paused = False
    fps_smooth = 0.0

    while True:
        loop_start = time.perf_counter()
        if not paused:
            ret, color_frame, depth_frame = cameras.read()
            if not ret: break

            vis = color_frame.copy()
            detections = detector.detect(color_frame)

            # Отрисовка детекций YOLO
            for key, target in [("ball", ball), ("drone", drone)]:
                if key in detections:
                    draw_detection_overlay(vis, detections[key], target.name)

            # Обработка трекеров (в 2 строчки!)
            ball.process_frame(detections.get("ball"), depth_frame, K_color)
            drone.process_frame(detections.get("drone"), depth_frame, K_color)

            # Отрисовка инфо
            ball.draw_info(vis, y_offset=35)
            drone.draw_info(vis, y_offset=85)

            # FPS и UI Панель
            loop_time = time.perf_counter() - loop_start
            current_fps = 1.0 / loop_time if loop_time > 0 else 0.0
            fps_smooth = current_fps if fps_smooth == 0.0 else 0.9 * fps_smooth + 0.1 * current_fps
            cv.putText(vis, f"FPS: {fps_smooth:.1f}", (10, 115), cv.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            panel = draw_dual_trajectory_panel(
                ball.trajectory.get_points(), drone.trajectory.get_points(),
                rec_ball=ball.is_recording, rec_drone=drone.is_recording,
                width=500, height=vis.shape[0]
            )
            cv.imshow("Drone Catch Ball Tracking", np.hstack([vis, panel]))

        key = cv.waitKey(1) & 0xFF
        action = handle_keyboard(key, ball, drone, BALL_TRAJECTORY_PLOT_FILE, DRONE_TRAJECTORY_PLOT_FILE)
        if action == "EXIT":
            break
        elif action == "TOGGLE_PAUSE":
            paused = not paused

    cameras.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()