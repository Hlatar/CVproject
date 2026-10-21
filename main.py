import cv2 as cv
import numpy as np
import time
from config import (
    REALSENSE_WIDTH, REALSENSE_HEIGHT, REALSENSE_FPS,
    BALL_TRAJECTORY_PLOT_FILE, DRONE_TRAJECTORY_PLOT_FILE,
    YOLO_MODEL, YOLO_CONF, TARGET_CLASSES
)
from core.camera_realsense import RealSenseCameraSystem
from core.detector import YoloDetector
from core.trajectory import Trajectory3D
from core.kalman3d import KalmanFilter3D
from core.geometry import (
    camera_to_world_coordinates,
    is_valid_3d_point,
    depth_pixel_to_3d,
    get_depth_at_center
)
from core.visualization import (
    draw_detection_overlay,
    draw_dual_trajectory_panel
)

rec_ball = False
rec_drone = False

def main():
    global rec_ball, rec_drone

    cameras = RealSenseCameraSystem(REALSENSE_WIDTH, REALSENSE_HEIGHT, REALSENSE_FPS)
    K_color = cameras.get_color_intrinsics()

    detector = YoloDetector(
        model_name=YOLO_MODEL,
        conf=YOLO_CONF,
        target_classes=TARGET_CLASSES
    )

    # Отдельные объекты траекторий и фильтров Калмана
    ball_trajectory = Trajectory3D(smoothing_window=3, max_jump=2.0)
    drone_trajectory = Trajectory3D(smoothing_window=3, max_jump=2.0)

    kf_ball = KalmanFilter3D(dt=1.0/REALSENSE_FPS, process_noise=0.1, measurement_noise=0.2, enable_gravity=True)
    kf_drone = KalmanFilter3D(dt=1.0/REALSENSE_FPS, process_noise=0.1, measurement_noise=0.2, enable_gravity=False)

    miss_ball_counter = 0
    miss_drone_counter = 0
    max_miss_frames = 10

    paused = False
    fps_smooth = 0.0

    print("=== Двухъядерный трекинг (Дрон + Мяч) ===")
    print("Управление:")
    print("  R - старт/стоп записи МЯЧА")
    print("  T - старт/стоп записи ДРОНА")
    print("  S - сохранить 3D-графики траекторий (ball & drone)")
    print("  C - очистить траекторию мяча")
    print("  X - очистить траекторию дрона")
    print("  P - пауза, ESC - выход")

    while True:
        loop_start = time.perf_counter()
        if not paused:
            ret, color_frame, depth_frame = cameras.read()
            if not ret:
                print("Ошибка чтения кадров")
                break

            vis = color_frame.copy()

            # Детекция объектов YOLO
            detections = detector.detect(color_frame)

            # --- 1. ОБРАБОТКА МЯЧА ---
            ball_valid = False
            if "ball" in detections:
                det_ball = detections["ball"]
                draw_detection_overlay(vis, det_ball, "Ball")
                cx, cy = det_ball["center"]
                depth_val = get_depth_at_center(depth_frame, cx, cy)

                if depth_val > 0:
                    X_cam = depth_pixel_to_3d(cx, cy, depth_val, K_color)
                    X_world_raw = camera_to_world_coordinates(X_cam)
                    if is_valid_3d_point(X_world_raw):
                        ball_valid = True
                        kf_ball.update(X_world_raw)
                        miss_ball_counter = 0

            if ball_valid:
                ball_smooth = kf_ball.get_state()
            else:
                ball_smooth = kf_ball.predict()
                miss_ball_counter += 1
                if miss_ball_counter > max_miss_frames:
                    kf_ball.reset()
                    ball_smooth = None

            if ball_smooth is not None:
                if rec_ball:
                    ball_trajectory.add(ball_smooth.copy())

                future_ball_pos = kf_ball.predict_future_point(t_future=0.3)
                cv.putText(vis, f"Ball 3D: X={ball_smooth[0]:.2f} Y={ball_smooth[1]:.2f} Z={ball_smooth[2]:.2f}",
                           (10, 35), cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)
                if future_ball_pos is not None:
                    cv.putText(vis, f"Ball (+0.3s): X={future_ball_pos[0]:.2f} Y={future_ball_pos[1]:.2f} Z={future_ball_pos[2]:.2f}",
                               (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 215, 255), 2)

            # --- 2. ОБРАБОТКА ДРОНА ---
            drone_valid = False
            if "drone" in detections:
                det_drone = detections["drone"]
                draw_detection_overlay(vis, det_drone, "Drone")
                cx, cy = det_drone["center"]
                depth_val = get_depth_at_center(depth_frame, cx, cy)

                if depth_val > 0:
                    X_cam = depth_pixel_to_3d(cx, cy, depth_val, K_color)
                    X_world_raw = camera_to_world_coordinates(X_cam)
                    if is_valid_3d_point(X_world_raw):
                        drone_valid = True
                        kf_drone.update(X_world_raw)
                        miss_drone_counter = 0

            if drone_valid:
                drone_smooth = kf_drone.get_state()
            else:
                drone_smooth = kf_drone.predict()
                miss_drone_counter += 1
                if miss_drone_counter > max_miss_frames:
                    kf_drone.reset()
                    drone_smooth = None

            if drone_smooth is not None:
                if rec_drone:
                    drone_trajectory.add(drone_smooth.copy())

                cv.putText(vis, f"Drone 3D: X={drone_smooth[0]:.2f} Y={drone_smooth[1]:.2f} Z={drone_smooth[2]:.2f}",
                           (10, 85), cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            # FPS
            loop_time = time.perf_counter() - loop_start
            current_fps = 1.0 / loop_time if loop_time > 0 else 0.0
            fps_smooth = current_fps if fps_smooth == 0.0 else 0.9 * fps_smooth + 0.1 * current_fps
            cv.putText(vis, f"FPS: {fps_smooth:.1f}", (10, 115), cv.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            # Панель проекций (содержит траектории мяча и дрона одновременно)
            trajectory_panel = draw_dual_trajectory_panel(
                ball_trajectory.get_points(),
                drone_trajectory.get_points(),
                rec_ball=rec_ball,
                rec_drone=rec_drone,
                width=500,
                height=vis.shape[0],
                scale=80
            )

            combined = np.hstack([vis, trajectory_panel])
            cv.imshow("Drone Catch Ball Tracking", combined)

        # Обработка клавиш
        key = cv.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key in (ord('r'), ord('R')):
            if not rec_ball:
                ball_trajectory.clear()
                rec_ball = True
                print("Запись траектории МЯЧА начата.")
            else:
                rec_ball = False
                print(f"Запись МЯЧА остановлена. Точек: {len(ball_trajectory)}")
                ball_trajectory.save_plot(BALL_TRAJECTORY_PLOT_FILE)
        elif key in (ord('t'), ord('T')):
            if not rec_drone:
                drone_trajectory.clear()
                rec_drone = True
                print("Запись траектории ДРОНА начата.")
            else:
                rec_drone = False
                print(f"Запись ДРОНА остановлена. Точек: {len(drone_trajectory)}")
                drone_trajectory.save_plot(DRONE_TRAJECTORY_PLOT_FILE)
        elif key in (ord('s'), ord('S')):
            ball_trajectory.save_plot(BALL_TRAJECTORY_PLOT_FILE)
            drone_trajectory.save_plot(DRONE_TRAJECTORY_PLOT_FILE)
            print("Оба графика 3D сохранены (trajectory_ball.png, trajectory_drone.png).")
        elif key in (ord('c'), ord('C')):
            ball_trajectory.clear()
            print("Траектория МЯЧА очищена.")
        elif key in (ord('x'), ord('X')):
            drone_trajectory.clear()
            print("Траектория ДРОНА очищена.")
        elif key in (ord('p'), ord('P')):
            paused = not paused
            print("Пауза:", paused)

    cameras.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()