import cv2 as cv
import numpy as np
import time
from config import (
    REALSENSE_WIDTH, REALSENSE_HEIGHT, REALSENSE_FPS,
    TRAJECTORY_PLOT_FILE, YOLO_MODEL, YOLO_CONF, TARGET_CLASS_NAME
)
from core.camera_realsense import RealSenseCameraSystem
from core.detector import YoloDetector
from core.trajectory import Trajectory3D
from core.kalman3d import KalmanFilter3D
from core.geometry import (
    camera_to_world_coordinates,
    is_valid_3d_point,
    depth_pixel_to_3d,
)
from core.visualization import (
    draw_detection_overlay,
    draw_3d_text,
    draw_trajectory_panel,
)

recording = False

def main():
    global recording
    cameras = RealSenseCameraSystem(REALSENSE_WIDTH, REALSENSE_HEIGHT, REALSENSE_FPS)
    K_color = cameras.get_color_intrinsics()

    detector = YoloDetector(
        model_name=YOLO_MODEL,
        conf=YOLO_CONF,
        target_class_name=TARGET_CLASS_NAME,
        use_tracker=True,
        tracker_type='CSRT',
        max_tracker_age=10,
        tracker_reset_iou=0.3,
    )

    trajectory = Trajectory3D(smoothing_window=5, max_jump=2.0)
    kf_3d = KalmanFilter3D(dt=1.0, process_noise=0.2, measurement_noise=0.5)
    max_miss_frames = 5
    miss_counter = 0

    paused = False
    fps_smooth = 0.0
    origin_world = None
    last_valid_world = None

    print("=== Drone Tracking with RealSense D435i (Color + Depth) ===")
    print("Управление: R - запись, S - сохранить график, C - очистить, P - пауза, ESC - выход, O - установить ноль, B - сбросить ноль")

    while True:
        loop_start = time.perf_counter()
        if not paused:
            ret, color_frame, depth_frame = cameras.read()
            if not ret:
                print("Ошибка чтения кадров")
                break

            vis = color_frame.copy()
            world_point_valid = False

            # ---- YOLO на цветном кадре ----
            det = detector.detect(color_frame)
            if det is not None:
                draw_detection_overlay(vis, det, "YOLO")
                cx, cy = det["center"]
                depth_value = depth_frame.get_distance(int(cx), int(cy))
                if depth_value > 0:
                    X_cam = depth_pixel_to_3d(cx, cy, depth_value, K_color)
                    X_world_raw = camera_to_world_coordinates(X_cam)
                    if is_valid_3d_point(X_world_raw):
                        world_point_valid = True
                        last_valid_world = X_world_raw.copy()
                        kf_3d.update(X_world_raw)
                        miss_counter = 0

            # ---- Калман ----
            if world_point_valid:
                world_smooth = kf_3d.get_state()
            else:
                world_smooth = kf_3d.predict()
                miss_counter += 1
                if miss_counter > max_miss_frames:
                    kf_3d.reset()
                    world_smooth = None

            tracking = world_smooth is not None

            if tracking:
                if origin_world is not None:
                    X_display = world_smooth - origin_world
                else:
                    X_display = world_smooth.copy()
                trajectory.add(X_display.copy())
            else:
                X_display = None

            draw_3d_text(vis, X_display)

            # Статус
            rec_text = "REC ON" if recording else "REC OFF"
            rec_color = (0, 0, 255) if recording else (180, 180, 180)
            cv.putText(vis, rec_text, (10, 120), cv.FONT_HERSHEY_SIMPLEX, 0.8, rec_color, 2)
            cv.putText(vis, f"Points: {len(trajectory)}", (10, 155), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            if det is None:
                cv.putText(vis, "No detection", (10, 190), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,140,255), 2)

            # FPS
            loop_time = time.perf_counter() - loop_start
            current_fps = 1.0 / loop_time if loop_time > 0 else 0.0
            fps_smooth = current_fps if fps_smooth == 0.0 else 0.9 * fps_smooth + 0.1 * current_fps

            # Панель траектории
            trajectory_panel = draw_trajectory_panel(trajectory.get_points(), width=500, height=vis.shape[0], scale=80)
            combined = np.hstack([vis, trajectory_panel])
            cv.imshow("Drone Tracking (RealSense)", combined)

        # Клавиши
        key = cv.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key in (ord('r'), ord('R')):
            if not recording:
                trajectory.clear()
                recording = True
                print("Запись начата.")
            else:
                recording = False
                print(f"Запись остановлена. Точек: {len(trajectory)}")
                trajectory.save_plot(TRAJECTORY_PLOT_FILE, fixed_limits=[(-2,2),(-2,2),(0,3)], equal_aspect=True)
        elif key in (ord('s'), ord('S')):
            trajectory.save_plot(TRAJECTORY_PLOT_FILE, fixed_limits=[(-1,1),(-1,1),(0,2)], equal_aspect=True)
            print("График сохранён.")
        elif key in (ord('c'), ord('C')):
            recording = False
            trajectory.clear()
            print("Траектория очищена.")
        elif key in (ord('p'), ord('P')):
            paused = not paused
            print("Пауза:", paused)
        elif key in (ord('o'), ord('O')):
            if last_valid_world is not None:
                origin_world = last_valid_world.copy()
                trajectory.clear()
                print(f"Ноль установлен: {origin_world}")
            else:
                print("Нет валидной точки.")
        elif key in (ord('b'), ord('B')):
            origin_world = None
            trajectory.clear()
            print("Ноль сброшен.")

    cameras.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()