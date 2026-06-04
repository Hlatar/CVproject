import cv2 as cv
import numpy as np
import time

from config import (
    CAM_ID_1,
    CAM_ID_2,
    CALIB_FILE,
    AXIS_LEN_M,
    TRAJECTORY_PLOT_FILE,
    YOLO_MODEL,
    YOLO_CONF,
    TARGET_CLASS_NAME,
)

from core.calibration import StereoCalibration
from core.camera import StereoCameraSystem
from core.detector import YoloDetector
from core.trajectory import Trajectory3D

from core.geometry import (
    triangulate_point,
    camera_to_world_coordinates,
    is_valid_3d_point,
)

from core.visualization import (
    draw_cam_axes,
    draw_world_axes_at_point,
    draw_detection_overlay,
    draw_3d_text,
    draw_trajectory_panel,
)


recording = False


def scale_detection_to_original(det, original_frame, resized_frame):
    """
    YOLO работает на resized_frame, например 640x480.
    Но триангуляция должна получать координаты в оригинальном размере кадра.

    Эта функция пересчитывает bbox и center обратно в координаты original_frame.
    """

    if det is None:
        return None

    orig_h, orig_w = original_frame.shape[:2]
    small_h, small_w = resized_frame.shape[:2]

    scale_x = orig_w / small_w
    scale_y = orig_h / small_h

    x1, y1, x2, y2 = det["bbox"]
    cx, cy = det["center"]

    scaled_det = det.copy()

    scaled_det["bbox"] = (
        int(x1 * scale_x),
        int(y1 * scale_y),
        int(x2 * scale_x),
        int(y2 * scale_y),
    )

    scaled_det["center"] = (
        float(cx * scale_x),
        float(cy * scale_y),
    )

    return scaled_det


def resize_to_height(img, target_h):
    """
    Меняет размер изображения так, чтобы высота стала target_h.
    Пропорции сохраняются.
    """

    h, w = img.shape[:2]
    scale = target_h / h
    target_w = int(w * scale)

    return cv.resize(img, (target_w, target_h))


def main():
    global recording

    calib = StereoCalibration(CALIB_FILE)

    cameras = StereoCameraSystem(CAM_ID_1, CAM_ID_2)

    detector = YoloDetector(
        model_name=YOLO_MODEL,
        conf=YOLO_CONF,
        target_class_name=TARGET_CLASS_NAME,
    )

    trajectory = Trajectory3D(
        smoothing_window=5,
        max_jump=2.0,
    )

    paused = False

    print("Запущен режим YOLO-детекции одного объекта.")
    print("")
    print("Управление:")
    print("  R   - старт / стоп записи 3D-траектории")
    print("  S   - сохранить текущий 3D-график без остановки записи")
    print("  C   - очистить траекторию")
    print("  P   - пауза")
    print("  ESC - выход")
    print("")
    print("YOLO автоматически ищет объект на обеих камерах.")
    print("")
    print("Важно:")
    print("  Камеры должны стоять как стереопара: рядом, направлены в одну сторону.")
    print("  После перестановки камер нужна новая stereo calibration.")
    print("")

    fps_smooth = 0.0
    while True:
        loop_start = time.perf_counter()
        if not paused:
            ret1, frame1, ret2, frame2 = cameras.read()

            if not ret1 or not ret2:
                print("Ошибка чтения кадров")
                break

            vis1 = frame1.copy()
            vis2 = frame2.copy()

            # YOLO можно гонять на уменьшенных кадрах для скорости
            small1 = cv.resize(frame1, (640, 480))
            small2 = cv.resize(frame2, (640, 480))

            det1_small = detector.detect(small1)
            det2_small = detector.detect(small2)

            # Возвращаем координаты bbox и center в оригинальный размер кадра
            det1 = scale_detection_to_original(det1_small, frame1, small1)
            det2 = scale_detection_to_original(det2_small, frame2, small2)

            # Рисуем детекцию уже на оригинальных кадрах
            draw_detection_overlay(vis1, det1, "Cam1")
            draw_detection_overlay(vis2, det2, "Cam2")

            X_cam = None
            X_world = None

            if det1 is not None and det2 is not None:
                pt1 = det1["center"]
                pt2 = det2["center"]

                # 1. Сырая 3D-точка в системе координат первой камеры OpenCV
                # OpenCV camera:
                #   X вправо
                #   Y вниз
                #   Z вперёд
                X_cam = triangulate_point(pt1, pt2, calib)

                # 2. Переводим в удобную мировую систему:
                #   X вправо
                #   Y вверх
                #   Z вперёд
                X_world = camera_to_world_coordinates(X_cam)

                if is_valid_3d_point(X_world):
                    if recording:
                        trajectory.add(X_world.copy())

                    print(
                        f"Cam:   X={X_cam[0]:+.3f}, "
                        f"Y={X_cam[1]:+.3f}, "
                        f"Z={X_cam[2]:+.3f}"
                    )

                    print(
                        f"World: X={X_world[0]:+.3f}, "
                        f"Y={X_world[1]:+.3f}, "
                        f"Z={X_world[2]:+.3f}"
                    )

                    print("-" * 50)

                    # Оси лучше рисовать по X_cam, потому что projectPoints
                    # работает в системе координат камеры / калибровки.
                    draw_world_axes_at_point(
                        vis1,
                        calib.K1,
                        calib.D1,
                        calib.rvec1,
                        calib.tvec1,
                        X_cam,
                        axis_len_m=AXIS_LEN_M,
                    )

                    draw_world_axes_at_point(
                        vis2,
                        calib.K2,
                        calib.D2,
                        calib.rvec2,
                        calib.tvec2,
                        X_cam,
                        axis_len_m=AXIS_LEN_M,
                    )

                    # А текст лучше показывать в world-координатах,
                    # потому что они понятнее человеку.
                    draw_3d_text(vis1, X_world)
                    draw_3d_text(vis2, X_world)

                else:
                    print("Невалидная 3D-точка:", X_world)
                    draw_3d_text(vis1, None)
                    draw_3d_text(vis2, None)

            else:
                draw_3d_text(vis1, None)
                draw_3d_text(vis2, None)

            # Статус записи
            rec_text = "REC ON" if recording else "REC OFF"
            rec_color = (0, 0, 255) if recording else (180, 180, 180)

            cv.putText(
                vis1,
                rec_text,
                (10, 120),
                cv.FONT_HERSHEY_SIMPLEX,
                0.8,
                rec_color,
                2,
            )

            cv.putText(
                vis2,
                rec_text,
                (10, 120),
                cv.FONT_HERSHEY_SIMPLEX,
                0.8,
                rec_color,
                2,
            )

            cv.putText(
                vis1,
                f"Points saved: {len(trajectory)}",
                (10, 155),
                cv.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            cv.putText(
                vis2,
                f"Points saved: {len(trajectory)}",
                (10, 155),
                cv.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            if det1 is None:
                cv.putText(
                    vis1,
                    "YOLO did not find object",
                    (10, 190),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 140, 255),
                    2,
                )

            if det2 is None:
                cv.putText(
                    vis2,
                    "YOLO did not find object",
                    (10, 190),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 140, 255),
                    2,
                )

            # Оси камер
            draw_cam_axes(vis1, K=calib.K1, label="Cam1")
            draw_cam_axes(vis2, K=calib.K2, label="Cam2")

            # Если ты добавил draw_trajectory_panel в visualization.py,
            # можно показывать всё в одном окне.
            points = trajectory.get_points()
            panel = draw_trajectory_panel(points, width=500, height=720, scale=80)

            panel_h = panel.shape[0]
            loop_end = time.perf_counter()
            loop_time = loop_end - loop_start

            current_fps = 1.0 / loop_time if loop_time > 0 else 0.0
            fps_smooth = 0.9 * fps_smooth + 0.1 * current_fps
            
            cv.putText(
                vis1,
                f"FPS: {fps_smooth:.1f}",
                (10, 230),
                cv.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

            vis1_resized = resize_to_height(vis1, panel_h)
            vis2_resized = resize_to_height(vis2, panel_h)

            combined = np.hstack([
                vis1_resized,
                vis2_resized,
                panel,
            ])

            cv.imshow("Drone 3D tracking", combined)

            # Если хочешь оставить два отдельных окна вместо одного,
            # можешь раскомментировать:
            #
            # cv.imshow("Camera 1 - YOLO Detection", vis1)
            # cv.imshow("Camera 2 - YOLO Detection", vis2)

        key = cv.waitKey(1) & 0xFF

        if key == 27:
            break

        elif key in (ord("r"), ord("R")):
            if not recording:
                trajectory.clear()
                recording = True
                print("Запись траектории началась.")
            else:
                recording = False
                print("Запись траектории остановлена.")
                print(f"Сохранено точек: {len(trajectory)}")

                trajectory.save_plot(
                    TRAJECTORY_PLOT_FILE,
                    fixed_limits=[(-2, 2), (-2, 2), (0, 3)],
                    equal_aspect=True,
                )

        elif key in (ord("s"), ord("S")):
            print(f"Сохранение графика. Точек: {len(trajectory)}")

            trajectory.save_plot(
                TRAJECTORY_PLOT_FILE,
                fixed_limits=[(-2, 2), (-2, 2), (0, 3)],
                equal_aspect=True,
            )

        elif key in (ord("c"), ord("C")):
            recording = False
            trajectory.clear()
            print("Траектория очищена.")

        elif key in (ord("p"), ord("P")):
            paused = not paused
            print("Пауза:", paused)

    cameras.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()