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
    REALSENSE_USE_COLOR,          # Использовать цветной поток для YOLO (если True) или IR (если False)
    REALSENSE_WIDTH,
    REALSENSE_HEIGHT,
    REALSENSE_FPS,
    REALSENSE_USE_DEPTH,  
)

from core.calibration import StereoCalibration
from core.camera import StereoCameraSystem
from core.detector import YoloDetector
from core.trajectory import Trajectory3D
from core.kalman3d import KalmanFilter3D
from core.camera_realsense import RealSenseCameraSystem
from core.calibration import StereoCalibration

from core.geometry import (
    triangulate_point,
    camera_to_world_coordinates,
    is_valid_3d_point,
    depth_pixel_to_3d
)

from core.visualization import (
    draw_cam_axes,
    draw_world_axes_at_point,
    draw_detection_overlay,
    draw_3d_text,
    create_single_drone_dashboard,
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
        use_tracker=True,        # включает трекер
        tracker_type='CSRT',     # или 'KCF' (быстрее, но менее точен)
        max_tracker_age=10,      # сколько кадров держать без YOLO
        tracker_reset_iou=0.3    # игнорировать YOLO, если новый бокс далеко от трека
    )

    trajectory = Trajectory3D(
        smoothing_window=5,
        max_jump=2.0,
    )

    kf_3d = KalmanFilter3D(dt=1.0, process_noise=0.2, measurement_noise=0.5)
    max_miss_frames = 5      # допустимое число кадров без детекций, после которого сбрасываем Калман
    miss_counter = 0

    paused = False

    print("Запущен режим YOLO-детекции одного объекта.")
    print("")
    print("Управление:")
    print("  R   - старт / стоп записи 3D-траектории")
    print("  S   - сохранить текущий 3D-график без остановки записи")
    print("  C   - очистить траекторию")
    print("  P   - пауза")
    print("  ESC - выход")
    print("  O   - установить текущую позицию как точку отсчёта")
    print("  B   - сбросить точку отсчёта")
    print("")
    print("YOLO автоматически ищет объект на обеих камерах.")
    print("")
    print("Важно:")
    print("  Камеры должны стоять как стереопара: рядом, направлены в одну сторону.")
    print("  После перестановки камер нужна новая stereo calibration.")
    print("")

    fps_smooth = 0.0
    origin_world = None
    last_valid_world = None
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
            small1 = frame1
            small2 = frame2

            det1_small, det2_small = detector.detect_batch([small1, small2])

            # Возвращаем координаты bbox и center в оригинальный размер кадра
            det1 = scale_detection_to_original(det1_small, frame1, small1)
            det2 = scale_detection_to_original(det2_small, frame2, small2)

            # Рисуем детекцию уже на оригинальных кадрах
            draw_detection_overlay(vis1, det1, "Cam1")
            draw_detection_overlay(vis2, det2, "Cam2")

            # --- 3D-оценка через Калман ---
            world_point_valid = False
            X_cam = None
            X_world_raw = None

            if det1 is not None and det2 is not None:
                pt1 = det1["center"]
                pt2 = det2["center"]
                X_cam = triangulate_point(pt1, pt2, calib)
                X_world_raw = camera_to_world_coordinates(X_cam)
                if is_valid_3d_point(X_world_raw):
                    world_point_valid = True
                    last_valid_world = X_world_raw.copy()
                    # Обновляем Калман реальным измерением
                    kf_3d.update(X_world_raw)
                    miss_counter = 0

            # Получаем сглаженную или предсказанную точку
            if world_point_valid:
                world_smooth = kf_3d.get_state()
            else:
                world_smooth = kf_3d.predict()
                miss_counter += 1
                if miss_counter > max_miss_frames:
                    kf_3d.reset()
                    world_smooth = None

            tracking = world_smooth is not None

            # Вычисляем координаты для отображения (со смещением начала отсчёта)
            if tracking:
                if origin_world is not None:
                    X_display = world_smooth - origin_world
                else:
                    X_display = world_smooth.copy()
                # Добавляем в траекторию сглаженную точку (даже если предсказание)
                trajectory.add(X_display.copy())
            else:
                X_display = None

            # Отрисовка на изображениях камер
            if tracking:
                # 3D-оси рисуем только когда была реальная триангуляция
                if world_point_valid and X_cam is not None:
                    draw_world_axes_at_point(
                        vis1, calib.K1, calib.D1, calib.rvec1, calib.tvec1,
                        X_cam, axis_len_m=AXIS_LEN_M,
                    )
                    draw_world_axes_at_point(
                        vis2, calib.K2, calib.D2, calib.rvec2, calib.tvec2,
                        X_cam, axis_len_m=AXIS_LEN_M,
                    )
                draw_3d_text(vis1, X_display)
                draw_3d_text(vis2, X_display)
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

            # FPS
            loop_end = time.perf_counter()
            loop_time = loop_end - loop_start

            current_fps = 1.0 / loop_time if loop_time > 0 else 0.0

            if fps_smooth == 0.0:
                fps_smooth = current_fps
            else:
                fps_smooth = 0.9 * fps_smooth + 0.1 * current_fps

            # Данные для dashboard
            if tracking and world_smooth is not None:
                if origin_world is not None:
                    X_dashboard = world_smooth - origin_world
                else:
                    X_dashboard = world_smooth.copy()
            else:
                X_dashboard = None

            conf1 = det1["conf"] if det1 is not None else None
            conf2 = det2["conf"] if det2 is not None else None

            dashboard = create_single_drone_dashboard(
                frame1=vis1,
                frame2=vis2,
                fps=fps_smooth,
                recording=recording,
                tracking=tracking,
                X_world=X_dashboard,
                conf1=conf1,
                conf2=conf2,
                saved_points=len(trajectory),
                drone_id=1,
            )

            trajectory_panel = draw_trajectory_panel(
                trajectory.get_points(),
                width=500,
                height=dashboard.shape[0],
                scale=80,
            )

            combined = np.hstack([
                dashboard,
                trajectory_panel,
            ])

            cv.imshow("Single Drone 3D Tracking", combined)

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
                fixed_limits=[(-1, 1), (-1, 1), (0, 2)],
                equal_aspect=True,
            )

        elif key in (ord("c"), ord("C")):
            recording = False
            trajectory.clear()
            print("Траектория очищена.")

        elif key in (ord("p"), ord("P")):
            paused = not paused
            print("Пауза:", paused)

        elif key in (ord("o"), ord("O")):
            if last_valid_world is not None:
                origin_world = last_valid_world.copy()
                trajectory.clear()
                print(
                    f"Новая точка отсчёта установлена: "
                    f"X={origin_world[0]:+.3f}, "
                    f"Y={origin_world[1]:+.3f}, "
                    f"Z={origin_world[2]:+.3f}"
                )
            else:
                print("Нельзя установить точку отсчёта: нет валидной 3D-точки.")

        elif key in (ord("b"), ord("B")):
            origin_world = None
            trajectory.clear()
            print("Точка отсчёта сброшена. Используются абсолютные координаты.")

    cameras.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()


# import cv2 as cv
# import numpy as np
# import time

# from config import (
#     REALSENSE_WIDTH,
#     REALSENSE_HEIGHT,
#     REALSENSE_FPS,
#     AXIS_LEN_M,
#     TRAJECTORY_PLOT_FILE,
#     YOLO_MODEL,
#     YOLO_CONF,
#     TARGET_CLASS_NAME,
# )

# from core.camera_realsense import RealSenseCameraSystem
# from core.detector import YoloDetector
# from core.trajectory import Trajectory3D
# from core.kalman3d import KalmanFilter3D
# from core.geometry import (
#     camera_to_world_coordinates,
#     is_valid_3d_point,
#     depth_pixel_to_3d,
# )
# from core.visualization import (
#     draw_detection_overlay,
#     draw_3d_text,
#     draw_trajectory_panel,
# )

# recording = False

# def main():
#     global recording

#     # 1. Инициализация камеры RealSense
#     cameras = RealSenseCameraSystem(
#         width=REALSENSE_WIDTH,
#         height=REALSENSE_HEIGHT,
#         fps=REALSENSE_FPS,
#     )
#     # Получаем матрицу интринсиков цветной камеры (для пересчёта глубина→3D)
#     K_color = cameras.get_color_intrinsics()

#     # 2. Детектор YOLO
#     detector = YoloDetector(
#         model_name=YOLO_MODEL,
#         conf=YOLO_CONF,
#         target_class_name=TARGET_CLASS_NAME,
#         use_tracker=True,
#         tracker_type='CSRT',
#         max_tracker_age=10,
#         tracker_reset_iou=0.3,
#     )

#     # 3. Траектория и фильтр Калмана
#     trajectory = Trajectory3D(smoothing_window=5, max_jump=2.0)
#     kf_3d = KalmanFilter3D(dt=1.0, process_noise=0.2, measurement_noise=0.5)
#     max_miss_frames = 5
#     miss_counter = 0

#     paused = False
#     fps_smooth = 0.0
#     origin_world = None
#     last_valid_world = None

#     print("Запущен режим детекции дрона через RealSense D435i.")
#     print("Управление:")
#     print("  R   - старт / стоп записи 3D-траектории")
#     print("  S   - сохранить текущий 3D-график без остановки записи")
#     print("  C   - очистить траекторию")
#     print("  P   - пауза")
#     print("  ESC - выход")
#     print("  O   - установить текущую позицию как точку отсчёта")
#     print("  B   - сбросить точку отсчёта")
#     print("")

#     while True:
#         loop_start = time.perf_counter()

#         if not paused:
#             # Читаем цветной кадр и соответствующий ему кадр глубины (выровненный)
#             ret, color_frame, depth_frame = cameras.read()
#             if not ret:
#                 print("Ошибка чтения кадров")
#                 break

#             # Копия для отрисовки
#             vis = color_frame.copy()

#             # Детекция на цветном кадре (один вызов)
#             det = detector.detect(color_frame)

#             # Рисуем детекцию
#             draw_detection_overlay(vis, det, "Camera")

#             # --- 3D-оценка через глубину ---
#             world_point_valid = False
#             X_cam = None
#             X_world_raw = None

#             if det is not None:
#                 cx, cy = det["center"]
#                 # Берём значение глубины в центре объекта (в метрах)
#                 depth_value = depth_frame.get_distance(int(cx), int(cy))
#                 if depth_value > 0 and depth_value < 10.0:  # разумный диапазон
#                     # Пересчёт пиксель + глубина → 3D в системе камеры
#                     X_cam = depth_pixel_to_3d(cx, cy, depth_value, K_color)
#                     X_world_raw = camera_to_world_coordinates(X_cam)
#                     if is_valid_3d_point(X_world_raw):
#                         world_point_valid = True
#                         last_valid_world = X_world_raw.copy()
#                         kf_3d.update(X_world_raw)
#                         miss_counter = 0

#             # Калмановское сглаживание / предсказание
#             if world_point_valid:
#                 world_smooth = kf_3d.get_state()
#             else:
#                 world_smooth = kf_3d.predict()
#                 miss_counter += 1
#                 if miss_counter > max_miss_frames:
#                     kf_3d.reset()
#                     world_smooth = None

#             tracking = world_smooth is not None

#             # Вычисляем координаты для отображения (со смещением начала отсчёта)
#             if tracking:
#                 if origin_world is not None:
#                     X_display = world_smooth - origin_world
#                 else:
#                     X_display = world_smooth.copy()
#                 trajectory.add(X_display.copy())
#             else:
#                 X_display = None

#             # Отрисовка 3D-текста
#             draw_3d_text(vis, X_display)

#             # Статус записи и количество точек
#             rec_text = "REC ON" if recording else "REC OFF"
#             rec_color = (0, 0, 255) if recording else (180, 180, 180)
#             cv.putText(vis, rec_text, (10, 120), cv.FONT_HERSHEY_SIMPLEX, 0.8, rec_color, 2)
#             cv.putText(vis, f"Points saved: {len(trajectory)}", (10, 155),
#                        cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

#             if det is None:
#                 cv.putText(vis, "YOLO did not find object", (10, 190),
#                            cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)

#             # FPS
#             loop_end = time.perf_counter()
#             loop_time = loop_end - loop_start
#             current_fps = 1.0 / loop_time if loop_time > 0 else 0.0
#             fps_smooth = current_fps if fps_smooth == 0.0 else 0.9 * fps_smooth + 0.1 * current_fps

#             # Сборка панели траектории (справа)
#             trajectory_panel = draw_trajectory_panel(
#                 trajectory.get_points(),
#                 width=500,
#                 height=vis.shape[0],
#                 scale=80,
#             )

#             # Объединяем кадр и панель траектории
#             combined = np.hstack([vis, trajectory_panel])
#             cv.imshow("Drone Tracking (RealSense)", combined)

#         # Обработка клавиш
#         key = cv.waitKey(1) & 0xFF
#         if key == 27:  # ESC
#             break

#         elif key in (ord('r'), ord('R')):
#             if not recording:
#                 trajectory.clear()
#                 recording = True
#                 print("Запись траектории началась.")
#             else:
#                 recording = False
#                 print("Запись траектории остановлена.")
#                 print(f"Сохранено точек: {len(trajectory)}")
#                 trajectory.save_plot(
#                     TRAJECTORY_PLOT_FILE,
#                     fixed_limits=[(-2, 2), (-2, 2), (0, 3)],
#                     equal_aspect=True,
#                 )

#         elif key in (ord('s'), ord('S')):
#             print(f"Сохранение графика. Точек: {len(trajectory)}")
#             trajectory.save_plot(
#                 TRAJECTORY_PLOT_FILE,
#                 fixed_limits=[(-1, 1), (-1, 1), (0, 2)],
#                 equal_aspect=True,
#             )

#         elif key in (ord('c'), ord('C')):
#             recording = False
#             trajectory.clear()
#             print("Траектория очищена.")

#         elif key in (ord('p'), ord('P')):
#             paused = not paused
#             print("Пауза:", paused)

#         elif key in (ord('o'), ord('O')):
#             if last_valid_world is not None:
#                 origin_world = last_valid_world.copy()
#                 trajectory.clear()
#                 print(f"Новая точка отсчёта установлена: X={origin_world[0]:+.3f}, Y={origin_world[1]:+.3f}, Z={origin_world[2]:+.3f}")
#             else:
#                 print("Нельзя установить точку отсчёта: нет валидной 3D-точки.")

#         elif key in (ord('b'), ord('B')):
#             origin_world = None
#             trajectory.clear()
#             print("Точка отсчёта сброшена. Используются абсолютные координаты.")

#     cameras.release()
#     cv.destroyAllWindows()

# if __name__ == "__main__":
#     main()