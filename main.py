import cv2 as cv

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
from core.geometry import triangulate_point
from core.trajectory import Trajectory3D
from core.detector import YoloDetector
from core.visualization import (
    draw_cam_axes,
    draw_world_axes_at_point,
    draw_detection_overlay,
    draw_3d_text,
)


recording = False


def main():
    global recording

    calib = StereoCalibration(CALIB_FILE)
    cameras = StereoCameraSystem(CAM_ID_1, CAM_ID_2)
    detector = YoloDetector(
        model_name=YOLO_MODEL,
        conf=YOLO_CONF,
        target_class_name=TARGET_CLASS_NAME,
    )
    trajectory = Trajectory3D()

    print("Запущен режим YOLO-детекции одного объекта.")
    print("")
    print("Управление:")
    print("  R   - старт / стоп записи 3D-траектории")
    print("  C   - очистить траекторию")
    print("  ESC - выход")
    print("")
    print("YOLO автоматически ищет объект на обеих камерах.")

    while True:
        ret1, frame1, ret2, frame2 = cameras.read()
        if not ret1 or not ret2:
            print("Ошибка чтения кадров")
            break

        vis1 = frame1.copy()
        vis2 = frame2.copy()
        
        small1 = cv.resize(frame1, (640, 480))
        small2 = cv.resize(frame2, (640, 480))

        det1 = detector.detect(small1)
        det2 = detector.detect(small2)

        # det1 = detector.detect(frame1)
        # det2 = detector.detect(frame2)

        draw_detection_overlay(vis1, det1, "Cam1")
        draw_detection_overlay(vis2, det2, "Cam2")

        X = None
        if det1 is not None and det2 is not None:

            scale_x1 = frame1.shape[1] / small1.shape[1]
            scale_y1 = frame1.shape[0] / small1.shape[0]
            scale_x2 = frame2.shape[1] / small2.shape[1]
            scale_y2 = frame2.shape[0] / small2.shape[0]

            pt1 = (
                det1["center"][0] * scale_x1,
                det1["center"][1] * scale_y1
            )
            
            pt2 = (
                det2["center"][0] * scale_x2,
                det2["center"][1] * scale_y2
            )
            # pt1 = det1["center"]
            # pt2 = det2["center"]

            X = triangulate_point(pt1, pt2, calib)

            draw_world_axes_at_point(
                vis1, calib.K1, calib.D1, calib.rvec1, calib.tvec1, X, axis_len_m=AXIS_LEN_M
            )
            draw_world_axes_at_point(
                vis2, calib.K2, calib.D2, calib.rvec2, calib.tvec2, X, axis_len_m=AXIS_LEN_M
            )

            draw_3d_text(vis1, X)
            draw_3d_text(vis2, X)

            if recording:
                trajectory.add(X.copy())
        else:
            draw_3d_text(vis1, None)
            draw_3d_text(vis2, None)

        rec_text = "REC ON" if recording else "REC OFF"
        rec_color = (0, 0, 255) if recording else (180, 180, 180)

        cv.putText(vis1, rec_text, (10, 120), cv.FONT_HERSHEY_SIMPLEX, 0.8, rec_color, 2)
        cv.putText(vis2, rec_text, (10, 120), cv.FONT_HERSHEY_SIMPLEX, 0.8, rec_color, 2)

        cv.putText(vis1, f"Points saved: {len(trajectory)}", (10, 155),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv.putText(vis2, f"Points saved: {len(trajectory)}", (10, 155),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if det1 is None:
            cv.putText(vis1, "YOLO did not find object", (10, 190),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)
        if det2 is None:
            cv.putText(vis2, "YOLO did not find object", (10, 190),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)

        draw_cam_axes(vis1, K=calib.K1, label="Cam1")
        draw_cam_axes(vis2, K=calib.K2, label="Cam2")

        cv.imshow("Camera 1 - YOLO Detection", vis1)
        cv.imshow("Camera 2 - YOLO Detection", vis2)

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
                trajectory.save_plot(TRAJECTORY_PLOT_FILE)

        elif key in (ord("c"), ord("C")):
            recording = False
            trajectory.clear()
            print("Траектория очищена.")

    cameras.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()