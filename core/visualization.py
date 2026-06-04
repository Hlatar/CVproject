import cv2 as cv
import numpy as np


def draw_cam_axes(img, K=None, label="", axis_len=60, thickness=2):
    h, w = img.shape[:2]
    if K is not None and K.shape == (3, 3):
        origin = (int(round(K[0, 2])), int(round(K[1, 2])))
    else:
        origin = (w // 2, h // 2)

    cv.circle(img, origin, 5, (255, 255, 255), -1)
    cv.circle(img, origin, 7, (0, 0, 0), 1)

    x_end = (origin[0] + axis_len, origin[1])
    y_end = (origin[0], origin[1] + axis_len)
    z_end = (origin[0] - int(axis_len * 0.7), origin[1] - int(axis_len * 0.7))

    cv.arrowedLine(img, origin, x_end, (0, 0, 255), thickness, tipLength=0.25)
    cv.arrowedLine(img, origin, y_end, (0, 255, 0), thickness, tipLength=0.25)
    cv.arrowedLine(img, origin, z_end, (255, 0, 0), thickness, tipLength=0.25)

    cv.putText(img, "X+", (x_end[0] + 5, x_end[1] + 5), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv.putText(img, "Y+", (y_end[0] - 30, y_end[1] + 25), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv.putText(img, "Z+", (z_end[0] - 35, z_end[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    if label:
        cv.putText(
            img,
            f"{label} origin ({origin[0]},{origin[1]})",
            (10, h - 10),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )


def draw_world_axes_at_point(img, K, D, rvec, tvec, X_world, axis_len_m=0.05, thickness=2):
    X_world = np.asarray(X_world, dtype=np.float32).reshape(3)
    pts3d = np.array([
        X_world,
        X_world + np.array([axis_len_m, 0, 0], dtype=np.float32),
        X_world + np.array([0, axis_len_m, 0], dtype=np.float32),
        X_world + np.array([0, 0, axis_len_m], dtype=np.float32),
    ], dtype=np.float32).reshape(-1, 1, 3)

    imgpts, _ = cv.projectPoints(pts3d, rvec, tvec, K, D)
    imgpts = imgpts.reshape(-1, 2).astype(np.int32)
    o, x, y, z = tuple(imgpts[0]), tuple(imgpts[1]), tuple(imgpts[2]), tuple(imgpts[3])

    cv.circle(img, o, 5, (255, 255, 255), -1)
    cv.arrowedLine(img, o, x, (0, 0, 255), thickness, tipLength=0.25)
    cv.arrowedLine(img, o, y, (0, 255, 0), thickness, tipLength=0.25)
    cv.arrowedLine(img, o, z, (255, 0, 0), thickness, tipLength=0.25)


def draw_detection_overlay(img, det, camera_label="Cam"):
    if det is None:
        cv.putText(
            img,
            f"{camera_label}: object not found",
            (10, 40),
            cv.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )
        return

    x1, y1, x2, y2 = det["bbox"]
    cx, cy = det["center"]
    conf = det["conf"]
    class_name = det["class_name"]

    cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv.circle(img, (int(round(cx)), int(round(cy))), 5, (0, 0, 255), -1)

    cv.putText(
        img,
        f"{camera_label}: {class_name} {conf:.2f}",
        (x1, max(20, y1 - 10)),
        cv.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )


def draw_3d_text(img, X):
    if X is None:
        cv.putText(img, "3D: N/A", (10, 80), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        cv.putText(
            img,
            f"3D: X={X[0]:.3f}  Y={X[1]:.3f}  Z={X[2]:.3f}",
            (10, 80),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

def draw_trajectory_panel(points, width=500, height=720, scale=80):
    """
    Рисует панель траектории с тремя проекциями:

    1. XZ — вид сверху
    2. XY — вид спереди
    3. ZY — вид сбоку
    """

    panel = np.zeros((height, width, 3), dtype=np.uint8)

    cv.putText(
        panel,
        "3D trajectory projections",
        (20, 30),
        cv.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    if points is None or len(points) < 2:
        cv.putText(
            panel,
            "Not enough points",
            (20, 70),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )
        return panel

    pts = np.asarray(points, dtype=np.float32)

    # Центры трёх маленьких графиков
    centers = [
        (width // 2, 170),  # XZ
        (width // 2, 390),  # XY
        (width // 2, 610),  # ZY
    ]

    titles = [
        "Top view: X-Z",
        "Front view: X-Y",
        "Side view: Z-Y",
    ]

    # Какие оси брать для каждой проекции
    projections = [
        (0, 2),  # XZ
        (0, 1),  # XY
        (2, 1),  # ZY
    ]

    for idx, ((cx, cy), title, (a, b)) in enumerate(zip(centers, titles, projections)):
        cv.putText(
            panel,
            title,
            (20, cy - 90),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1
        )

        # Оси
        cv.line(panel, (cx - 180, cy), (cx + 180, cy), (80, 80, 80), 1)
        cv.line(panel, (cx, cy - 80), (cx, cy + 80), (80, 80, 80), 1)

        prev = None

        for p in pts:
            u = int(cx + p[a] * scale)
            v = int(cy - p[b] * scale)

            if 0 <= u < width and 0 <= v < height:
                if prev is not None:
                    cv.line(panel, prev, (u, v), (0, 255, 255), 2)

                prev = (u, v)

        # Последняя точка
        last = pts[-1]
        u = int(cx + last[a] * scale)
        v = int(cy - last[b] * scale)

        if 0 <= u < width and 0 <= v < height:
            cv.circle(panel, (u, v), 5, (0, 0, 255), -1)

    last = pts[-1]

    cv.putText(
        panel,
        f"X={last[0]:.2f} m",
        (20, height - 70),
        cv.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1
    )

    cv.putText(
        panel,
        f"Y={last[1]:.2f} m",
        (20, height - 45),
        cv.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1
    )

    cv.putText(
        panel,
        f"Z={last[2]:.2f} m",
        (20, height - 20),
        cv.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1
    )

    return panel