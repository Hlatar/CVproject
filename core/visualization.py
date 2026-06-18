import cv2 as cv
import numpy as np


# ============================================================
# Basic camera / detection visualization
# ============================================================

def draw_cam_axes(img, K=None, label="", axis_len=60, thickness=2):
    """
    Рисует условные оси камеры на изображении.

    Цвета:
        X - красный
        Y - зелёный
        Z - синий
    """

    h, w = img.shape[:2]

    if K is not None and K.shape == (3, 3):
        origin = (int(round(K[0, 2])), int(round(K[1, 2])))
    else:
        origin = (w // 2, h // 2)

    cv.circle(img, origin, 5, (255, 255, 255), -1)
    cv.circle(img, origin, 7, (0, 0, 0), 1)

    x_end = (origin[0] + axis_len, origin[1])
    y_end = (origin[0], origin[1] + axis_len)
    z_end = (
        origin[0] - int(axis_len * 0.7),
        origin[1] - int(axis_len * 0.7),
    )

    cv.arrowedLine(img, origin, x_end, (0, 0, 255), thickness, tipLength=0.25)
    cv.arrowedLine(img, origin, y_end, (0, 255, 0), thickness, tipLength=0.25)
    cv.arrowedLine(img, origin, z_end, (255, 0, 0), thickness, tipLength=0.25)

    cv.putText(
        img,
        "X+",
        (x_end[0] + 5, x_end[1] + 5),
        cv.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2,
        cv.LINE_AA,
    )

    cv.putText(
        img,
        "Y+",
        (y_end[0] - 30, y_end[1] + 25),
        cv.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
        cv.LINE_AA,
    )

    cv.putText(
        img,
        "Z+",
        (z_end[0] - 35, z_end[1] - 10),
        cv.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 0, 0),
        2,
        cv.LINE_AA,
    )

    if label:
        cv.putText(
            img,
            f"{label} origin ({origin[0]},{origin[1]})",
            (10, h - 10),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv.LINE_AA,
        )


def draw_world_axes_at_point(
    img,
    K,
    D,
    rvec,
    tvec,
    X_world,
    axis_len_m=0.05,
    thickness=2,
):
    """
    Рисует 3D-оси в точке объекта, спроецированные на изображение камеры.

    ВАЖНО:
    X_world здесь должен быть в той системе координат,
    которая соответствует rvec/tvec данной камеры.
    """

    if X_world is None:
        return

    X_world = np.asarray(X_world, dtype=np.float32).reshape(3)

    pts3d = np.array(
        [
            X_world,
            X_world + np.array([axis_len_m, 0, 0], dtype=np.float32),
            X_world + np.array([0, axis_len_m, 0], dtype=np.float32),
            X_world + np.array([0, 0, axis_len_m], dtype=np.float32),
        ],
        dtype=np.float32,
    ).reshape(-1, 1, 3)

    imgpts, _ = cv.projectPoints(pts3d, rvec, tvec, K, D)
    imgpts = imgpts.reshape(-1, 2).astype(np.int32)

    o = tuple(imgpts[0])
    x = tuple(imgpts[1])
    y = tuple(imgpts[2])
    z = tuple(imgpts[3])

    cv.circle(img, o, 5, (255, 255, 255), -1)

    cv.arrowedLine(img, o, x, (0, 0, 255), thickness, tipLength=0.25)
    cv.arrowedLine(img, o, y, (0, 255, 0), thickness, tipLength=0.25)
    cv.arrowedLine(img, o, z, (255, 0, 0), thickness, tipLength=0.25)

    cv.putText(img, "X", x, cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv.LINE_AA)
    cv.putText(img, "Y", y, cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv.LINE_AA)
    cv.putText(img, "Z", z, cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv.LINE_AA)


def draw_detection_overlay(img, det, camera_label="Cam"):
    """
    Рисует bbox, центр объекта и confidence.
    """

    if det is None:
        cv.putText(
            img,
            f"{camera_label}: object not found",
            (10, 40),
            cv.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv.LINE_AA,
        )
        return

    x1, y1, x2, y2 = det["bbox"]
    cx, cy = det["center"]
    conf = det["conf"]
    class_name = det["class_name"]

    cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv.circle(
        img,
        (int(round(cx)), int(round(cy))),
        5,
        (0, 0, 255),
        -1,
    )

    if conf is not None:
        conf_text = f"{conf:.2f}"
    else:
        conf_text = "track"  # или "N/A"

    cv.putText(
        img,
        f"{camera_label}: {class_name} {conf_text}",
        (x1, max(20, y1 - 10)),
        cv.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv.LINE_AA,
    )


def draw_3d_text(img, X):
    """
    Рисует текущую 3D-координату объекта на изображении.
    """

    if X is None:
        cv.putText(
            img,
            "3D: N/A",
            (10, 80),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv.LINE_AA,
        )
        return

    cv.putText(
        img,
        f"3D: X={X[0]:+.3f}  Y={X[1]:+.3f}  Z={X[2]:+.3f}",
        (10, 80),
        cv.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
        cv.LINE_AA,
    )


# ============================================================
# Single-drone dashboard
# ============================================================

def resize_to_height_local(img, target_h):
    """
    Меняет размер изображения по высоте с сохранением пропорций.
    """

    h, w = img.shape[:2]
    scale = target_h / h
    target_w = int(w * scale)

    return cv.resize(img, (target_w, target_h))


def draw_info_line(panel, y, label, value, label_x=25, value_x=230):
    """
    Строка параметра в нижней панели.
    """

    label_color = (70, 110, 70)
    value_color = (60, 130, 60)

    cv.putText(
        panel,
        label,
        (label_x, y),
        cv.FONT_HERSHEY_SIMPLEX,
        0.5,
        label_color,
        1,
        cv.LINE_AA,
    )

    cv.putText(
        panel,
        value,
        (value_x, y),
        cv.FONT_HERSHEY_SIMPLEX,
        0.55,
        value_color,
        1,
        cv.LINE_AA,
    )


def create_single_drone_dashboard(
    frame1,
    frame2,
    fps=0.0,
    recording=False,
    tracking=False,
    X_world=None,
    conf1=None,
    conf2=None,
    saved_points=0,
    drone_id=1,
):
    """
    Dashboard под одного дрона.

    Структура окна:

        Camera 1 | Camera 2
        status bar
        single drone info panel
    """

    cam_h = 420
    status_h = 45
    info_h = 235

    # ---------- Верхняя часть: две камеры ----------

    frame1 = resize_to_height_local(frame1, cam_h)
    frame2 = resize_to_height_local(frame2, cam_h)

    min_w = min(frame1.shape[1], frame2.shape[1])

    frame1 = cv.resize(frame1, (min_w, cam_h))
    frame2 = cv.resize(frame2, (min_w, cam_h))

    top = np.hstack([frame1, frame2])
    total_w = top.shape[1]

    # ---------- Цвета ----------

    bg = (235, 245, 235)
    green = (70, 150, 70)
    dark_green = (40, 90, 40)
    gray = (130, 130, 130)
    red = (0, 0, 255)

    # ---------- Статусная строка ----------

    status = np.full((status_h, total_w, 3), bg, dtype=np.uint8)

    active_tracks = 1 if tracking else 0
    stereo_pairs = 1 if tracking else 0

    cv.putText(
        status,
        f"FPS  {fps:.1f}",
        (20, 28),
        cv.FONT_HERSHEY_SIMPLEX,
        0.55,
        green,
        1,
        cv.LINE_AA,
    )

    cv.putText(
        status,
        f"ACTIVE TRACKS : {active_tracks}",
        (200, 28),
        cv.FONT_HERSHEY_SIMPLEX,
        0.55,
        gray,
        1,
        cv.LINE_AA,
    )

    cv.putText(
        status,
        f"STEREO PAIRS : {stereo_pairs}",
        (480, 28),
        cv.FONT_HERSHEY_SIMPLEX,
        0.55,
        gray,
        1,
        cv.LINE_AA,
    )

    rec_text = "REC ON" if recording else "REC OFF"
    rec_color = red if recording else gray

    cv.putText(
        status,
        rec_text,
        (total_w - 140, 28),
        cv.FONT_HERSHEY_SIMPLEX,
        0.55,
        rec_color,
        1,
        cv.LINE_AA,
    )

    # ---------- Нижняя информационная панель ----------

    info = np.full((info_h, total_w, 3), bg, dtype=np.uint8)

    cv.line(info, (0, 0), (total_w, 0), green, 2)

    status_text = "TRACKING" if tracking else "LOST"
    status_color = green if tracking else red

    cv.putText(
        info,
        f"DRONE  ID {drone_id}",
        (25, 34),
        cv.FONT_HERSHEY_SIMPLEX,
        0.65,
        dark_green,
        1,
        cv.LINE_AA,
    )

    cv.putText(
        info,
        status_text,
        (total_w - 150, 34),
        cv.FONT_HERSHEY_SIMPLEX,
        0.55,
        status_color,
        1,
        cv.LINE_AA,
    )

    if X_world is not None and tracking:
        x, y, z = X_world
        distance = float(np.linalg.norm(X_world))
    else:
        x, y, z = 0.0, 0.0, 0.0
        distance = 0.0

    if conf1 is not None and conf2 is not None:
        avg_conf = (conf1 + conf2) / 2.0
    elif conf1 is not None:
        avg_conf = conf1
    elif conf2 is not None:
        avg_conf = conf2
    else:
        avg_conf = 0.0

    camera_text = "CAM1 + CAM2" if tracking else "NONE"

    row_y0 = 70
    row_h = 25

    rows = [
        ("X OFFSET", f"{x:+.3f} m"),
        ("Y OFFSET", f"{y:+.3f} m"),
        ("DEPTH Z", f"{z:.3f} m"),
        ("DISTANCE", f"{distance:.3f} m"),
        ("CONF", f"{avg_conf:.2f}"),
        ("POINTS SAVED", str(saved_points)),
        ("CAMERA", camera_text),
    ]

    for i, _ in enumerate(rows):
        if i % 2 == 0:
            cv.rectangle(
                info,
                (0, row_y0 - 17 + i * row_h),
                (total_w, row_y0 + 5 + i * row_h),
                (245, 250, 245),
                -1,
            )

    for i, (label, value) in enumerate(rows):
        draw_info_line(
            info,
            row_y0 + i * row_h,
            label,
            value,
        )

    # ---------- Финальная сборка ----------

    dashboard = np.vstack([top, status, info])

    return dashboard


# ============================================================
# Optional trajectory projections panel
# Можно оставить для отладки, если потом захочешь вернуть.
# ============================================================

def draw_trajectory_panel(points, width=500, height=720, scale=80):
    """
    Рисует панель траектории в трёх проекциях:

        XZ — вид сверху
        XY — вид спереди
        ZY — вид сбоку
    """

    panel = np.zeros((height, width, 3), dtype=np.uint8)

    cv.putText(
        panel,
        "3D trajectory projections",
        (20, 30),
        cv.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv.LINE_AA,
    )

    if points is None or len(points) < 2:
        cv.putText(
            panel,
            "Not enough points",
            (20, 70),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv.LINE_AA,
        )
        return panel

    pts = np.asarray(points, dtype=np.float32)

    centers = [
        (width // 2, 170),
        (width // 2, 390),
        (width // 2, 610),
    ]

    titles = [
        "Top view: X-Z",
        "Front view: X-Y",
        "Side view: Z-Y",
    ]

    projections = [
        (0, 2),
        (0, 1),
        (2, 1),
    ]

    for (cx, cy), title, (a, b) in zip(centers, titles, projections):
        cv.putText(
            panel,
            title,
            (20, cy - 90),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv.LINE_AA,
        )

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
        1,
        cv.LINE_AA,
    )

    cv.putText(
        panel,
        f"Y={last[1]:.2f} m",
        (20, height - 45),
        cv.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv.LINE_AA,
    )

    cv.putText(
        panel,
        f"Z={last[2]:.2f} m",
        (20, height - 20),
        cv.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv.LINE_AA,
    )

    return panel