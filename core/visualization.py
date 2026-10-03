import cv2 as cv
import numpy as np


# ============================================================
# Basic camera / detection visualization
# ============================================================


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