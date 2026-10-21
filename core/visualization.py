import cv2 as cv
import numpy as np

CLASS_COLORS = {
    'drone': (0, 255, 0),    # Зеленый
    'ball': (0, 165, 255)    # Оранжевый
}

def draw_detection_overlay(img, det, camera_label="Cam"):
    if det is None:
        return

    x1, y1, x2, y2 = det["bbox"]
    cx, cy = det["center"]
    conf = det["conf"]
    class_name = det.get("class_name", "object")

    color = CLASS_COLORS.get(class_name, (255, 255, 255))

    cv.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv.circle(img, (int(round(cx)), int(round(cy))), 5, (0, 0, 255), -1)

    conf_text = f"{conf:.2f}" if conf is not None else "track"
    label_str = f"{class_name.upper()} {conf_text}"

    cv.putText(
        img,
        label_str,
        (x1, max(20, y1 - 10)),
        cv.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv.LINE_AA
    )

def draw_dual_trajectory_panel(ball_points, drone_points, rec_ball=False, rec_drone=False, width=500, height=480, scale=80):
    """
    Рисует правую панель с совмещенными проекциями траекторий МЯЧА (Оранжевый) и ДРОНА (Зеленый).
    """
    panel = np.zeros((height, width, 3), dtype=np.uint8)

    # Статус записи на правой панели
    ball_rec_str = "REC ON" if rec_ball else "REC OFF"
    ball_rec_col = (0, 165, 255) if rec_ball else (120, 120, 120)
    drone_rec_str = "REC ON" if rec_drone else "REC OFF"
    drone_rec_col = (0, 255, 0) if rec_drone else (120, 120, 120)

    cv.putText(panel, f"BALL: {ball_rec_str}", (20, 25), cv.FONT_HERSHEY_SIMPLEX, 0.55, ball_rec_col, 2, cv.LINE_AA)
    cv.putText(panel, f"DRONE: {drone_rec_str}", (260, 25), cv.FONT_HERSHEY_SIMPLEX, 0.55, drone_rec_col, 2, cv.LINE_AA)

    ball_pts = np.asarray(ball_points, dtype=np.float32) if ball_points is not None and len(ball_points) > 0 else None
    drone_pts = np.asarray(drone_points, dtype=np.float32) if drone_points is not None and len(drone_points) > 0 else None

    centers = [
        (width // 2, 110),
        (width // 2, 240),
        (width // 2, 370),
    ]

    titles = ["Top view: X-Z", "Front view: X-Y", "Side view: Z-Y"]
    projections = [(0, 2), (0, 1), (2, 1)]

    for (cx, cy), title, (a, b) in zip(centers, titles, projections):
        cv.putText(panel, title, (20, cy - 45), cv.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv.LINE_AA)
        cv.line(panel, (cx - 180, cy), (cx + 180, cy), (50, 50, 50), 1)
        cv.line(panel, (cx, cy - 35), (cx, cy + 35), (50, 50, 50), 1)

        # Отрисовка траектории МЯЧА (Оранжевый)
        if ball_pts is not None and len(ball_pts) >= 2:
            prev = None
            for p in ball_pts:
                u = int(cx + p[a] * scale)
                v = int(cy - p[b] * scale)
                if 0 <= u < width and 0 <= v < height:
                    if prev is not None:
                        cv.line(panel, prev, (u, v), (0, 165, 255), 2)
                    prev = (u, v)
            last = ball_pts[-1]
            u_l = int(cx + last[a] * scale)
            v_l = int(cy - last[b] * scale)
            if 0 <= u_l < width and 0 <= v_l < height:
                cv.circle(panel, (u_l, v_l), 4, (0, 0, 255), -1)

        # Отрисовка траектории ДРОНА (Зеленый)
        if drone_pts is not None and len(drone_pts) >= 2:
            prev = None
            for p in drone_pts:
                u = int(cx + p[a] * scale)
                v = int(cy - p[b] * scale)
                if 0 <= u < width and 0 <= v < height:
                    if prev is not None:
                        cv.line(panel, prev, (u, v), (0, 255, 0), 2)
                    prev = (u, v)
            last = drone_pts[-1]
            u_l = int(cx + last[a] * scale)
            v_l = int(cy - last[b] * scale)
            if 0 <= u_l < width and 0 <= v_l < height:
                cv.circle(panel, (u_l, v_l), 4, (255, 0, 0), -1)

    # Статистика снизу
    n_ball = len(ball_pts) if ball_pts is not None else 0
    n_drone = len(drone_pts) if drone_pts is not None else 0

    cv.putText(panel, f"Ball points: {n_ball}", (20, height - 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1, cv.LINE_AA)
    cv.putText(panel, f"Drone points: {n_drone}", (260, height - 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv.LINE_AA)

    return panel