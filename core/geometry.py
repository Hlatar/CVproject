import cv2 as cv
import numpy as np


def triangulate_point(pt1, pt2, calib):
    """
    Триангулирует 3D-точку по двум 2D-точкам.

    pt1 — центр объекта на первой камере: (x, y)
    pt2 — центр объекта на второй камере: (x, y)

    Возвращает X_cam в системе координат первой камеры OpenCV:
        X — вправо
        Y — вниз
        Z — вперёд от камеры
    """

    p1 = np.array(pt1, dtype=np.float64).reshape(1, 1, 2)
    p2 = np.array(pt2, dtype=np.float64).reshape(1, 1, 2)

    # Переводим пиксельные координаты в нормализованные координаты камеры
    und1 = cv.undistortPoints(p1, calib.K1, calib.D1)
    und2 = cv.undistortPoints(p2, calib.K2, calib.D2)

    x1 = und1.reshape(2, 1)
    x2 = und2.reshape(2, 1)

    X_h = cv.triangulatePoints(
        calib.P1_norm,
        calib.P2_norm,
        x1,
        x2
    )

    X = (X_h[:3] / X_h[3]).reshape(3)

    return X.astype(np.float32)


def camera_to_world_coordinates(X_cam):
    """
    Преобразует координаты OpenCV-камеры в удобную мировую систему.

    OpenCV camera coordinates:
        X — вправо
        Y — вниз
        Z — вперёд

    Наша world-система:
        X — вправо
        Y — вверх
        Z — вперёд

    Поэтому меняем только знак Y.
    """

    x, y, z = X_cam

    return np.array([
        x,
        -y,
        z
    ], dtype=np.float32)


def is_valid_3d_point(X, min_z=0.05, max_z=100.0):
    """
    Проверяет, что 3D-точка физически адекватная.
    """

    if X is None:
        return False

    X = np.asarray(X)

    if not np.all(np.isfinite(X)):
        return False

    z = X[2]

    if z < min_z or z > max_z:
        return False

    return True

# def depth_pixel_to_3d(x, y, depth, intrinsics):
#     """Преобразует пиксель (x,y) и значение глубины (в метрах) в 3D точку в системе камеры."""
#     fx = intrinsics.fx
#     fy = intrinsics.fy
#     ppx = intrinsics.ppx
#     ppy = intrinsics.ppy
#     X = (x - ppx) * depth / fx
#     Y = (y - ppy) * depth / fy
#     Z = depth
#     return np.array([X, Y, Z], dtype=np.float32)

def depth_pixel_to_3d(x, y, depth, K):
    """
    Преобразует пиксель (x,y) и значение глубины (в метрах) в 3D точку.
    K - матрица интринсиков 3x3 (fx, fy, cx, cy).
    """
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    X = (x - cx) * depth / fx
    Y = (y - cy) * depth / fy
    Z = depth
    return np.array([X, Y, Z], dtype=np.float32)

def get_depth_at_center(depth_frame, cx, cy, radius=3):
    """Возвращает медианную глубину в квадрате radius вокруг (cx, cy)."""
    import numpy as np
    depth_image = np.asanyarray(depth_frame.get_data())
    h, w = depth_image.shape
    x1 = max(0, cx - radius)
    x2 = min(w, cx + radius + 1)
    y1 = max(0, cy - radius)
    y2 = min(h, cy + radius + 1)
    patch = depth_image[y1:y2, x1:x2]
    valid = patch[patch > 0]
    if len(valid) == 0:
        return 0.0
    return np.median(valid) / 1000.0  # в метрах