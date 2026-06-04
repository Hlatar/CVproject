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