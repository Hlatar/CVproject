import cv2 as cv
import numpy as np


def triangulate_point(pt1, pt2, calib):
    p1 = np.array(pt1, dtype=np.float32).reshape(1, 1, 2)
    p2 = np.array(pt2, dtype=np.float32).reshape(1, 1, 2)

    und1 = cv.undistortPoints(p1, calib.K1, calib.D1, P=calib.K1)
    und2 = cv.undistortPoints(p2, calib.K2, calib.D2, P=calib.K2)

    x1 = und1.reshape(2, 1)
    x2 = und2.reshape(2, 1)

    X_h = cv.triangulatePoints(calib.P1, calib.P2, x1, x2)
    X = (X_h[:3] / X_h[3]).reshape(3)
    return X

def to_cartesian_coordinates(X_raw, mode="standard"):
    """
    Преобразование в прямоугольную систему координат (X влево-вправо, Y вверх-вниз, Z прямо)
    
    mode:
        "standard" - стандартное преобразование (меняет местами X и Z)
        "swap_xz"  - X и Z меняются местами
        "full_swap" - полная перенастройка осей
    """
    x, y, z = X_raw[0], X_raw[1], X_raw[2]
    
    if mode == "standard":
        # Меняем X и Z местами (Z исходный становится X, X исходный становится Z)
        return np.array([z, y, x])
    
    elif mode == "swap_xz":
        # Только X и Z меняются местами
        return np.array([z, y, x])
    
    elif mode == "full_swap":
        # X ← Z, Y ← X, Z ← Y
        return np.array([z, x, y])
    
    elif mode == "invert_y":
        # Инвертируем Y (если вверх-вниз перевёрнуты)
        return np.array([x, -y, z])
    
    else:
        return X_raw

def get_cartesian_transform_for_drone():
    """
    Возвращает рекомендуемое преобразование для дрона.
    Попробуйте эти варианты по очереди:
    
    Вариант 1 (рекомендуется): to_cartesian_coordinates(X, "standard")
    Вариант 2: to_cartesian_coordinates(X, "full_swap")  
    Вариант 3: to_cartesian_coordinates(X, "swap_xz")
    Вариант 4: to_cartesian_coordinates(X, "invert_y")
    """
    return "standard"