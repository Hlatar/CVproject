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