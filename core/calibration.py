import os
import cv2 as cv
import numpy as np


class StereoCalibration:
    def __init__(self, calib_file: str):
        if not os.path.exists(calib_file):
            raise FileNotFoundError(
                f"Файл {calib_file} не найден. "
                "Сначала запусти скрипт стерео-калибровки."
            )

        calib = np.load(calib_file)
        self.K1 = calib["K1"]
        self.D1 = calib["D1"]
        self.K2 = calib["K2"]
        self.D2 = calib["D2"]
        self.R = calib["R"]
        self.T = calib["T"]

        self.P1 = self.K1 @ np.hstack([np.eye(3), np.zeros((3, 1))])
        self.P2 = self.K2 @ np.hstack([self.R, self.T])

        self.P1_norm = np.hstack([
            np.eye(3, dtype=np.float64),
            np.zeros((3, 1), dtype=np.float64)
        ])

        self.P2_norm = np.hstack([
            self.R.astype(np.float64),
            self.T.astype(np.float64).reshape(3, 1)
        ])

        self.rvec1 = np.zeros((3, 1), dtype=np.float32)
        self.tvec1 = np.zeros((3, 1), dtype=np.float32)
        self.rvec2, _ = cv.Rodrigues(self.R.astype(np.float32))
        self.tvec2 = self.T.astype(np.float32).reshape(3, 1)

        # в calibration.py добавить метод
    def set_intrinsics(self, K1_new, K2_new):
        self.K1 = K1_new
        self.K2 = K2_new
        self.P1 = self.K1 @ np.hstack([np.eye(3), np.zeros((3,1))])
        self.P2 = self.K2 @ np.hstack([self.R, self.T])

