import numpy as np
import cv2
from config import REALSENSE_FPS


class KalmanFilter3D:
    """
    Линейный фильтр Калмана для 3D-трекинга (Constant Velocity + Gravity).
    Состояние: [x, y, z, vx, vy, vz]^T
    Измерение: [x, y, z]^T
    """

    def __init__(
        self,
        dt=1.0 / REALSENSE_FPS,
        process_noise=15.0,
        measurement_noise=0.0025,
        z_noise_factor=4.0,
        enable_gravity=False,
    ):
        self.dt = float(dt)
        self.enable_gravity = enable_gravity
        self.g = 9.81
        self.accel_var = float(process_noise)

        self.kf = cv2.KalmanFilter(6, 3)

        # H — Матрица измерений
        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]],
            np.float32,
        )

        # F — Матрица перехода состояний
        self.kf.transitionMatrix = np.array(
            [
                [1, 0, 0, self.dt, 0, 0],
                [0, 1, 0, 0, self.dt, 0],
                [0, 0, 1, 0, 0, self.dt],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ],
            np.float32,
        )

        # B — Матрица управления для гравитации (ось Y направлена вверх)
        if enable_gravity:
            self.kf.controlMatrix = np.array(
                [[0], [-0.5 * self.dt * self.dt], [0], [0], [-self.dt], [0]],
                np.float32,
            )
        else:
            self.kf.controlMatrix = np.zeros((6, 1), np.float32)

        # Q — Матрица шума процесса
        self.kf.processNoiseCov = self._build_process_noise()

        # R — Матрица шума измерений
        R_xy = float(measurement_noise)
        R_z = R_xy * float(z_noise_factor)
        self._R = np.diag([R_xy, R_xy, R_z]).astype(np.float32)
        self.kf.measurementNoiseCov = self._R.copy()

        self.initialized = False

    def _build_process_noise(self):
        dt = self.dt
        q = self.accel_var
        dt2 = dt**2
        dt3 = (dt**3) / 2.0
        dt4 = (dt**4) / 4.0

        Q = np.array(
            [
                [dt4, 0, 0, dt3, 0, 0],
                [0, dt4, 0, 0, dt3, 0],
                [0, 0, dt4, 0, 0, dt3],
                [dt3, 0, 0, dt2, 0, 0],
                [0, dt3, 0, 0, dt2, 0],
                [0, 0, dt3, 0, 0, dt2],
            ],
            dtype=np.float32,
        )
        return Q * q

    def _init_cov(self):
        R_xy = self._R[0, 0]
        R_z = self._R[2, 2]
        # Дисперсия скорости снижена до 25.0 (std = 5 м/с), чтобы избежать ложных сбросов
        self.kf.errorCovPost = np.diag(
            [R_xy, R_xy, R_z, 25.0, 25.0, 25.0]
        ).astype(np.float32)

    def step(self, measurement=None):
        """
        Единственная точка входа. Вызывать 1 раз за кадр.
        """
        if not self.initialized:
            if measurement is None:
                return None
            self.kf.statePost[:3] = np.asarray(measurement, np.float32).reshape(
                3, 1
            )
            self.kf.statePost[3:] = 0.0
            self._init_cov()
            self.initialized = True
            return self.kf.statePost[:3].flatten()

        # 1) Predict
        if self.enable_gravity:
            self.kf.predict(np.array([[self.g]], np.float32))
        else:
            self.kf.predict()

        # 2) Correct / Sync
        if measurement is not None:
            z = np.asarray(measurement, np.float32).reshape(3, 1)
            self.kf.correct(z)
        else:
            # ИСПРАВЛЕНИЕ: Копируем предсказанное состояние в statePost при отсутствии измерения
            self.kf.statePost = self.kf.statePre.copy()
            self.kf.errorCovPost = self.kf.errorCovPre.copy()

        return self.kf.statePost[:3].flatten()

    def get_state(self):
        if not self.initialized:
            return None
        return self.kf.statePost[:3].flatten()

    def get_velocity(self):
        if not self.initialized:
            return np.zeros(3, dtype=np.float32)
        return self.kf.statePost[3:].flatten()

    def predict_future_point(self, t_future):
        if not self.initialized:
            return None
        st = self.kf.statePost.flatten()
        x, y, z = st[:3]
        vx, vy, vz = st[3:]

        x_fut = x + vx * t_future
        z_fut = z + vz * t_future
        if self.enable_gravity:
            y_fut = y + vy * t_future - 0.5 * self.g * (t_future**2)
        else:
            y_fut = y + vy * t_future

        return np.array([x_fut, y_fut, z_fut], dtype=np.float32)

    def reset(self):
        self.initialized = False
        self.kf.statePost[:] = 0.0
        self.kf.statePre[:] = 0.0
        self.kf.errorCovPost = np.eye(6, dtype=np.float32) * 10.0
        self.kf.errorCovPre = np.eye(6, dtype=np.float32) * 10.0