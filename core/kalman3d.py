import numpy as np
import cv2
from config import REALSENSE_FPS

class KalmanFilter3D:
    def __init__(self, dt=1.0/REALSENSE_FPS, process_noise=0.1, measurement_noise=0.2, enable_gravity=False):
        self.dt = dt
        self.enable_gravity = enable_gravity
        self.g = 9.81  # Ускорение свободного падения (м/с^2) по ось Y (вверх)

        # Вектор состояния: [x, y, z, vx, vy, vz]
        self.kf = cv2.KalmanFilter(6, 3)
        self.kf.measurementMatrix = np.array([
            [1,0,0,0,0,0],
            [0,1,0,0,0,0],
            [0,0,1,0,0,0]
        ], np.float32)

        self.kf.transitionMatrix = np.array([
            [1,0,0,dt,0, 0],
            [0,1,0,0, dt,0],
            [0,0,1,0, 0, dt],
            [0,0,0,1, 0, 0],
            [0,0,0,0, 1, 0],
            [0,0,0,0, 0, 1]
        ], np.float32)

        self.kf.processNoiseCov = np.eye(6, dtype=np.float32) * process_noise
        self.kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * measurement_noise
        self.kf.errorCovPost = np.eye(6, dtype=np.float32)
        self.kf.statePost = np.zeros((6,1), np.float32)
        self.initialized = False

    def update(self, world_point):
        measurement = np.array(world_point, np.float32).reshape(3, 1)
        if not self.initialized:
            self.kf.statePost[:3] = measurement
            self.kf.statePost[3:] = 0.0
            self.initialized = True
        else:
            self.kf.correct(measurement)

    def predict(self):
        if not self.initialized:
            return None
        predicted = self.kf.predict()
        if self.enable_gravity:
            # Моделирование влияния падения под действием гравитации
            self.kf.statePost[1] -= 0.5 * self.g * (self.dt ** 2)
            self.kf.statePost[4] -= self.g * self.dt
        return predicted[:3].flatten()

    def get_state(self):
        if not self.initialized:
            return None
        return self.kf.statePost[:3].flatten()

    def get_velocity(self):
        if not self.initialized:
            return np.zeros(3, dtype=np.float32)
        return self.kf.statePost[3:].flatten()

    def predict_future_point(self, t_future):
        """ Предсказание координат объекта через t_future секунд вперед """
        if not self.initialized:
            return None
        st = self.kf.statePost.flatten()
        x, y, z = st[:3]
        vx, vy, vz = st[3:]

        x_fut = x + vx * t_future
        z_fut = z + vz * t_future
        if self.enable_gravity:
            y_fut = y + vy * t_future - 0.5 * self.g * (t_future ** 2)
        else:
            y_fut = y + vy * t_future

        return np.array([x_fut, y_fut, z_fut], dtype=np.float32)

    def reset(self):
        self.initialized = False