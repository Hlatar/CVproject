import cv2
import numpy as np

class KalmanFilter3D:
    def __init__(self, dt=1.0, process_noise=0.1, measurement_noise=0.5):
        self.dt = dt
        self.kf = cv2.KalmanFilter(6, 3)  # состояние: x,y,z,vx,vy,vz
        self.kf.measurementMatrix = np.array([[1,0,0,0,0,0],
                                              [0,1,0,0,0,0],
                                              [0,0,1,0,0,0]], np.float32)
        self.kf.transitionMatrix = np.array([[1,0,0,dt,0,0],
                                             [0,1,0,0,dt,0],
                                             [0,0,1,0,0,dt],
                                             [0,0,0,1,0,0],
                                             [0,0,0,0,1,0],
                                             [0,0,0,0,0,1]], np.float32)
        self.kf.processNoiseCov = np.eye(6, dtype=np.float32) * process_noise
        self.kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * measurement_noise
        self.kf.errorCovPost = np.eye(6, dtype=np.float32)
        self.kf.statePost = np.zeros((6,1), np.float32)
        self.initialized = False

    def update(self, world_point):
        measurement = np.array(world_point, np.float32).reshape(3,1)
        if not self.initialized:
            self.kf.statePost[:3] = measurement
            self.initialized = True
        else:
            self.kf.correct(measurement)

    def predict(self):
        if not self.initialized:
            return None
        predicted = self.kf.predict()
        return predicted[:3].flatten()

    def get_state(self):
        if not self.initialized:
            return None
        return self.kf.statePost[:3].flatten()

    def reset(self):
        self.initialized = False