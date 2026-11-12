# core/target_tracker.py
import cv2 as cv
import numpy as np
from core.geometry import camera_to_world_coordinates, is_valid_3d_point, depth_pixel_to_3d, get_depth_at_center
from core.kalman3d import KalmanFilter3D
from core.trajectory import Trajectory3D

class TrackedTarget:
    def __init__(self, name, color, kf_params, max_miss_frames=10):
        self.name = name
        self.color = color
        self.kf = KalmanFilter3D(**kf_params)
        self.trajectory = Trajectory3D(smoothing_window=1, max_jump=2.0)
        self.miss_counter = 0
        self.max_miss_frames = max_miss_frames
        self.is_recording = False
        self.smooth_pos = None

    def process_frame(self, detection, depth_frame, K_color):
        valid_3d = None
        if detection is not None:
            cx, cy = detection["center"]
            depth_val = get_depth_at_center(depth_frame, cx, cy)
            if depth_val > 0:
                X_cam = depth_pixel_to_3d(cx, cy, depth_val, K_color)
                X_world = camera_to_world_coordinates(X_cam)
                if is_valid_3d_point(X_world):
                    valid_3d = X_world
                    self.miss_counter = 0

        if valid_3d is not None:
            self.smooth_pos = self.kf.step(valid_3d)
        else:
            self.miss_counter += 1
            if self.miss_counter > self.max_miss_frames:
                self.kf.reset()
                self.smooth_pos = None
            else:
                self.smooth_pos = self.kf.step(None)

        if self.smooth_pos is not None:
            vel = self.kf.get_velocity()
            if np.linalg.norm(vel) > 50.0:  # Защитный сброс при аномальной скорости
                self.kf.reset()
                self.smooth_pos = None
            elif self.is_recording:
                self.trajectory.add(self.smooth_pos.copy())

        return self.smooth_pos

    def draw_info(self, img, y_offset=35):
        if self.smooth_pos is not None:
            text = f"{self.name} 3D: X={self.smooth_pos[0]:.2f} Y={self.smooth_pos[1]:.2f} Z={self.smooth_pos[2]:.2f}"
            cv.putText(img, text, (10, y_offset), cv.FONT_HERSHEY_SIMPLEX, 0.55, self.color, 2)