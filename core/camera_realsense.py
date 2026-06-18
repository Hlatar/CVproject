import pyrealsense2 as rs
import numpy as np

class RealSenseCameraSystem:
    def __init__(self, width=1280, height=720, fps=30):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)
        self.color_intrinsics = None
        for _ in range(30):
            frames = self.pipeline.wait_for_frames()
            c = frames.get_color_frame()
            if c:
                self.color_intrinsics = c.get_profile().as_video_stream_profile().get_intrinsics()
                break
        if self.color_intrinsics is None:
            raise RuntimeError("No color intrinsics")

    def get_color_intrinsics(self):
        K = np.array([[self.color_intrinsics.fx, 0, self.color_intrinsics.ppx],
                      [0, self.color_intrinsics.fy, self.color_intrinsics.ppy],
                      [0, 0, 1]], dtype=np.float64)
        return K

    def read(self):
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)
        color = aligned.get_color_frame()
        depth = aligned.get_depth_frame()
        if not color or not depth:
            return False, None, None
        return True, np.asanyarray(color.get_data()), depth

    def release(self):
        self.pipeline.stop()