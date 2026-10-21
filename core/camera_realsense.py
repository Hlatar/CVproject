import pyrealsense2 as rs
import numpy as np
from config import REALSENSE_EXPOSURE, USE_AUTO_EXPOSURE

class RealSenseCameraSystem:
    def __init__(self, width=640, height=480, fps=60):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        profile = self.pipeline.start(config)

        # Настройка экспозиции
        try:
            color_sensor = profile.get_device().query_sensors()[1]
            if USE_AUTO_EXPOSURE:
                color_sensor.set_option(rs.option.enable_auto_exposure, True)
            else:
                color_sensor.set_option(rs.option.enable_auto_exposure, False)
                color_sensor.set_option(rs.option.exposure, REALSENSE_EXPOSURE)
        except Exception as e:
            print(f"Предупреждение по настройке экспозиции: {e}")

        self.align = rs.align(rs.stream.color)
        self.color_intrinsics = None

        self.spatial = rs.spatial_filter()
        self.temporal = rs.temporal_filter()
        self.hole_filling = rs.hole_filling_filter()

        for _ in range(30):
            frames = self.pipeline.wait_for_frames()
            c = frames.get_color_frame()
            if c:
                self.color_intrinsics = c.get_profile().as_video_stream_profile().get_intrinsics()
                break
        if self.color_intrinsics is None:
            raise RuntimeError("Не удалось получить интринсики камеры")

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

        depth = self.spatial.process(depth)
        depth = self.temporal.process(depth)
        depth = self.hole_filling.process(depth)
        depth = depth.as_depth_frame()

        return True, np.asanyarray(color.get_data()), depth

    def release(self):
        self.pipeline.stop()