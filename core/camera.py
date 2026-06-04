import cv2 as cv


class StereoCameraSystem:
    def __init__(self, cam_id_1: int, cam_id_2: int):
        self.cap1 = cv.VideoCapture(cam_id_1)
        self.cap2 = cv.VideoCapture(cam_id_2)

        self.cap1.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap1.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

        self.cap2.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap2.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

        if not self.cap1.isOpened() or not self.cap2.isOpened():
            raise RuntimeError("Ошибка открытия камер")

    def read(self):
        ret1, frame1 = self.cap1.read()
        ret2, frame2 = self.cap2.read()
        return ret1, frame1, ret2, frame2

    def release(self):
        self.cap1.release()
        self.cap2.release()