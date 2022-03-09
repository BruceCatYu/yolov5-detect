import cv2


class VideoReader:
    def __init__(self, uri) -> None:
        cap = cv2.VideoCapture(uri)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        raw_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        raw_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print("width:", raw_width, "height:", raw_height, "fps:", fps,
              "frames:", frame_count, "total time:", frame_count/fps)
        self.__dict__.update(locals())

    def isOpen(self) -> bool:
        return self.cap.isOpened()

    def read(self):
        return self.cap.read()

    def release(self) -> None:
        self.cap.release()
