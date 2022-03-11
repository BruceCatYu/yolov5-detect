import cv2
import time

from pathlib import Path
from .annotation import Annotator


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

    def annotateTime(self, frame, sec):
        return cv2.putText(frame.copy(), str(sec), (20, 20), thickness=2)

    def saveFrame(self, output, showTime=True, generateName=lambda x: str(x)+".png", filter=lambda x: True, interval=10, winName="save frames"):
        output = Path(output)
        if not output.exists:
            output.mkdir()

        startTime = time.time()
        i = 0
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret is not True:
                break
            cv2.imshow(winName, Annotator(frame.copy(),line_width=2).text())
            if filter(i):
                cv2.imwrite(str(output/generateName(i)), frame)
            i += 1

            c = cv2.waitKey(interval)
            if c == ord('q') or c == 27:
                break
        self.release()
        cv2.destroyAllWindows()
