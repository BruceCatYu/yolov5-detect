import cv2
import yaml
import torch

import numpy as np
import torch.nn as nn


class DetectMultiBackend(nn.Module):
    # YOLOv5 MultiBackend class for python inference on various backends
    def __init__(self, weights='yolov5s.pt', device=None, data=None):
        super().__init__()
        stride, names = 64, [
            f'class{i}' for i in range(1000)]  # assign defaults
        if data:  # data.yaml path (optional)
            with open(data, errors='ignore') as f:
                names = yaml.safe_load(f)['names']  # class names
        net = cv2.dnn.readNetFromONNX(weights)
        self.__dict__.update(locals())  # assign all variables to self

    def forward(self, im, val=False):
        # YOLOv5 MultiBackend inference
        im = im.cpu().numpy()  # torch to numpy
        self.net.setInput(im)
        y = self.net.forward()

        y = torch.tensor(y) if isinstance(y, np.ndarray) else y
        return (y, []) if val else y

    def warmup(self, imgsz=(1, 3, 640, 640)):
        # Warmup model by running inference once
        # only warmup GPU models
        if isinstance(self.device, torch.device) and self.device.type != 'cpu':
            im = torch.zeros(
                *imgsz).to(self.device).type(torch.float)  # input image
            self.forward(im)  # warmup
