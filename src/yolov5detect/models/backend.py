import yaml
import torch
import onnxruntime

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
        # net = cv2.dnn.readNetFromONNX(weights)

        cuda = torch.cuda.is_available()
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else [
            'CPUExecutionProvider']
        session = onnxruntime.InferenceSession(weights, providers=providers)
        self.__dict__.update(locals())  # assign all variables to self

    def forward(self, im):
        # YOLOv5 MultiBackend inference
        im = im.cpu().numpy()  # torch to numpy

        # self.net.setInput(im)
        # y = self.net.forward()
        y = self.session.run([self.session.get_outputs()[0].name], {
                             self.session.get_inputs()[0].name: im})[0]

        return torch.tensor(y) if isinstance(y, np.ndarray) else y

    def warmup(self, imgsz=(1, 3, 640, 640)):
        # Warmup model by running inference once
        # only warmup GPU models
        if isinstance(self.device, torch.device) and self.device.type != 'cpu':
            im = torch.zeros(
                *imgsz).to(self.device).type(torch.float)  # input image
            self.forward(im)  # warmup
