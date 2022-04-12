import yaml
import torch

import numpy as np
import torch.nn as nn

from pathlib import Path


class DetectMultiBackend(nn.Module):
    # YOLOv5 MultiBackend class for python inference on various backends
    def __init__(self, weights='yolov5s.pt', device=None, data=None,type="onnx"):
        super().__init__()
        stride, names = 64, [
            f'class{i}' for i in range(1000)]  # assign defaults
        if data:  # data.yaml path (optional)
            with open(data, errors='ignore') as f:
                names = yaml.safe_load(f)['names']  # class names
        # net = cv2.dnn.readNetFromONNX(weights)

        if type=="onnx":
            import onnxruntime
            cuda = torch.cuda.is_available()
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else [
            'CPUExecutionProvider']
            session = onnxruntime.InferenceSession(weights, providers=providers)
        elif type=="openvino":
            import openvino.inference_engine as ie
            core = ie.IECore()
            network = core.read_network(model=weights,weights=Path(weights).with_suffix(".bin"))
            executable_network = core.load_network(network,device_name="CPU",num_requests=1)
        self.__dict__.update(locals())  # assign all variables to self

    def forward(self, im):
        # YOLOv5 MultiBackend inference
        if self.type=="onnx":
            im = im.cpu().numpy()  # torch to numpy

            # self.net.setInput(im)
            # y = self.net.forward()
            y = self.session.run([self.session.get_outputs()[0].name], {self.session.get_inputs()[0].name: im})[0]
        elif self.type=="openvino":
            im = im.cpu().numpy()  # FP32
            desc=self.ie.TensorDesc(precision="FP32",dims=im.shape,layout="NCHW")
            request=self.executable_network.requests[0]
            request.set_blob(blob_name="images",blob=self.ie.Blob(desc,im))
            request.infer()
            y = request.output_blobs["output"].buffer

        return torch.tensor(y) if isinstance(y, np.ndarray) else y

    def warmup(self, imgsz=(1, 3, 640, 640)):
        # Warmup model by running inference once
        # only warmup GPU models
        if isinstance(self.device, torch.device) and self.device.type != 'cpu':
            im = torch.zeros(
                *imgsz).to(self.device).type(torch.float)  # input image
            self.forward(im)  # warmup
