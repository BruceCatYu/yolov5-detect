import os
import cv2
import torch

import numpy as np

from .utils.vision import clip_coords, check_img_size, non_max_suppression
from .models.backend import DetectMultiBackend


def select_device(device='', batch_size=0):
    s = f'🚀 torch {torch.__version__} '  # string
    device = str(device).strip().lower().replace(
        'cuda:', '')  # to string, 'cuda:0' to '0'
    cpu = device == 'cpu'
    if cpu:
        # force torch.cuda.is_available() = False
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    elif device:  # non-cpu device requested
        # set environment variable - must be before assert is_available()
        os.environ['CUDA_VISIBLE_DEVICES'] = device
        assert torch.cuda.is_available() and torch.cuda.device_count() >= len(device.replace(',', '')), \
            f"Invalid CUDA '--device {device}' requested, use '--device cpu' or pass valid CUDA device(s)"

    cuda = not cpu and torch.cuda.is_available()
    if cuda:
        # range(torch.cuda.device_count())  # i.e. 0,1,6,7
        devices = device.split(',') if device else '0'
        n = len(devices)  # device count
        if n > 1 and batch_size > 0:  # check batch_size is divisible by device_count
            assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'
        space = ' ' * (len(s) + 1)
        for i, d in enumerate(devices):
            p = torch.cuda.get_device_properties(i)
            # bytes to MB
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / 1024 ** 2:.0f}MiB)\n"
    else:
        s += 'CPU\n'
    return torch.device('cuda:0' if cuda else 'cpu')


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - \
        new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / \
            shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right,
                            cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0],
                   img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / \
            2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


class YoloDetect:
    def __init__(self, weights: str, data: str, maxDetect: int, imageSize: list = [640, 640]) -> None:
        self.max_det = maxDetect  # maximum detections per image
        self.device = select_device('cpu')
        self.model = DetectMultiBackend(weights, device=self.device, data=data)
        self.stride, self.names = self.model.stride, self.model.names
        self.imgsz = check_img_size(
            imageSize, s=self.stride)  # check image size
        # Run inference
        self.model.warmup(imgsz=(1, 3, *self.imgsz))  # warmup

    def get_names(self):
        return self.names

    def detect(self, image, conf_thres: float, iou_thres: float = 0.45):
        img = letterbox(image, self.imgsz, stride=self.stride, auto=False)[0]

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        imgTorch = torch.from_numpy(img).to(self.device)
        imgTorch = imgTorch.float()  # uint8 to fp16/32
        imgTorch /= 255  # 0 - 255 to 0.0 - 1.0
        if len(imgTorch.shape) == 3:
            imgTorch = imgTorch[None]  # expand for batch dim
        # Inference
        pred = self.model(imgTorch)

        # NMS
        pred = non_max_suppression(
            pred, conf_thres, iou_thres, None, False, max_det=self.max_det)

        # Process predictions
        det = pred[0]
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(
                imgTorch.shape[2:], det[:, :4], image.shape).round()
            return det
        return []
