import torch

import numpy as np


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def is_overlap(rec1, rec2) -> bool:
    minx = max(int(rec1[0]), int(rec2[0]))
    miny = max(int(rec1[1]), int(rec2[1]))
    maxx = min(int(rec1[2]), int(rec2[2]))
    maxy = min(int(rec1[3]), int(rec2[3]))

    if maxx > minx and maxy > miny:
        return True
