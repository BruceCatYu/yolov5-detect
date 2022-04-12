# YOLOv5 detect  

🚀simple yolov5 wrapper for detect(onnx, dnn, openvino)  

## Installation  
```bash
pip install yolov5-detect
```

## Example  
```python
import cv2

from pathlib import Path
from yolov5detect import detect, annotation

output = Path("output dir")
if not output.exists():
    output.mkdir()

yolo = detect.YoloDetect("test.onnx", "test.yaml", 1000)
names = yolo.get_names()

root = Path("images dir")

for item in root.rglob("*.jpg"):  # detect all images and save label results
    img = cv2.imread(str(item))
    det = yolo.detect(img.copy(), 0.4)
    if len(det) > 0:
        annotator = annotation.Annotator(img.copy())
        for *xyxy, conf, cls in det:
            annotator.box_label(xyxy, f"{names[int(cls)]} {conf:.2f}")
        cv2.imwrite(annotator.result(), str(output / item.name))
```

## Reference  
* https://github.com/ultralytics/yolov5
* https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch
