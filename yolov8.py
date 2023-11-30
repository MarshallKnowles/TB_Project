from ultralytics import YOLO
import torch
from pathlib import Path

torch.cuda.set_device(0)

# Loading a model in
model = YOLO("/home/arthurxu/Documents/Coding/TB_Project/runs/detect/train36/weights/best.pt")


#  Train the model
#  results = model.train(data='yolov8.yaml', epochs=120, imgsz=512, batch=64, verbose=True, resume = True)

#  Validate a model
metrics = model.val()  # no arguments needed, dataset and settings remembered
metrics.box.map    # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps   # a list contains map50-95 of each category