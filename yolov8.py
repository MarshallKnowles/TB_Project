from ultralytics import YOLO
import torch, os
import cv2

torch.cuda.set_device(0)

# Set task and model
task = "train"
model = YOLO("add_model_here.pt")


if task == "train":
  results = model.train(data='yolov8.yaml', epochs=120, imgsz=512, batch=64, verbose=True)

elif task == "val":
  metrics = model.val()  # no arguments needed, dataset and settings remembered
  metrics.box.map    # map50-95
  metrics.box.map50  # map50
  metrics.box.map75  # map75
  metrics.box.maps   # a list contains map50-95 of each category

elif task == "detect":
  file = 'add_image_here.png'
  results = model([file])  # return a list of Results objects

  # Read the image
  image = cv2.imread(file) 

  for result in results:
    boxes = result.boxes
    for box in boxes:
      data = box.xywh[0].tolist()
      data = [int(c) for c in data]
      x, y, w, h = data
      cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
      cv2.putText(image, "Confidence: " + str(box.conf.item())[:5], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36,255,12), 2)

  cv2.imshow("TB", image)
  cv2.waitKey(0) 
  cv2.destroyAllWindows()