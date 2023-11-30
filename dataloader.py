import os
import xml.etree.ElementTree as ET
import xmltodict
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import getData
from yolov3_utils.utils import iou

# class CustomDataset(Dataset):
#     def __init__(self, image_dir, label_dir, anchors, num_classes, transform=None):
#         self.image_dir = image_dir
#         self.label_dir = label_dir
#         self.transform = transform
#         self.image_files = os.listdir(image_dir)
#         self.label_files = os.listdir(label_dir)
#         self.grid_sizes = [13, 26, 52] 
#         self.anchors = torch.tensor( 
#             anchors[0] + anchors[1] + anchors[2]) 
#         # Number of anchor boxes  
#         self.num_anchors = self.anchors.shape[0] 
#         # Number of anchor boxes per scale 
#         self.num_anchors_per_scale = self.num_anchors // 3
#         # Number of classes 
#         self.num_classes = num_classes 
#         # Ignore IoU threshold 
#         self.ignore_iou_thresh = 0.5

#     def __len__(self):
#         return len(self.image_files)

#     def __getitem__(self, idx):
#         bboxes, labels = getData.load_annotation(self.label_dir+self.label_files[idx])
#         image = Image.open(self.image_dir + self.image_files[idx])

#         targets = [torch.zeros((self.num_anchors_per_scale, s, s, 6)) 
#                    for s in self.grid_sizes] 
          
#         # Identify anchor box and cell for each bounding box 
#         for count, box in enumerate(bboxes): 
#             # Calculate iou of bounding box with anchor boxes 
#             iou_anchors = iou(torch.tensor(box[2:4]),  
#                               self.anchors,  
#                               is_pred=False) 
#             # Selecting the best anchor box 
#             anchor_indices = iou_anchors.argsort(descending=True, dim=0) 
#             x, y, width, height = box
#             class_label = labels[count]
  
#             # At each scale, assigning the bounding box to the  
#             # best matching anchor box 
#             has_anchor = [False] * 3
#             for anchor_idx in anchor_indices: 
#                 scale_idx = anchor_idx // self.num_anchors_per_scale 
#                 anchor_on_scale = anchor_idx % self.num_anchors_per_scale 
                  
#                 # Identifying the grid size for the scale 
#                 s = self.grid_sizes[scale_idx] 
                  
#                 # Identifying the cell to which the bounding box belongs 
#                 i, j = int(s * y), int(s * x) 
#                 anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0] 
                  
#                 # Check if the anchor box is already assigned 
#                 if not anchor_taken and not has_anchor[scale_idx]: 
  
#                     # Set the probability to 1 
#                     targets[scale_idx][anchor_on_scale, i, j, 0] = 1
  
#                     # Calculating the center of the bounding box relative 
#                     # to the cell 
#                     x_cell, y_cell = s * x - j, s * y - i  
  
#                     # Calculating the width and height of the bounding box  
#                     # relative to the cell 
#                     width_cell, height_cell = (width * s, height * s) 
  
#                     # Idnetify the box coordinates 
#                     box_coordinates = torch.tensor( 
#                                         [x_cell, y_cell, width_cell,  
#                                          height_cell] 
#                                     ) 
  
#                     # Assigning the box coordinates to the target 
#                     targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates 
  
#                     # Assigning the class label to the target 
#                     targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label) 
  
#                     # Set the anchor box as assigned for the scale 
#                     has_anchor[scale_idx] = True
  
#                 # If the anchor box is already assigned, check if the  
#                 # IoU is greater than the threshold 
#                 elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh: 
#                     # Set the probability to -1 to ignore the anchor box 
#                     targets[scale_idx][anchor_on_scale, i, j, 0] = -1
  
#         # Return the image and the target 
#         return image, tuple(targets)

class CustomDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_files = os.listdir(image_dir)
        self.label_files = os.listdir(label_dir)
        self.transform = transforms.Compose([
          transforms.Resize((512, 512)),
          transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        bboxes, labels = getData.load_annotation(self.label_dir+self.label_files[idx])
        image = Image.open(self.image_dir + self.image_files[idx]).convert("RGB")

        # Resize the image to 512x512 pixels
        if self.transform:
            image = self.transform(image)

        return image, bboxes, labels