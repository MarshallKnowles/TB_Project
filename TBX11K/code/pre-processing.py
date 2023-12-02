import json
import cv2
import numpy as np
from albumentations import Compose, Resize, RandomRotate90, Flip, BboxParams
from albumentations.pytorch.transforms import ToTensorV2

processed_images = []  

augmentations = Compose([
    Resize(512, 512),
    RandomRotate90(),
    Flip(),
], bbox_params=BboxParams(format='coco', label_fields=['category_ids']))

def transform_image_and_bbox(image, bbox, category_id, augmentations):
    augmented = augmentations(image=image, bboxes=[bbox], category_ids=[category_id])
    return augmented['image'], augmented['bboxes'][0]

def adaptive_filter(image):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    merged_lab = cv2.merge((cl, a, b))

    final_img = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2RGB)
    return final_img

# Load JSON data
with open('/content/drive/MyDrive/APS360 Project/archive/TBX11K/annotations/json/TBX11K_train.json', 'r') as file:
    data = json.load(file)

for item in data['images']:
    image_path = '/content/drive/MyDrive/APS360 Project/archive/TBX11K/imgs/'  + item['file_name']
    image_id = item['id']
    print(image_path)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image from {image_path}")
        continue 

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    image = adaptive_filter(image)
    
    # Find the annotations for the current image
    annotations = [ann for ann in data['annotations'] if ann['image_id'] == image_id]
    for annotation in annotations:
        bbox = annotation['bbox']
        category_id = annotation['category_id'] 
        augmented_image, augmented_bbox = transform_image_and_bbox(image, bbox, category_id, augmentations)
        annotation['bbox'] = augmented_bbox
    

    augmented_image = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR) 
    processed_images.append(augmented_image)
    cv2.imwrite(f'augmented_{image_path}', augmented_image)

image_data_array = np.array(processed_images)
print(image_data_array.shape)


with open('New_MSCOCO_annotations.json', 'w') as file:
    json.dump(data, file)