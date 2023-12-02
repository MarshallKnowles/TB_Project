import json
import numpy as np

def get_bbox_from_polygon(points):
    x_coordinates, y_coordinates = zip(*points)
    bbox = [min(x_coordinates), min(y_coordinates),
            max(x_coordinates) - min(x_coordinates),
            max(y_coordinates) - min(y_coordinates)]
    return bbox

def create_annotation_v3(annotation_id, image_id, region, category_mapping):
    shape_attributes = region.get("shape_attributes", {})
    if "all_points_x" not in shape_attributes or "all_points_y" not in shape_attributes:
        return None  


    points = list(zip(shape_attributes["all_points_x"], shape_attributes["all_points_y"]))
    bbox = get_bbox_from_polygon(points)
    area = bbox[2] * bbox[3]  


    region_category = next(iter(region["region_attributes"]), "default_category")
    category_id = category_mapping.get(region_category, 1) 

    return {
        "id": float(annotation_id),  
        "image_id": float(image_id),  
        "category_id": int(category_id), 
        "bbox": [float(b) for b in bbox],  
        "segmentation": [list(np.ravel(points))], 
        "area": float(area),  
        "iscrowd": 0
    }

with open('TBX11K/annotations/json/Annotations_AllinOne_json.json', 'r') as file_annotations:
    annotations_allinone = json.load(file_annotations)

with open('TBX11K/annotations/json/TBX11K_train.json', 'r') as file:
    tbx11k_format = json.load(file)

category_mapping = {category["name"]: category["id"] for category in tbx11k_format["categories"]}

# Initialize COCO format dictionary with TBX11K categories
coco_format = {
    "images": [],  # Images will be added from the annotations data
    "annotations": [],
    "categories": tbx11k_format["categories"]
}

# Process each entry in the annotations data
original_image_width = 2048
original_image_height = 1992
x_scale = 512 / original_image_width
y_scale = 512 / original_image_height
annotation_id = 1  

for filename, image_data in annotations_allinone.items():
    image_id = annotation_id  
    coco_format["images"].append({
        "id": image_id,
        "file_name": filename,
        "height": 512,
        "width": 512
    })

    # Add annotations for this image
    for region in image_data["regions"]:
        annotation = create_annotation_v3(annotation_id, image_id, region, category_mapping)
        if annotation:
            annotation["segmentation"][0][::2] = [i * x_scale for i in annotation["segmentation"][0][::2]]
            annotation["segmentation"][0][1::2] = [i * y_scale for i in annotation["segmentation"][0][1::2]]
            coco_format["annotations"].append(annotation)
            annotation_id += 1

#print(coco_format_sample)
#print("hi")


class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)): 
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    
with open('TBX11K/annotations/json/converting_poly_to_MSCOCO.json', 'w') as file:
    json.dump(coco_format, file, cls=NumpyEncoder)


