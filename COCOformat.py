import json, os
from PIL import Image

resize = False

# Read the annotations file
with open('/home/arthurxu/Documents/Coding/TB_Project/new_data_MSCOCO_to_json') as f:
  data = json.load(f)
  d = {}
  target_size = (512, 512)

  for image in data["images"]:
    file_path = image["file_name"]
    # Get the name of the image file after the last slash
    file_name = file_path.split("/")[-1]

    if (resize):
      try:
          # Open the image
          image_path = os.path.join("old_image_folder", file_path)
          save_path = os.path.join("new_image_folder", file_path)
          img = Image.open(image_path)
          
          # Resize the image to the target size
          img = img.resize(target_size, Image.Resampling.LANCZOS)
          
          # Save the resized image back to the same folder
          img.save(save_path)
          print(f"Resized and saved: {file_name}")
      except Exception as e:
          print(f"Error processing {file_name}: {str(e)}")
    else:
      # Copy the image to testing/images
      os.system("cp old_image_folder" + file_name + " new_image_folder")

    d[image['id']] = file_name
  
  for annotation in data["annotations"]:
    file_name = d[annotation["image_id"]]

    with open("label_folder/" + file_name.split(".")[0] + ".txt", "a") as f:
      x, y, width, height = annotation["bbox"]

      # Depending on the format of the bounding box, we can convert it to the format that YOLOv8 uses
      # x_center = x + width/2
      # y_center = y + height/2

      # x_center = (x + x_max)/2
      # y_center = (y + y_max)/2
      # width = x_max - x
      # height = y_max - y

      f.write(str(annotation["category_id"]-1) + " " + " ".join([str(c/512) for c in [x, y, width, height]]) + "\n")