"""The goal of this script is to merge 2 datasets from different to one major one. 
- The 1st dataset (Ahmed class) was collected using the  ../acquire_dataset project.
- The 2nd dataset (not Ahmed class) was from Kaggle: https://www.kaggle.com/datasets/vasukipatel/face-recognition-dataset
"""

import os 
import json
from pycocotools.coco import COCO
import albumentations as A 
import cv2
import numpy as np 



IMG_WIDTH = 324
IMG_HEIGHT = IMG_WIDTH


# Merge Data
images = []
annotations = []

categories = [
    {
      "id": 0,
      "name": "Ahmed"
    }, 
    {
      "id": 1,
      "name": "not_Ahmed"
    }
]

img_idx = 0 
anns_idx = 0

# Path to the images folder
images_path = "/home/ahmed/Desktop/coralmicro-transfer-learning/dataset/merged/images"


# Load images and their anns from the 1st class
ahmed_dataset_json_file = "../dataset/Ahmed/result.json"
coco_ahmed = COCO(annotation_file=ahmed_dataset_json_file)
ahmed_imgs_ids = coco_ahmed.getImgIds()
ahmed_imgs = coco_ahmed.loadImgs(ids=ahmed_imgs_ids)


for img in ahmed_imgs: 

    # Change image file name
    #in_img_path = os.path.join(images_path, img["file_name"][16:])
    #out_img_path = os.path.join(images_path, str(img_idx))
    #os.rename(in_img_path, out_img_path)
    
    img["file_name"] = f"{str(img_idx)}.jpg"
    images.append(img)
    annotations.append(coco_ahmed.loadAnns(ids=img["id"])[0])
    #print(len(coco_ahmed.getAnnIds(imgIds=img["id"])))
    #print(coco_ahmed.loadAnns(ids=img["id"]))

    # Get the number of anns for all the ahmed dataset 
    anns_idx += len(coco_ahmed.getAnnIds(imgIds=img["id"]))

    # Increment the image index
    img_idx += 1

# Load images and their anns from the 2nd class
not_ahmed_dataset_json_file = "../dataset/kaggle_face_detection/out_coco/annotations.json"
coco_not_ahmed = COCO(annotation_file=not_ahmed_dataset_json_file)
not_ahmed_imgs_ids = coco_not_ahmed.getImgIds()
not_ahmed_imgs = coco_not_ahmed.loadImgs(ids=not_ahmed_imgs_ids)

for img in not_ahmed_imgs: 
    
    # Change image filename
    #in_img_path = os.path.join(images_path, img["file_name"])
    #out_img_path = os.path.join(images_path, str(img_idx))
    #os.rename(in_img_path, out_img_path)

    img["file_name"] = f"{str(img_idx)}.jpg"
    # Get the anns list
    img_anns_ids = coco_not_ahmed.getAnnIds(imgIds=img["id"])
    for ann_id in  img_anns_ids: 
        ann = coco_not_ahmed.loadAnns(ids=ann_id)
        ann[0]["image_id"]= int(img_idx)
        ann[0]["id"] = anns_idx
        annotations.append(ann[0])
        anns_idx += 1

    img["id"] = img_idx
    img_idx += 1
    images.append(img)

print(img_idx)


output_json_file = "../dataset/merged/annotations.json"

output_data = {
    "images": images,
    "annotations": annotations,
    "categories": categories
}

# Write data to the output json file 
with open(output_json_file, 'w') as file:
    json.dump(output_data, file, indent=3)


# Augment data 
## Configure augmentations 
pre_aug_json = output_json_file
pre_aug = COCO(annotation_file=pre_aug_json)

class_labels = ['Ahmed', 'not_Ahmed']
aug_anns = []

## Resize the images: 324 x 324
Resize = A.Compose([
    A.Resize(width=IMG_WIDTH, height=IMG_HEIGHT),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
], bbox_params=A.BboxParams(format='coco',label_fields=['bbox_classes']))

aug_path = "/home/ahmed/Desktop/coralmicro-transfer-learning/dataset/aug"
aug_images = os.path.join(aug_path, "images")
aug_json = os.path.join(aug_path, "aug.json")

# Loop through all the images 
for img in images: 
    
    arr_img = cv2.imread(os.path.join(images_path, img["file_name"]))
    #print(os.path.join(images_path, img["file_name"]))
    # Get anns for this image 
    img_bbox = []
    bbox_classes = []
    img_anns_ids = pre_aug.getAnnIds(imgIds=img["id"])
    nbr_anns = len(img_anns_ids)

    # Collect bboxes for each ima
    for ann_id in img_anns_ids: 
        ann = pre_aug.loadAnns(ids=ann_id)
        #conc_bbox = ann[0]["bbox"] + [class_labels[ann[0]["category_id"]]]
        img_bbox.append(ann[0]["bbox"])
        bbox_classes.append(class_labels[ann[0]["category_id"]])

    # Perform image resize 
    transformed = Resize(image=arr_img, bboxes=img_bbox, bbox_classes=bbox_classes)
    transformed_image = transformed['image']
    transformed_bboxes = transformed['bboxes']
    transformed_class_labels = transformed['bbox_classes']


    # Export new image
    aug_filename = os.path.join(aug_images, img["file_name"])
    cv2.imwrite(aug_filename, transformed_image)
    # Update Json file 
    img["file_name"] = aug_filename 
    img["width"] = IMG_WIDTH
    img["height"] = IMG_HEIGHT
           
    for ann_id, transformed_bbox, transformed_class_label in zip(img_anns_ids, transformed_bboxes, transformed_class_labels):
        # Load the annotation using its ID
        ann = pre_aug.loadAnns(ids=ann_id)
    
        # Update the bounding box
        ann[0]["bbox"] = transformed_bbox

        # Update the category_id based on the transformed class labels
        ann[0]["category_id"] = 0 if transformed_class_label == "Ahmed" else 1

              
        print(ann)

    



# Output augmented 
output_aug = {
    "images": images,
    "annotations": annotations,
    "categories": categories
}

# Write data to the output json file 
with open(aug_json, 'w') as file:
  json.dump(output_data, file, indent=3)
