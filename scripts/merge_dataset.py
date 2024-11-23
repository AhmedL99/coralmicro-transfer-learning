"""The goal of this script is to merge 2 datasets from different to one major one. 
- The 1st dataset (Ahmed class) was collected using the  ../acquire_dataset project.
- The 2nd dataset (not Ahmed class) was from Kaggle: https://www.kaggle.com/datasets/vasukipatel/face-recognition-dataset
"""

import os 
import json
from pycocotools.coco import COCO

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

# Load images and their anns from the 1st class
ahmed_dataset_json_file = "../dataset/Ahmed/result.json"
coco_ahmed = COCO(annotation_file=ahmed_dataset_json_file)
ahmed_imgs_ids = coco_ahmed.getImgIds()
ahmed_imgs = coco_ahmed.loadImgs(ids=ahmed_imgs_ids)


for img in ahmed_imgs: 

    print(img["file_name"][7:])
    images.append(img)
    #print(img["id"])
    annotations.append(coco_ahmed.loadAnns(ids=img["id"]))

    #print(len(coco_ahmed.getAnnIds(imgIds=img["id"])))
    #print(coco_ahmed.loadAnns(ids=img["id"]))

    # Get the number of anns for all the ahmed dataset 
    anns_idx += len(coco_ahmed.getAnnIds(imgIds=img["id"]))

    # Increment the image index
    img_idx += 1

# Load images and their anns from the 1st class
not_ahmed_dataset_json_file = "../dataset/kaggle_face_detection/out_coco/annotations.json"
coco_not_ahmed = COCO(annotation_file=not_ahmed_dataset_json_file)
not_ahmed_imgs_ids = coco_not_ahmed.getImgIds()
not_ahmed_imgs = coco_not_ahmed.loadImgs(ids=not_ahmed_imgs_ids)

for img in not_ahmed_imgs: 
    
    # Get the anns list
    img_anns_ids = coco_not_ahmed.getAnnIds(imgIds=img["id"])
    for ann_id in  img_anns_ids: 
        ann = coco_not_ahmed.loadAnns(ids=ann_id)
        ann[0]["image_id"]= img_idx
        ann[0]["id"] = anns_idx
        annotations.append(ann)
        anns_idx += 1
        #print(ann)
        #print(img_idx)

    img["id"] = img_idx
    img_idx += 1
    images.append(img)

print(img_idx)



output_json_file = "../dataset/final/annotations.json"

output_data = {
    "categories": categories,
    "images": images,
    "annotations": annotations
}

# Write data to the output json file 
with open(output_json_file, 'w') as file:
    json.dump(output_data, file)

