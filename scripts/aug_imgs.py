import albumentations as alb 
import os 
import fiftyone as fo
import fiftyone.zoo as foz 


IMAGES = "../dataset/images"
ANNOTATIONS = "../dataset/annotation"

dataset = foz.load_zoo_dataset('coco-2017', split="train",
    label_types=["detections"],
    classes=["person"],
    max_samples=100,
)