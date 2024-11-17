import os 
import shutil


NUMBER_OF_IMAGES = 50
# Kaggle dataset absolute path 
kaggle_dataset_path = "/home/ahmed/Desktop/coralmicro-transfer-learning/dataset/kaggle_face_detection/kaggle_src"

images = os.path.join(kaggle_dataset_path, "images/val")
anns = os.path.join(kaggle_dataset_path, "labels/val")

yolo_folder_path =  "/home/ahmed/Desktop/coralmicro-transfer-learning/dataset/kaggle_face_detection/tmp_pre_yolo"
# Output folder containing images and annotations file
output_folder = "/home/ahmed/Desktop/coralmicro-transfer-learning/dataset/kaggle_face_detection/out_coco/images"

counter = 0
for file in os.listdir(images):

    if counter < NUMBER_OF_IMAGES:
        
        filename = os.fsdecode(file)
        # Test if the annotation file exists in labels2 
        anns_name = f"{filename[:-4]}.txt"
        anns_full_path = os.path.join(anns,anns_name)
        if os.path.exists(anns_full_path):

            # Append the absolute pathname to the .txt file     
            img_full_path = os.path.join(images, filename)

            with open(os.path.join(yolo_folder_path, "images.txt"), 'a') as f: 
                f.write(f"{img_full_path} \n")

            # Copy the image and its anns to the same folder
            dst_folder = os.path.join(yolo_folder_path, "data")
            dst_img_path = os.path.join(dst_folder, filename)
            dst_anns_path = os.path.join(dst_folder, anns_name)

            output_img_path = os.path.join(output_folder, filename)

            shutil.copy(img_full_path, dst_img_path)
            #print(dst_img_path)
            shutil.copy(anns_full_path, dst_anns_path)
            #print(dst_anns_path)
            # Output image folder export 
            shutil.copy(img_full_path, output_img_path)

            counter += 1

        else: 
            print("anns path doesn't exist")
            pass