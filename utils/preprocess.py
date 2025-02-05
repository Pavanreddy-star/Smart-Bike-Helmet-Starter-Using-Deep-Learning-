import cv2
import os
from pathlib import Path

def preprocess_images(input_dir, output_dir, image_size=(224, 224)):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for label in os.listdir(input_dir):
        label_path = os.path.join(input_dir, label)
        output_label_path = os.path.join(output_dir, label)
        Path(output_label_path).mkdir(parents=True, exist_ok=True)

        for image_name in os.listdir(label_path):
            img_path = os.path.join(label_path, image_name)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, image_size)
                cv2.imwrite(os.path.join(output_label_path, image_name), img)

if __name__ == "__main__":
    preprocess_images("data", "processed_data")
