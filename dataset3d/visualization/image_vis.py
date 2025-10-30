import rerun as rr
import numpy as np
import os 
import cv2


def visualize_image(image_path : str , entity_path :str = "kitti/image") -> None:
        """Loads and logs a KITTI RGB image to Rerun."""
        if not os.path.exists(image_path):
                print(f"⚠️ Image not found: {image_path}")
                return
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # rerun expects RGB
        rr.log(entity_path , rr.Image(img))

