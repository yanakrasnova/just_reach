import xml.etree.ElementTree as ET
import os
import cv2
import numpy as np
import src.features as f
import src.utils as u

XML_FILES = ['data/training_ds/annotations1.xml', 'data/training_ds/annotations2.xml']
IMAGE_DIR = 'data/training_ds'
OUTPUT_BASE = 'data/processed'

def process_annotations(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    for image_tag in root.findall('image'):
        img_name = image_tag.get('name')
        img_path = os.path.join(IMAGE_DIR, img_name)
        
        if not os.path.exists(img_path):
            print(f"Skipping {img_name}: Image not found.")
            continue

        print(f"Processing {img_name} from {xml_path}...")
        image = cv2.imread(img_path)

        features_stack = f.extract_features(image)
        
        for point_obj in image_tag.findall('points'):
            label = point_obj.get('label')
            coords = point_obj.get('points').split(',')
            x, y = int(float(coords[0])), int(float(coords[1]))
            
            patch = u.extract_patch(features_stack, x, y, 64)
            
            save_dir = os.path.join(OUTPUT_BASE, label)
            os.makedirs(save_dir, exist_ok=True)
            
            save_name = f"{img_name.split('.')[0]}_{x}_{y}.npy"
            np.save(os.path.join(save_dir, save_name), patch)
            
        del image
        del features_stack

for xml in XML_FILES:
    process_annotations(xml)

print("All patches extracted successfully.")