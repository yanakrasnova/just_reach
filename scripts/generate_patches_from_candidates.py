import glob
import cv2
import numpy as np
import src.candidates as c
import src.features as f
import src.utils as u
import os

INPUT_DIR = 'data/training_ds/*.jpg'
OUTPUT_DIR = 'data/interim'

image_paths = glob.glob(INPUT_DIR)

for path in image_paths:
    img_name = os.path.basename(path).split('.')[0]
    if not os.path.exists(path):
        print(f"Skipping {img_name}: Image not found.")
        continue
    
    image = cv2.imread(path)

    candidate_coords = c.get_combined_candidates(image)
    
    features_stack = f.extract_features(image)
    
    for coords in candidate_coords:
        x_coord = coords[0]
        y_coord = coords[1]

        patch = u.extract_patch(features_stack, x_coord, y_coord, 64)
        
        save_dir = os.path.join(OUTPUT_DIR, f"{img_name}_{x_coord}_{y_coord}")
        np.save(save_dir, patch)
        
    del image
    del features_stack
        
    
    
        