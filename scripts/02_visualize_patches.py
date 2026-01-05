import os
import numpy as np
import cv2
import random

def create_debug_sheet(label_name, num_samples=64):
    patch_dir = f'data/processed/{label_name}'
    files = [f for f in os.listdir(patch_dir) if f.endswith('.npy')]
    samples = random.sample(files, min(num_samples, len(files)))
    
    grid_size = int(np.sqrt(num_samples))
    rows = []
    
    for i in range(grid_size):
        cols = []
        for j in range(grid_size):
            idx = i * grid_size + j
            patch = np.load(os.path.join(patch_dir, samples[idx]))
            rgb_patch = patch[:, :, :3].astype(np.uint8)
            cols.append(rgb_patch)
        rows.append(np.hstack(cols))
    
    full_grid = np.vstack(rows)
    
    os.makedirs("data/debug", exist_ok=True)
    output_path = os.path.join("data", "debug", f"debug_{label_name}_grid.jpg")
    cv2.imwrite(output_path, full_grid)
    print(f"Saved debug sheet for {label_name}")

for label in ['wall_hole', 'background', 'hold_hole', 'module_hole']:
    create_debug_sheet(label)