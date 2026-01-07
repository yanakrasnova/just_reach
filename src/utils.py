import cv2
import numpy as np
from pathlib import Path


def keypoints_2_heatmap(img, kp_arr):
    h, w = img.shape[:2]
    kp_matrix = np.zeros(img.shape[:2], dtype=np.uint8)

    coords_list = []
    
    for kp in kp_arr:
        x, y = int(round(kp.pt[0])), int(round(kp.pt[1]))
        if 0 <= y < h and 0 <= x < w:
            kp_matrix[y, x] = 255
            coords_list.append((x, y))

    kp_heatmap = cv2.GaussianBlur(kp_matrix, (15, 15), 0)
    
    return kp_heatmap, coords_list

def extract_patch(img, x_center, y_center, size):
    r = size // 2
    img_padded = cv2.copyMakeBorder(img, size, size, size, size, cv2.BORDER_REFLECT_101)
    x_padded = x_center + size
    y_padded = y_center + size
    
    patch = img_padded[y_padded - r:y_padded + r, x_padded - r:x_padded + r]
    return patch

def load_dataset(root_path, classes=None, expected_shape=None):
    """
    Loads .npy patches from subdirectories.
    
    Args:
        root_path (str): Path to the folder containing class folders.
        classes (list, optional): List of class names to load. 
                                  If None, defaults to ['background', 'hold_hole', 'wall_hole'].
        expected_shape (tuple, optional): If provided, skips files that do not match this shape.
        
    Returns:
        X (np.array): Shape (N_samples, ...)
        y (np.array): Shape (N_samples, )
        class_names (list): The list of classes used for mapping.
    """
    root = Path(root_path)
    
    if classes is None:
        classes = sorted([d.name for d in root.iterdir() if d.is_dir()])
        print(f"Auto-detected classes: {classes}")
    
    X_list = []
    y_list = []
    
    print(f"Loading data from {root}...")
    
    for class_id, class_name in enumerate(classes):
        class_dir = root / class_name
        
        if not class_dir.exists():
            print(f"Warning: Folder '{class_name}' not found in {root}")
            continue
            
        files = list(class_dir.glob('*.npy'))
        print(f"  Found {len(files)} samples for class '{class_name}' (ID: {class_id})")
        
        for file_path in files:
            try:
                patch = np.load(file_path)

                if expected_shape is not None and patch.shape != expected_shape:
                    print(f"Skipping {file_path.name}: shape {patch.shape} != {expected_shape}")
                    continue
                    
                X_list.append(patch)
                y_list.append(class_id)
                
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

    X = np.array(X_list)
    y = np.array(y_list)
    
    print("-" * 30)
    print(f"Data Loaded.")
    print(f"X Shape: {X.shape}") 
    print(f"y Shape: {y.shape}")
    
    return X, y, classes

def preprocess_patches_dataset(X):
    """
    Downsamples and flattens image patches to match the Random Forest input format.
    
    Args:
        X (np.array): Shape (Batch, Height, Width, Channels) 
                      e.g. (N, 64, 64, 13)
                      
    Returns:
        np.array: Shape (Batch, Flattened_Features)
                  e.g. (N, 3328)
    """
    # downsample: keep every 4th pixel in height and width
    # preserves all samples (axis 0) and all channels (axis 3)
    X_small = X[:, ::4, ::4, :]
    
    # flatten: (batch, h*w*c)
    X_flat = X_small.reshape(X_small.shape[0], -1)
    
    return X_flat