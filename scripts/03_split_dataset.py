import os
import shutil
import random
from glob import glob

SOURCE_DIR = 'data/processed'
TARGET_DIR = 'data/dataset'
SPLITS = {'train': 0.70, 'val': 0.15, 'test': 0.15}

# wall holes is dominant class; count other datapoints to balance dataset
counts = {
    'wall_hole': len(glob(os.path.join(SOURCE_DIR, 'wall_hole/*.npy'))),
    'hold_hole': len(glob(os.path.join(SOURCE_DIR, 'hold_hole/*.npy'))),
    'background': len(glob(os.path.join(SOURCE_DIR, 'background/*.npy'))),
    'module_hole': len(glob(os.path.join(SOURCE_DIR, 'module_hole/*.npy')))
}

# dynamically calculate limit for wall holes
# treat modules holes as background
negatives_count = counts['background'] + counts['module_hole']
hold_count = counts['hold_hole']

# set the limit to the average of other two categories to keep it balanced
WALL_HOLE_LIMIT = int((negatives_count + hold_count) / 2) * 2
print(f"Wall hole limit set to: {WALL_HOLE_LIMIT}")

label_map = {
    'wall_hole': 'wall_hole',
    'hold_hole': 'hold_hole',
    'module_hole': 'background', # modules holes are background
    'background': 'background'
}

for split in SPLITS.keys():
    for final_cat in ['wall_hole', 'hold_hole', 'background']:
        os.makedirs(os.path.join(TARGET_DIR, split, final_cat), exist_ok=True)

for original_cat, final_cat in label_map.items():
    files = glob(os.path.join(SOURCE_DIR, original_cat, '*.npy'))
    random.shuffle(files)
    
    if original_cat == 'wall_hole':
        files = files[:WALL_HOLE_LIMIT]
    
    total = len(files)
    train_end = int(total * SPLITS['train'])
    val_end = train_end + int(total * SPLITS['val'])
    
    for i, f_path in enumerate(files):
        fname = os.path.basename(f_path)
        target = 'train' if i < train_end else ('val' if i < val_end else 'test')
        shutil.copy(f_path, os.path.join(TARGET_DIR, target, final_cat, fname))

print("Dataset is balanced and split")