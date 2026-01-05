import cv2
import numpy as np


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

# todo: setup two nn - one for 32px patch and another for 96. if both agree - positive
def extract_patch(img, x_center, y_center, size):
    r = size // 2
    img_padded = cv2.copyMakeBorder(img, size, size, size, size, cv2.BORDER_REFLECT_101)
    x_padded = x_center + size
    y_padded = y_center + size
    
    patch = img_padded[y_padded - r:y_padded + r, x_padded - r:x_padded + r]
    return patch