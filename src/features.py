import cv2
import numpy as np
from . import candidates as c

def extract_features(img):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # converting to grayscale before bilateral filtering makes the computer colorblind,
    # causing it to blur edges that should be distinct -> therefore filter is applied on rgb
    bilateral = cv2.bilateralFilter(img, 9, 50, 50)  
    bil_gray_image = cv2.cvtColor(bilateral, cv2.COLOR_BGR2GRAY)

    edges_canny = cv2.Canny(bil_gray_image, 100, 200)

    sift_heatmap, _ = c.sift_heatmap(bil_gray_image)

    circles_heatmap, _ = c.hough_circles_heatmap(bil_gray_image)
    
    blob_heatmap, _ = c.blob_heatmap(gray_image)
    
    distance_transform_heatmap = c.distance_transform_heatmap(edges_canny)

    laplacian_64 = cv2.Laplacian(bil_gray_image,cv2.CV_64F)
    laplacian = cv2.normalize(laplacian_64, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    sobelx_64 = cv2.Sobel(src=bil_gray_image, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)
    sobelx = cv2.normalize(sobelx_64, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    sobely_64 = cv2.Sobel(src=bil_gray_image, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)
    sobely = cv2.normalize(sobely_64, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    features = (
        img,
        gray_image,
        bil_gray_image,
        edges_canny,
        sift_heatmap,
        circles_heatmap,
        blob_heatmap,
        distance_transform_heatmap,
        laplacian,
        sobelx,
        sobely
    )
    
    features_stack = np.dstack(features)

    return features_stack
