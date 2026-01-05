import cv2
import numpy as np
from . import utils as u

# better to pass grayscale image with applied bilateral filter:
# so rgb -> bilateral -> convert to grayscale -> input image
def sift_heatmap(img):
    sift = cv2.SIFT_create()
    keypoints, _ = sift.detectAndCompute(img, None)

    sift_matrix, coords_list = u.keypoints_2_heatmap(img, keypoints)
    
    return sift_matrix, coords_list


# grayscale bilateral blurred image as input
def hough_circles_heatmap(img):
    h, w = img.shape[:2]

    circles = cv2.HoughCircles(
        img, 
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=15,
        param1=40,
        param2=8,
        minRadius=1,
        maxRadius=10
    )

    circles_matrix = np.zeros(img.shape[:2], dtype=np.uint8)
    coords_list = []

    for c in circles[0, :]:
        x, y = int(round(c[0])), int(round(c[1]))
        if 0 <= y < h and 0 <= x < w:
            circles_matrix[y, x] = 255
            coords_list.append((x, y))

    hough_blur = cv2.GaussianBlur(circles_matrix, (15, 15), 0)
    
    return hough_blur, coords_list


# grayscale image as input
def blob_heatmap(img):
    params = cv2.SimpleBlobDetector_Params()

    params.filterByArea = True
    params.minArea = 10   
    params.maxArea = 300

    params.filterByCircularity = True
    params.minCircularity = 0.6

    params.filterByConvexity = True
    params.minConvexity = 0.87

    params.filterByInertia = True
    params.minInertiaRatio = 0.5

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(img)
    
    blob_matrix, coords_list = u.keypoints_2_heatmap(img, keypoints)
    
    return blob_matrix, coords_list

# edges canny input
def distance_transform_heatmap(img):
    img_inverted = cv2.bitwise_not(img)

    # Calculate Distance to the nearest Zero (Black) pixel
    dist_transform = cv2.distanceTransform(img_inverted, cv2.DIST_L2, 5)

    dist_transform_heatmap = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    return dist_transform_heatmap


def get_combined_candidates(img):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    bilateral = cv2.bilateralFilter(img, 9, 50, 50)  
    bil_gray_image = cv2.cvtColor(bilateral, cv2.COLOR_BGR2GRAY)
    
    _, hough_circles_coords = hough_circles_heatmap(bil_gray_image)
    _, blob_coords = blob_heatmap(gray_image)
    
    comb_coords = list(set(hough_circles_coords) | set(blob_coords))
    
    return comb_coords