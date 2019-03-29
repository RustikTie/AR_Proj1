import numpy as np
import cv2

# User input
targetPath = input("Enter the target image: ")
imagePath = input("Enter the image: ")
threshold = input("Enter the threshold: ")

# Load images
target = cv2.imread(targetPath, 0)
image = cv2.imread(imagePath, 0)





