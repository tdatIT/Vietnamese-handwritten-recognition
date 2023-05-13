import cv2
import numpy as np
# import image
image = cv2.imread('test_case_multi_2.jpg')
# cv2.imshow('orig',image)
# cv2.waitKey(0)

# binary
# grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (7, 7), 0)
cv2.imshow('gray', gray)
cv2.waitKey(0)

# binary
thresh = cv2.adaptiveThreshold(gray, 255,
                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 4)

cv2.imshow('second', thresh)
cv2.waitKey(0)

# dilation
kernel = np.ones((25, 100), np.uint8)
img_dilation = cv2.dilate(thresh, kernel, iterations=1)
cv2.imshow('dilated', img_dilation)
cv2.waitKey(0)

# find contours
ctrs, hier = cv2.findContours(
    img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# sort contours
sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

for i, ctr in enumerate(sorted_ctrs):
    # Get bounding box
    x, y, w, h = cv2.boundingRect(ctr)

    # Getting ROI
    roi = image[y:y+h, x:x+w]

    # show ROI
    cv2.imshow('segment no:'+str(i), roi)
    cv2.rectangle(image, (x, y), (x + w, y + h), (90, 0, 255), 2)
    cv2.waitKey(0)

cv2.imshow('marked areas', image)
cv2.waitKey(0)
