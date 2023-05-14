import cv2
import numpy as np


def segmentation_text_line(image):

    img_clone = image.copy()
    segments = []

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    # binary
    thresh = cv2.adaptiveThreshold(gray, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 4)

    kernel = np.ones((10, 1), np.uint8)
    ero = cv2.erode(thresh, kernel, iterations=1)

    # dilation
    kernel = np.ones((5, 150), np.uint8)
    img_dilation = cv2.dilate(ero, kernel, iterations=1)
    # find contours
    ctrs, hier = cv2.findContours(
        img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # sort contours
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[1])
    min_width = 50  # giá trị ngưỡng chiều rộng tối thiểu
    min_height = 10  # giá trị ngưỡng chiều cao tối thiểu
    for i, ctr in enumerate(sorted_ctrs):
        # Get bounding box
        x, y, w, h = cv2.boundingRect(ctr)
        print('segment no:', str(i), '[', w, '-', h, ']')
        if (w > min_width and h > min_height):
            roi = image[y:y+h, x:x+w]
            segments.append(roi)
            # show ROI
            # cv2.imshow('segment no:'+str(i), roi)
            cv2.rectangle(img_clone , (x, y), (x + w, y + h), (90, 0, 255), 2)
            # cv2.waitKey(0)

    return img_clone, segments
