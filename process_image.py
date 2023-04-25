import cv2 
import numpy as np


def readImage(img_file):
    valid_img = []
    img = cv2.cvtColor(cv2.imread(img_file), cv2.COLOR_BGR2GRAY)
    height = 118
    width = 2122

    img = cv2.resize(img,(int(118/height*width),118))
    img = np.pad(img, ((0,0),(0, 2167-width)), 'median')
    img = cv2.GaussianBlur(img, (5,5), 0)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 4)
    img = np.expand_dims(img , axis = 2)
    img = img/255.
    print(img.shape)

    valid_img.append(img)
    valid_img = np.array(valid_img)
    
    return valid_img
