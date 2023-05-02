import cv2 
import numpy as np

#Read and process image
def process_image(img_file):
    img = cv2.cvtColor(cv2.imread(img_file), cv2.COLOR_BGR2GRAY)
    height = 118
    width = 2122
    img = cv2.resize(img,(int(118/height*width),118))
    img = np.pad(img, ((0,0),(0, 2167-width)), 'median')
    img = cv2.GaussianBlur(img, (5,5), 0)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 4)
    return img
#Load original image
def load_original_img(path):
    return cv2.imread(path)
#convert image to np.array
def convert_img_to_input(img_file):
    img_file = img_file/255
    img_file = np.expand_dims(img_file , axis = 2)
    valid_img = []
    valid_img.append(img_file)
    valid_img = np.array(valid_img)
    return valid_img
