import cv2 
import numpy as np

#Read and process image
def process_image(img_file):
    img = cv2.cvtColor(cv2.imread(img_file), cv2.COLOR_BGR2GRAY)
    height = 118
    width = 2167
    new_img = padding_image(img, width, height)
    img = cv2.resize(new_img,(int(118/height*width),118))
    img = np.pad(img, ((0,0),(0, 2167-width)), 'median')

    img = cv2.GaussianBlur(img, (5,5), 0)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 4)
    img = np.expand_dims(img , axis = 2)
    img = img/255.
    return img

#Padding images
def padding_image(image, width, height):
    h, w = image.shape[:2]
    color = [255, 255, 255]
    if(h < height and w < width):
        top, bottom = int((height - h)/2), int((height - h)/2)
        left, right = int((width - w)/2), int((width - w)/2)       
        new_img = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    else:
        if(h > height):
            left, right = int(((width*h)/height)/2), int(((width*h)/height)/2)
            new_img = cv2.copyMakeBorder(image, 0, 0, left, right, cv2.BORDER_CONSTANT, value=color)
        else:
            top, bottom = int(((height*w)/width)/2), int(((height*w)/width)/2)
            new_img = cv2.copyMakeBorder(image, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=color)
    return new_img

#Erosion, Dilation images
def erosion_dilation_image(image, kernel_size, isErosion):
    kernel = np.ones((kernel_size,kernel_size),np.uint8)
    if(isErosion == True):
        img = cv2.erode(image,kernel,iterations=1)
    else:
        img = cv2.dilate(image,kernel,iterations=1)
    return img

#Load original image
def load_original_img(path):
    return cv2.imread(path)

#convert image to np.array
def convert_img_to_input(img_file):
    valid_img = []
    valid_img.append(img_file)
    valid_img = np.array(valid_img)
    return valid_img