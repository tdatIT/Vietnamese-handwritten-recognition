import cv2
import numpy as np
from matplotlib import pyplot as plt

# Read and process image


def process_multi(segments):
    copy_valid = []
    size = 0
    valid_imgs = []
    for img in segments:
        pre_img = process_image_mul(img)
        valid_imgs.append(pre_img)
        size += 1
        # plt.subplot(len(segments), 1, size)
        # plt.imshow(pre_img, cmap="gray")
    # plt.show()
    copy_valid = valid_imgs.copy()
    valid_imgs = np.array(valid_imgs)
    return valid_imgs, copy_valid, size


def process_image_mul(cv2_img):
    img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
    height = 118
    width = 2167
    img = cv2.bilateralFilter(img, 9, 80, 80)
    img = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 4)
    img_pad = padding_image(img, width, height)
    img = cv2.resize(img_pad, (int(118/height*width), 118))
    img = np.pad(img, ((0, 0), (0, 2167-width)), 'median')
    # img = cv2.GaussianBlur(img, (5, 5), 0)

    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)

    img = np.expand_dims(img, axis=2)
    img = img/255.
    return img


def process_image(img_file):
    img = cv2.cvtColor(img_file, cv2.COLOR_BGR2GRAY)
    height = 118
    width = 2122
    img = cv2.resize(img, (int(118/height*width), 118))
    img = np.pad(img, ((0, 0), (0, 2167-width)), 'median')
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 4)
    img = np.expand_dims(img, axis=2)
    img = img/255.
    return img


def load_original_img(path):
    return cv2.imread(path)

# convert image to np.array


def convert_img_to_input(img_file):

    valid_img = []
    valid_img.append(img_file)
    valid_img = np.array(valid_img)
    return valid_img

# Padding images


def padding_image(image, width, height):

    h, w = image.shape[:2]
    color = [0, 0, 0]
    if (h < height and w < width):
        top, bottom = int((height - h)/2), int((height - h)/2)
        left, right = 0, int((width - w)/2)
        new_img = cv2.copyMakeBorder(
            image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    else:
        if (h > height):
            left, right = 0, int((width*h)/height)
            new_img = cv2.copyMakeBorder(
                image, 0, 0, left, right, cv2.BORDER_CONSTANT, value=color)
        if (w > width):
            top, bottom = int(((height*w)/width)/2), int(((height*w)/width)/2)
            new_img = cv2.copyMakeBorder(
                image, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=color)
    return new_img


# Cropping images
def crop_image(image, width, height):

    h, w = image.shape[:2]
    if (h > height and w > width):
        startx = w // 2 - (width // 2)
        starty = h // 2 - (height // 2)
        return image[starty:starty + height, startx:startx + width]
    else:
        return image

# Erosion, Dilation images


def erosion_dilation_image(image, kernel_size, isErosion):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    if (isErosion == True):
        img = cv2.erode(image, kernel, iterations=1)
    else:
        img = cv2.dilate(image, kernel, iterations=1)
    return img
