import numpy as np
import matplotlib.pyplot as plt
import cv2

import process_image
import vietnamese_ocr

#upload image
valid_img = process_image.readImage('d:\\Work Or Study\\Python\\Final PID\\vietnamese_hcr\\raw\\data\\1822_samples.png')
#prediction
str_pred = vietnamese_ocr.prediction_ocr(valid_img)
print('Du doan ket qua la')
print(str_pred)



