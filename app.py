# chuyen thanh pip neu khong dung conda
# conda activate vietnamese-ocr
# conda install -c conda-forge opencv
# conda install -c conda-forge tensorflow
# conda install -c anaconda scikit-learn
# conda install -c conda-force Matplotlib
# conda install -c conda-forge editdistance  
import numpy as np
import matplotlib.pyplot as plt
import cv2

import process_image
import vietnamese_ocr

#upload image
#thay doi anh o day
valid_img = process_image.readImage('test_img.png')
#prediction
str_pred = vietnamese_ocr.prediction_ocr(valid_img)
print('Du doan ket qua la:')
print(str_pred)



