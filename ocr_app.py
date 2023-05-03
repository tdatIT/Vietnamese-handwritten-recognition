# chuyen thanh pip neu khong dung conda
# conda activate vietnamese-ocr
# conda install -c conda-forge opencv
# conda install -c conda-forge tensorflow
# conda install -c anaconda scikit-learn
# conda install -c conda-forge matplotlib
# conda install -c conda-forge editdistance
import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image

import module.process_image as process_image
import module.vietnamese_ocr as vietnamese_ocr
import module.vietocr_module as vietocr_module

# prediction ocr


def prediction_ocr_crnn_ctc(img_model_input):
    str_pred = vietnamese_ocr.prediction_ocr(img_model_input)
    print('Prediction:')
    print(str_pred)
    return str_pred


def prediction_ocr_vietocr(img_model_input):
    str_pred = vietocr_module.vietOCR_prediction(img_model_input)
    print('Prediction:')
    print(str_pred)
    return str_pred
# upload image
# thay doi anh o day


def test_prediction(image_path):

    ori_img = process_image.load_original_img(image_path)
    image = process_image.process_image(ori_img)

    # prediction with vietnamese_ocr train
    valid_img = process_image.convert_img_to_input(image)
    str_pred = vietnamese_ocr.prediction_ocr(valid_img)

    # prediction with vietocr

    # str_pred = vietocr_module.vietOCR_prediction(path)

    subf = plt.subplot(2, 1, 1)

    plt.title('Ảnh gốc:')
    plt.imshow(ori_img)

    subf = plt.subplot(2, 1, 2)
    plt.title('PID')
    plt.imshow(image, cmap='gray_r')
    plt.xlabel("Kết quả OCR : "+str_pred, color="red")
    plt.tight_layout()
    plt.show()
