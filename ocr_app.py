import numpy as np
import cv2

from PIL import Image

import module.process_image as process_image
import module.vietnamese_ocr as vietnamese_ocr
import module.vietocr_module as vietocr_module
import module.crop_text_line as segments

# prediction ocr


def prediction_ocr_crnn_ctc(img_model_input):
    str_pred = vietnamese_ocr.prediction_ocr(img_model_input)
    print('Prediction:')
    print(str_pred)
    return str_pred


def prediction_multiline(img_model_input, size):
    str_pred = vietnamese_ocr.prediction_ocr_multi(img_model_input, size)
    print('Prediction: ', str_pred)
    return str_pred


def prediction_ocr_vietocr(img_model_input):
    str_pred = vietocr_module.vietOCR_prediction(img_model_input)
    print('Prediction:')
    print(str_pred)
    return str_pred


def prediction_ocr_vietocr_mul(img_model_input):
    all_predictions = []
    for img in img_model_input:
        np_image = np.asarray(img)
        image_rgb = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        str_pred = vietocr_module.vietOCR_prediction(image_pil)
        all_predictions.append(str_pred)
    pred = '\n'.join(all_predictions)
    print("Prediction :{}".format(pred))
    return pred


def test_prediction_mul(image_path):
    all_predictions = []
    ori_img = cv2.imread(image_path)

    # prediction with vietnamese_ocr train
    valid_img, arr = segments.segmentation_text_line(ori_img)
    for img in arr:
        np_image = np.asarray(img)
        image_rgb = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        str_pred = vietocr_module.vietOCR_prediction(image_pil)
        all_predictions.append(str_pred)
    print('\n'.join(all_predictions))


# test_prediction_mul('test_n1.jpg')
