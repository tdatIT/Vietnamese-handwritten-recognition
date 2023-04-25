from tensorflow.keras.layers import Dense, LSTM, Reshape, BatchNormalization, Input, Conv2D, MaxPool2D, Lambda, Bidirectional, Add, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.activations import relu, sigmoid, softmax
import tensorflow.keras.backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import CSVLogger, TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

import numpy as np

import crnn_model

#Constant
NO_PREDICTS = 1
OFFSET=0
char_list =[' ', '#', "'", '(', ')', '+', ',', '-', '.', 
            '/', '0', '1', '2', '3', '4', '5', '6', '7',
            '8', '9', ':', 'A', 'B', 'C', 'D', 'E', 'F',
            'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
            'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
            'Y', 'a', 'b', 'c', 'd', 'e', 'g', 'h', 'i',
            'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
            't', 'u', 'v', 'w', 'x', 'y', 'z', 'Â', 'Ê',
            'Ô', 'à', 'á', 'â', 'ã', 'è', 'é', 'ê', 'ì', 
            'í', 'ò', 'ó', 'ô', 'õ', 'ù', 'ú', 'ý', 'ă',
            'Đ', 'đ', 'ĩ', 'ũ', 'Ơ', 'ơ', 'ư', 'ạ', 'ả',
            'ấ', 'ầ', 'ẩ', 'ậ', 'ắ', 'ằ', 'ẵ', 'ặ', 'ẻ',
            'ẽ', 'ế', 'ề', 'ể', 'ễ', 'ệ', 'ỉ', 'ị', 'ọ',
            'ỏ', 'ố', 'ồ', 'ổ', 'ỗ', 'ộ', 'ớ', 'ờ', 'ở',
            'ỡ', 'ợ', 'ụ', 'ủ', 'Ứ', 'ứ', 'ừ', 'ử', 'ữ',
            'ự', 'ỳ', 'ỵ', 'ỷ', 'ỹ']
#Predict and encoder
def prediction_ocr(valid_img):
    prediction = crnn_model.model.predict(valid_img[OFFSET:OFFSET+NO_PREDICTS])
    prediction.shape
    #decoder
    out = K.get_value(K.ctc_decode(prediction, input_length=np.ones(prediction.shape[0])*prediction.shape[1],
                         greedy=True)[0][0])
    # see the results
    pred = ""
    for x in out:
        for p in x:  
            if int(p) != -1:
                pred += char_list[int(p)]
    return pred