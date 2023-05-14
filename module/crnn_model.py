# import our model, different layers and activation function
import os
from tensorflow import keras
from keras.layers import Dense, LSTM, Reshape, BatchNormalization, Input, Conv2D, MaxPool2D, Lambda, Bidirectional, Add, Activation
from keras.models import Model
import keras.backend as K


# Mô hình CRNN và LSTM nhận dạng ký tự

# input
inputs = Input(shape=(118, 2167, 1))

# Block 1
x = Conv2D(64, (3, 3), padding='same')(inputs)
x = MaxPool2D(pool_size=3, strides=3)(x)
x = Activation('relu')(x)
x_1 = x

# Block 2
x = Conv2D(128, (3, 3), padding='same')(x)
x = MaxPool2D(pool_size=3, strides=3)(x)
x = Activation('relu')(x)
x_2 = x

# Block 3
x = Conv2D(256, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x_3 = x

# Block4
x = Conv2D(256, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Add()([x, x_3])
x = Activation('relu')(x)
x_4 = x

# Block5
x = Conv2D(512, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x_5 = x

# Block6
x = Conv2D(512, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Add()([x, x_5])
x = Activation('relu')(x)

# Block7
x = Conv2D(1024, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = MaxPool2D(pool_size=(3, 1))(x)
x = Activation('relu')(x)

# pooling layer with kernel size (2,2) to make the height/2 #(1,9,512)
x = MaxPool2D(pool_size=(3, 1))(x)

# Lambda được sử dụng để loại bỏ chiều thứ nhất của tensor đầu ra (chiều batch_size)
# và đưa ra đầu ra có hình dạng (height, width, channels).
squeezed = Lambda(lambda x: K.squeeze(x, 1))(x)

# Mỗi khối LSTM có 512 đơn vị đầu ra và sử dụng dropout để tránh tình trạng overfitting.
blstm_1 = Bidirectional(
    LSTM(512, return_sequences=True, dropout=0.2))(squeezed)
blstm_2 = Bidirectional(LSTM(512, return_sequences=True, dropout=0.2))(blstm_1)

# this is our softmax character proprobility with timesteps
outputs = Dense(140+1, activation='softmax')(blstm_2)

# model to be used at test time
model = Model(inputs, outputs)

# read model
model.load_weights('./data/model_weights.hdf5')
