import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Đọc ảnh input.png và chuyển thành numpy array

img = cv2.cvtColor(cv2.imread('vietnamese_hcr/raw/data/0001_samples.png'), cv2.COLOR_BGR2GRAY)
img = cv2.resize(img,(int(118/32*128),118))
    

img = np.pad(img, ((0,0),(0, 2167-128)), 'median')
img = cv2.GaussianBlur(img, (5,5), 0)
img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 4)
img = np.expand_dims(img , axis = 2)
img = img/255


model.load_weights('model_weights.hdf5')
# Đưa ảnh qua mô hình để đưa ra dự đoán về các ký tự trong ảnh
predictions = model.predict(img)

# Chuyển đổi các dự đoán thành chuỗi ký tự
predicted_text = []
for i in range(predictions.shape[1]):
    predicted_char = char_list[np.argmax(predictions[0,i,:])]
    predicted_text.append(predicted_char)

# In ra kết quả
print("Predicted text: ", ''.join(predicted_text))

