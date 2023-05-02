
import streamlit as st
import numpy as np
import cv2


import module.process_image as dip
import app as ocr

# Dung de lay anh tu st.file_uploader
IMAGE_UPLOAD = None
# Dung de dua vao ham xu ly anh cua opencv
OPENCV_IMAGE = None
# Dung de chuyen ve dang dau vao cua model
MODEL_INPUT = None
# Chuoi duoc du doan
PREDICTION_STR = ''

st.set_page_config('Vietnamese ORC', 'orc.ico')

st.title("Nhận dạng chữ viết tay Tiếng Việt")
st.markdown('<div style="margin-top:2rem"></div>',
            unsafe_allow_html=True)  # Insert a blank line
st.text("Hướng dẫn sử dụng: Upload ảnh cần nhận diện sau đó chọn xử lý tự động")
st.text("hoặc tự chỉnh bên trái màn hình sau đó chọn nhận diện hệ thống sẽ trả ra kết quả")
st.markdown('<div style="margin-top:2rem"></div>', unsafe_allow_html=True)


IMAGE_UPLOAD = st.file_uploader(
    "Định dạng cho phép: JPG, PNG, JPEG - Độ phân giải 1200x500", type=['png', 'jpg', 'jpeg'])
# Display image
if IMAGE_UPLOAD is not None:
    st.image(IMAGE_UPLOAD, caption='Ảnh đã tải lên', use_column_width=True)

st.markdown('<div style="margin-top:3rem"></div>', unsafe_allow_html=True)
st.header("Ảnh đã xử lý")
processed_image_container = st.empty()
processed_image_container.markdown('<div style="margin-top:5rem"></div>', unsafe_allow_html=True)
st.markdown('<div style="margin-top:2rem"></div>', unsafe_allow_html=True)
st.header("Kết quả")
st.markdown('<div style="margin-top:1rem"></div>', unsafe_allow_html=True)
result_container = st.empty()
result_container.code("Result text will appear here")

# Sidebar
with st.sidebar:
    st.title("Chức năng")
    if st.sidebar.button("Tự động xử lý ảnh", type="primary"):
        # Xử lý khi nút [Tự động xử lý ảnh] được nhấn
        if IMAGE_UPLOAD is not None:
            # Lấy file từ uploader chuyển về file mà openCV đọc được sau đó truyền vào hàm dip.process_image
            OPENCV_IMAGE = IMAGE_UPLOAD.name
            # process image and display
            MODEL_INPUT = dip.process_image(OPENCV_IMAGE)
            processed_image_container.image(MODEL_INPUT)
        else:
            result_container.warning('Please upload an image first!')

    # Giá trị của slider 1
    param1 = st.sidebar.slider(
        "Chức năng 1", 0, 8, 2, help="Nội dung dấu ? ở slider 1", key=1)

    # Giá trị của slider 2
    param2 = st.sidebar.slider(
        "Chức năng 2", 0, 8, 2, help="Nội dung dấu ? ở slider 2", key=2)

    if st.sidebar.button("Nhận diện văn bản", type="primary"):
        # Xử lý khi nút [Nhận diện văn bản] được nhấn
        if IMAGE_UPLOAD is not None:
            # Chuyển ảnh về dạng mà model có thể đọc được bằng dip.convert_img_to_input
            OPENCV_IMAGE = IMAGE_UPLOAD.name
            # process image and display
            MODEL_INPUT = dip.process_image(OPENCV_IMAGE)
            processed_image_container.image(MODEL_INPUT)
            # Dự đoán chuỗi
            MODEL_INPUT =  dip.convert_img_to_input(MODEL_INPUT)
            PREDICTION_STR = ocr.prediction_ocr(MODEL_INPUT)
            # Đưa chuổi dự đoán vào khung kết quả
            result_container.code(PREDICTION_STR)
        else:
            result_container.warning('Please upload an image first!')

    st.markdown('<div style="margin-top:3rem"></div>', unsafe_allow_html=True)

    st.text("Tên thành viên thực hiện")
    st.text("20110457 - Trần Tiến Đạt")
    st.text("20110132 - Nguyễn Minh Cường")
    st.text("19110202 - Đỗ Thanh Hiếu")
    st.text("19110037 - Huỳnh Minh Thông")
    st.text("GVHD: PGS.TS Hoàng Văn Dũng")
