import streamlit as st
import numpy as np
import cv2
from PIL import Image
import pyperclip


import module.process_image as dip
import ocr_app as ocr


def init_session_var():
    if 'MODEL_OPTION' not in st.session_state:
        st.session_state.MODEL_OPTION = None

    if 'MODEL_INPUT' not in st.session_state:
        st.session_state.MODEL_INPUT = None

    if 'OPENCV_IMAGE' not in st.session_state:
        st.session_state.OPENCV_IMAGE = None


def main():
    # init session state
    init_session_var()

    st.set_page_config(
        'Vietnamese Handwritten Recognition (OCR)', './img_streamlit/logo.ico')
    message_container = st.empty()
    st.title("Nhận dạng chữ viết tay Tiếng Việt")
    st.markdown('<div style="margin-top:2rem"></div>',
                unsafe_allow_html=True)  # Insert a blank line
    st.text("Hướng dẫn sử dụng: Upload ảnh cần nhận diện sau đó chọn xử lý tự động")
    st.text("hoặc tự chỉnh bên trái màn hình sau đó"
            + "chọn nhận diện hệ thống sẽ trả ra kết quả")
    st.markdown('<div style="margin-top:2rem"></div>', unsafe_allow_html=True)

    IMAGE_UPLOAD = st.file_uploader(
        "Định dạng cho phép: JPG, PNG, JPEG - Độ phân giải 2200x200", type=['png', 'jpg', 'jpeg'])
    # Display image
    if IMAGE_UPLOAD is not None:
        st.image(IMAGE_UPLOAD, caption='Ảnh đã tải lên', use_column_width=True)

    processed_image_container = st.empty()
    processed_image_container.markdown(
        '<div style="margin-top:2rem"></div>', unsafe_allow_html=True)
    st.markdown('<div style="margin-top:2rem"></div>', unsafe_allow_html=True)
    st.header("Kết quả")
    st.markdown('<div style="margin-top:1rem"></div>', unsafe_allow_html=True)
    st.markdown(
        '<style>.css-pxxe24 {visibility: hidden;}</style>', unsafe_allow_html=True)

    result_container = st.container()

    # Sidebar
    with st.sidebar:

        st.title("Chức năng")
        option = st.selectbox('Chọn model để nhận diện',
                              ('CRNN + LTSM + CTC', 'AttentionOCR (VietOCR)'))

        if st.sidebar.button("Tự động xử lý ảnh", type="primary"):
            # Xử lý khi nút [Tự động xử lý ảnh] được nhấn
            if IMAGE_UPLOAD is not None:
                # Lấy file từ uploader chuyển về file mà openCV đọc được sau đó truyền vào hàm dip.process_image
                img_array = np.frombuffer(IMAGE_UPLOAD.read(), np.uint8)
                st.session_state.OPENCV_IMAGE = cv2.imdecode(
                    img_array, cv2.IMREAD_COLOR)
                # process image and display
                st.session_state.MODEL_INPUT = dip.process_image(
                    st.session_state.OPENCV_IMAGE)
                processed_image_container.image(
                    dip.process_image(st.session_state.OPENCV_IMAGE), caption='Ảnh đã xử lý')
            else:
                message_container.error('Vui lòng upload ảnh cần xử lý')

        if st.sidebar.button("Reset"):
            if IMAGE_UPLOAD is not None:
                # Lấy file từ uploader chuyển về file mà openCV đọc được sau đó truyền vào hàm dip.process_image
                img_array = np.frombuffer(IMAGE_UPLOAD.read(), np.uint8)
                st.session_state.OPENCV_IMAGE = cv2.imdecode(
                    img_array, cv2.IMREAD_COLOR)
                # process image and display
                st.session_state.MODEL_INPUT = dip.process_image(
                    st.session_state.OPENCV_IMAGE)
                processed_image_container.image(
                    dip.process_image(st.session_state.OPENCV_IMAGE), caption='Ảnh đã xử lý')
            else:
                message_container.error('Vui lòng upload ảnh cần xử lý')

        # Giá trị của slider 1
        param1 = st.sidebar.slider(
            "Chức năng 1", 0, 8, 2, help="Nội dung dấu ? ở slider 1", key=1)

        # Giá trị của slider 2
        param2 = st.sidebar.slider(
            "Chức năng 2", 0, 8, 2, help="Nội dung dấu ? ở slider 2", key=2)

        if st.sidebar.button("Nhận diện văn bản", type="primary"):
            if (option == 'CRNN + LTSM + CTC'):
                # Xử lý khi nút [Nhận diện văn bản] được nhấn
                if st.session_state.MODEL_INPUT is not None:
                    processed_image_container.image(
                        dip.process_image(st.session_state.OPENCV_IMAGE))
                    # Dự đoán chuỗi
                    PREDICTION_STR = ocr.prediction_ocr_crnn_ctc(
                        dip.convert_img_to_input(st.session_state.MODEL_INPUT))
                    # Đưa chuổi dự đoán vào khung kết quả
                    with result_container:
                        rs_txt = st.text_input(
                            "Kết quả dự đoán", PREDICTION_STR)
                        if st.button("Copy and reset", type="secondary"):
                            pyperclip.copy("Chao em")
                else:
                    message_container.error(
                        'Vui lòng upload hoặc xử lý ảnh đầu vào')
            else:
                if IMAGE_UPLOAD is not None:
                    image = Image.open(IMAGE_UPLOAD)
                    PREDICTION_STR = ocr.prediction_ocr_vietocr(image)
                    with result_container:
                        rs_txt = st.text_input(
                            "Kết quả dự đoán", PREDICTION_STR)
                        if st.button("Copy and reset", type="secondary"):
                            pyperclip.copy(rs_txt)
                else:
                    message_container.error(
                        'Vui lòng upload hoặc xử lý ảnh đầu vào')

        st.title("Tên thành viên thực hiện")
        st.text("20110457 - Trần Tiến Đạt")
        st.text("20110132 - Nguyễn Minh Cường")
        st.text("19110202 - Đỗ Thanh Hiếu")
        st.text("19110037 - Huỳnh Minh Thông")
        st.title("GVHD: PGS.TS Hoàng Văn Dũng")


if __name__ == "__main__":
    main()
