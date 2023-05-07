import streamlit as st
import numpy as np
import cv2
from PIL import Image
import pyperclip
from streamlit_cropper import st_cropper


import module.process_image as dip
import ocr_app as ocr


def init_session_var():
    if 'MODEL_OPTION' not in st.session_state:
        st.session_state.MODEL_OPTION = None

    if 'MODEL_INPUT' not in st.session_state:
        st.session_state.MODEL_INPUT = None

    if 'PREDICTION_STR' not in st.session_state:
        st.session_state.PREDICTION_STR = None

    if 'OPENCV_IMAGE' not in st.session_state:
        st.session_state.OPENCV_IMAGE = None

    if 'OLD_DIL' not in st.session_state:
        st.session_state.OLD_DIL = None

    if 'OLD_ERO' not in st.session_state:
        st.session_state.OLD_ERO = None

    if 'RESIZE_ENABLE' not in st.session_state:
        st.session_state.RESIZE_ENABLE = None


def reset():
    st.session_state.IMG_DATA = None
    st.session_state.MODEL_INPUT = None
    st.session_state.PREDICTION_STR = None
    st.session_state.RESIZE_ENABLE = False
    st.experimental_rerun()


def main():
    # init session state
    init_session_var()

    st.set_page_config(
        'Vietnamese Handwritten Recognition (OCR)', './img_streamlit/logo.ico')
    message_container = st.empty()
    st.title("Nhận dạng chữ viết tay Tiếng Việt")
    st.markdown('<div style="margin-top:2rem"></div>',
                unsafe_allow_html=True)  # Insert a blank line
    st.markdown('<p style="text-align: justify; font-size:20px;">Hướng dẫn sử dụng: Upload ảnh cần nhận diện sau đó chọn xử lý ảnh đầu vào.' +
                ' Người dùng có thể thay đổi kích thước (resize) hoặc sử dụng phép co giãn (erosion/dilation) để tăng mức độ chính xác,' +
                ' ngoài ra sau khi dự đoán người dùng có thể chỉnh sửa kết quả cho chính xác hơn.</p>', unsafe_allow_html=True)

    st.markdown('<div style="margin-top:2rem"></div>', unsafe_allow_html=True)

    IMAGE_UPLOAD = st.file_uploader(
        "Định dạng cho phép: JPG, PNG, JPEG - Độ phân giải 2200x200", type=['png', 'jpg', 'jpeg'])
    # Display image
    if IMAGE_UPLOAD is not None:
        st.image(IMAGE_UPLOAD, caption='Ảnh đã tải lên', use_column_width=True)
        if (st.session_state.RESIZE_ENABLE == True):
            if st.button("Hủy"):
                st.session_state.RESIZE_ENABLE = False
                st.experimental_rerun()
            img = Image.open(IMAGE_UPLOAD)
            cropped_img = st_cropper(img, realtime_update=True, box_color='#0000FF',
                                     aspect_ratio=None)
            np_image = np.asarray(cropped_img)
            # Chuyển đổi định dạng hình ảnh từ RGB sang BGR
            st.session_state.OPENCV_IMAGE = cv2.cvtColor(np_image,
                                                         cv2.COLOR_RGB2BGR)
        else:
            if st.button("Resize"):
                st.session_state.MODEL_INPUT = None
                st.session_state.RESIZE_ENABLE = True
                st.experimental_rerun()

    processed_image_container = st.empty()
    if st.session_state.MODEL_INPUT is None:
        processed_image_container.markdown(
            '<div style="margin-top:2rem"></div>', unsafe_allow_html=True)
    else:
        processed_image_container.image(
            st.session_state.MODEL_INPUT, caption='Ảnh đã xử lý')

    st.markdown('<div style="margin-top:2rem"></div>', unsafe_allow_html=True)
    st.header("Kết quả")
    st.markdown('<div style="margin-top:1rem"></div>', unsafe_allow_html=True)

    if st.session_state.PREDICTION_STR is not None:
        st.text(
            "Click [Copy and reset] để sao chép và reset lại toàn bộ ứng dụng")
        st.session_state.PREDICTION_STR = st.text_input(
            "Kết quả dự đoán", st.session_state.PREDICTION_STR)
        if st.button("Copy and reset", type="secondary"):
            pyperclip.copy(st.session_state.PREDICTION_STR)
            st.session_state.IMG_DATA = None
            st.session_state.MODEL_INPUT = None
            st.session_state.PREDICTION_STR = None
            st.experimental_rerun()

    # Sidebar
    with st.sidebar:

        st.title("Chức năng")

        if st.sidebar.button("Xử lý ảnh đầu vào", type="primary"):
            # Xử lý khi nút [Tự động xử lý ảnh] được nhấn
            if IMAGE_UPLOAD is not None:
                if (st.session_state.OPENCV_IMAGE is None):
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
            st.experimental_rerun()
            # Giá trị của slider 1
        param1 = st.sidebar.slider(
            "Co đối tượng trong ảnh", 0, 8, 0, help="Tăng hoặc giảm kenel", key=1)
        if (st.session_state.MODEL_INPUT is not None
                and st.session_state.OLD_ERO != param1):
            st.session_state.OLD_ERO = param1
            st.session_state.MODEL_INPUT = dip.erosion_dilation_image(st.session_state.MODEL_INPUT,
                                                                      param1, True)
            processed_image_container.image(st.session_state.MODEL_INPUT)

        # Giá trị của slider 2
        param2 = st.sidebar.slider(
            "Giãn đối tượng trong ảnh", 0, 8, 0, help="Tăng hoặc giảm kenel", key=2)
        if (st.session_state.MODEL_INPUT is not None
                and st.session_state.OLD_DIL != param2):
            st.session_state.OLD_DIL = param2
            st.session_state.MODEL_INPUT = dip.erosion_dilation_image(st.session_state.MODEL_INPUT,
                                                                      param2, False)
            processed_image_container.image(st.session_state.MODEL_INPUT)
        option = st.selectbox('Chọn model để nhận diện',
                              ('CRNN + LTSM + CTC', 'AttentionOCR (VietOCR)'))
        if st.sidebar.button("Nhận diện văn bản", type="primary"):
            if (option == 'CRNN + LTSM + CTC'):
                # Xử lý khi nút [Nhận diện văn bản] được nhấn
                if st.session_state.MODEL_INPUT is not None:
                    processed_image_container.image(
                        st.session_state.MODEL_INPUT, caption='Ảnh đã xử lý')
                    # Dự đoán chuỗi
                    st.session_state.PREDICTION_STR = ocr.prediction_ocr_crnn_ctc(
                        dip.convert_img_to_input(st.session_state.MODEL_INPUT))
                    st.experimental_rerun()
                else:
                    message_container.error(
                        'Vui lòng upload hoặc xử lý ảnh đầu vào')
            else:
                if IMAGE_UPLOAD is not None:
                    image = Image.open(IMAGE_UPLOAD)
                    st.session_state.PREDICTION_STR = ocr.prediction_ocr_vietocr(
                        image)
                    st.experimental_rerun()
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
