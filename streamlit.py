
import streamlit as st


st.title("Nhận dạng chữ viết tay Tiếng Việt")
st.markdown('<div style="margin-top:2rem"></div>', unsafe_allow_html=True)  # Insert a blank line
st.text("Hướng dẫn sử dụng: Upload ảnh cần nhận diện sau đó chọn xử lý tự động")
st.text("hoặc tự chỉnh bên trái màn hình sau đó chọn nhận diện hệ thống sẽ trả ra kết quả")
st.markdown('<div style="margin-top:2rem"></div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Định dạng cho phép: JPG, PNG, JPEG - Độ phân giải 1200x500", type=['png', 'jpg', 'jpeg'])
st.markdown('<div style="margin-top:3rem"></div>', unsafe_allow_html=True)
st.header("Ảnh đã xử lý")
processed_image_container = st.empty()
processed_image_container.markdown(
    '<div style="margin-top:5rem"></div>', unsafe_allow_html=True)
st.markdown('<div style="margin-top:2rem"></div>', unsafe_allow_html=True)
st.header("Kết quả")
st.markdown('<div style="margin-top:1rem"></div>', unsafe_allow_html=True)
result_container = st.empty()
result_container.code("st.code")
# Sidebar
with st.sidebar:
    st.title("Chức năng")
    if st.sidebar.button("Tự động xử lý ảnh", type="primary"):
        # Xử lý khi nút [Tự động xử lý ảnh] được nhấn
        if uploaded_file is not None:
            # Các lệnh xử lý
            processed_image_container.image(uploaded_file)
            result_container.code("Kết quả sau xử lý")

    # Giá trị của slider 1
    param1 = st.sidebar.slider(
        "Chức năng 1", 0, 8, 2, help="Nội dung dấu ? ở slider 1", key=1)

    # Giá trị của slider 2
    param2 = st.sidebar.slider(
        "Chức năng 2", 0, 8, 2, help="Nội dung dấu ? ở slider 2", key=2)

    if st.sidebar.button("Nhận diện văn bản", type="primary"):
        # Các lệnh xử lý, dùng các biến số param1, param2
        pass  # Xử lý khi nút [Nhận diện văn bản] được nhấn

    st.markdown('<div style="margin-top:5rem"></div>', unsafe_allow_html=True)
    st.text("Tên thành viên thực hiện")
    for x in range(4):
        st.text("x")
