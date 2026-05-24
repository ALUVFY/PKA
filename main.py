import streamlit as st

st.title("我的AI知识助手")

uploaded_file = st.file_uploader(
    "上传文件"
)

if uploaded_file:
    st.write(
        f"文件名:{uploaded_file.name}"
    )