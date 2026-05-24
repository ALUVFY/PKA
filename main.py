import streamlit as st
from docx import Document
from langchain_community.document_loaders import PyPDFLoader
import tempfile
import os

st.title("我的AI知识助手")

uploaded_file = st.file_uploader(
    "上传文件",
    type=["pdf","docx"]
)

if uploaded_file:

    suffix=uploaded_file.name.split(".")[-1]

    with tempfile.NamedTemporaryFile(
            delete=False,
            suffix="."+suffix
    ) as tmp:

        tmp.write(uploaded_file.read())

        temp_path=tmp.name

    text=""

    if suffix=="pdf":

        loader=PyPDFLoader(temp_path)

        docs=loader.load()

        text="\n".join(
            [doc.page_content for doc in docs]
        )

    elif suffix=="docx":

        doc=Document(temp_path)

        text="\n".join(
            [p.text for p in doc.paragraphs]
        )

    st.success("文件读取成功")

    st.text_area(
        "文件内容预览",
        text,
        height=300
    )

    os.remove(temp_path)