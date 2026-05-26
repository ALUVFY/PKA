import streamlit as st
from docx import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tempfile
import os

st.title("我的AI知识助手")

uploaded_file = st.file_uploader(
    "上传文件",
    type=["pdf", "docx"]
)

if uploaded_file:

    suffix = uploaded_file.name.split(".")[-1]

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix="." + suffix
    ) as tmp:

        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    text = ""

    try:

        # PDF读取
        if suffix == "pdf":

            loader = PyPDFLoader(temp_path)

            docs = loader.load()

            text = "\n".join(
                [doc.page_content for doc in docs]
            )

        # Word读取
        elif suffix == "docx":

            doc = Document(temp_path)

            text = "\n".join(
                [p.text for p in doc.paragraphs]
            )

        # 显示原始内容
        st.success("文件读取成功")

        st.text_area(
            "文件内容预览",
            text,
            height=300
        )

        # 文本切块
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )

        chunks = splitter.split_text(text)

        st.success(
            f"切块成功，共 {len(chunks)} 块"
        )

        # 展示前5个文本块
        for i, chunk in enumerate(chunks[:5]):

            st.text_area(
                f"文本块 {i+1}",
                chunk,
                height=120
            )

    except Exception as e:

        st.error(
            f"发生错误: {str(e)}"
        )

    finally:

        # 删除临时文件
        if os.path.exists(temp_path):
            os.remove(temp_path)