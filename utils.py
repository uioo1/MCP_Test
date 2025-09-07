import tempfile
import os
from typing import List
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import streamlit as st


def load_documents(uploaded_files) -> List[Document]:
    """
    업로드된 파일들을 LangChain Document 객체로 로드합니다.
    
    Args:
        uploaded_files: Streamlit의 UploadedFile 객체 리스트
    
    Returns:
        List[Document]: 로드된 문서 리스트
    """
    all_documents = []
    
    for uploaded_file in uploaded_files:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_file_path = tmp_file.name
        
        try:
            if file_extension == 'pdf':
                loader = PyPDFLoader(tmp_file_path)
                documents = loader.load()
            elif file_extension == 'txt':
                loader = TextLoader(tmp_file_path, encoding='utf-8')
                documents = loader.load()
            else:
                st.warning(f"지원하지 않는 파일 형식입니다: {uploaded_file.name}")
                continue
            
            for doc in documents:
                doc.metadata['source'] = uploaded_file.name
            
            all_documents.extend(documents)
            
        finally:
            os.unlink(tmp_file_path)
    
    return all_documents


def split_text(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """
    문서를 적절한 크기의 청크로 분할합니다.
    
    Args:
        documents: 분할할 문서 리스트
        chunk_size: 각 청크의 최대 크기
        chunk_overlap: 청크 간 중복 문자 수
    
    Returns:
        List[Document]: 분할된 문서 청크 리스트
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    split_docs = text_splitter.split_documents(documents)
    return split_docs


def validate_api_key(api_key: str) -> bool:
    """
    OpenAI API 키의 기본 유효성을 검증합니다.
    
    Args:
        api_key: 검증할 API 키
    
    Returns:
        bool: 유효성 여부
    """
    if not api_key:
        return False
    
    if not api_key.startswith('sk-'):
        return False
    
    if len(api_key) < 20:
        return False
    
    return True