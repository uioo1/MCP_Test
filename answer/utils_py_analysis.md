# utils.py 코드 상세 해설

`utils.py`는 시스템 전반에서 사용되는 유틸리티 함수들을 제공하는 모듈입니다. 문서 로딩, 텍스트 분할, API 키 검증 등의 보조 기능을 담당합니다.

## 코드 라인별 상세 해설

```python
import tempfile
# 임시 파일을 생성하고 관리하기 위한 모듈입니다.
# 업로드된 파일을 임시로 저장할 때 사용됩니다.

import os
# 운영체제와 상호작용하기 위한 모듈입니다.
# 파일 시스템 조작, 임시 파일 삭제 등에 사용됩니다.

from typing import List
# Python의 타입 힌팅을 위한 모듈입니다.
# List: 리스트 타입을 명시하는데 사용됩니다.

from langchain.document_loaders import PyPDFLoader, TextLoader
# LangChain의 문서 로더들을 import합니다.
# PyPDFLoader: PDF 파일을 로드하는 로더
# TextLoader: 텍스트 파일을 로드하는 로더

from langchain.text_splitter import RecursiveCharacterTextSplitter
# LangChain의 텍스트 분할기를 import합니다.
# 긴 문서를 검색에 적합한 크기로 분할하는데 사용됩니다.

from langchain.schema import Document
# LangChain의 Document 스키마를 import합니다.
# 문서의 내용과 메타데이터를 구조화하여 저장합니다.

import streamlit as st
# Streamlit 프레임워크를 import합니다.
# 경고 메시지 표시 등에 사용됩니다.

def load_documents(uploaded_files) -> List[Document]:
    """
    업로드된 파일들을 LangChain Document 객체로 로드합니다.
    
    Args:
        uploaded_files: Streamlit의 UploadedFile 객체 리스트
    
    Returns:
        List[Document]: 로드된 문서 리스트
    """
    # Streamlit을 통해 업로드된 파일들을 처리하여 Document 객체로 변환하는 함수입니다.
    # 다양한 파일 형식을 지원하고 통합된 형태로 반환합니다.
    
    all_documents = []
    # 모든 로드된 문서를 저장할 리스트를 초기화합니다.
    
    for uploaded_file in uploaded_files:
        # 업로드된 각 파일에 대해 반복처리합니다.
        
        file_extension = uploaded_file.name.split('.')[-1].lower()
        # 파일명에서 확장자를 추출하고 소문자로 변환합니다.
        # 'document.PDF' → 'pdf'
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_file:
            # 임시 파일을 생성합니다.
            # delete=False: 파일을 자동으로 삭제하지 않음
            # suffix: 원본 파일과 같은 확장자 사용
            
            tmp_file.write(uploaded_file.getbuffer())
            # Streamlit UploadedFile 객체의 내용을 임시 파일에 씁니다.
            
            tmp_file_path = tmp_file.name
            # 임시 파일의 경로를 저장합니다.
        
        try:
            # 예외 처리를 위한 try-except-finally 블록을 시작합니다.
            
            if file_extension == 'pdf':
                # PDF 파일인 경우
                loader = PyPDFLoader(tmp_file_path)
                # PyPDFLoader 인스턴스를 생성합니다.
                documents = loader.load()
                # PDF 파일을 로드하여 Document 객체들을 생성합니다.
                
            elif file_extension == 'txt':
                # 텍스트 파일인 경우
                loader = TextLoader(tmp_file_path, encoding='utf-8')
                # TextLoader 인스턴스를 생성합니다 (UTF-8 인코딩).
                documents = loader.load()
                # 텍스트 파일을 로드하여 Document 객체들을 생성합니다.
                
            else:
                # 지원하지 않는 파일 형식인 경우
                st.warning(f"지원하지 않는 파일 형식입니다: {uploaded_file.name}")
                # 사용자에게 경고 메시지를 표시합니다.
                continue
                # 현재 파일을 건너뛰고 다음 파일로 진행합니다.
            
            for doc in documents:
                # 로드된 각 문서에 대해 처리합니다.
                doc.metadata['source'] = uploaded_file.name
                # 문서의 메타데이터에 원본 파일명을 추가합니다.
                # 나중에 출처를 추적할 수 있게 됩니다.
            
            all_documents.extend(documents)
            # 로드된 문서들을 전체 문서 리스트에 추가합니다.
            
        finally:
            # 예외 발생 여부와 관계없이 실행되는 블록입니다.
            os.unlink(tmp_file_path)
            # 임시 파일을 삭제하여 디스크 공간을 확보합니다.
    
    return all_documents
    # 모든 로드된 문서들의 리스트를 반환합니다.

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
    # 긴 문서를 검색에 적합한 크기로 분할하는 함수입니다.
    # 벡터 검색의 정확도와 효율성을 향상시키기 위해 사용됩니다.
    
    text_splitter = RecursiveCharacterTextSplitter(
        # 재귀적 문자 기반 텍스트 분할기를 생성합니다.
        # 의미적으로 관련된 텍스트를 함께 유지하려고 시도합니다.
        
        chunk_size=chunk_size,
        # 각 청크의 최대 문자 수를 설정합니다.
        # 기본값: 1000자
        
        chunk_overlap=chunk_overlap,
        # 인접한 청크 간에 중복되는 문자 수를 설정합니다.
        # 기본값: 200자 (컨텍스트 연속성 보장)
        
        length_function=len,
        # 텍스트 길이를 측정하는 함수를 지정합니다.
        # len 함수를 사용하여 문자 수로 측정합니다.
        
        separators=["\n\n", "\n", " ", ""]
        # 텍스트 분할 시 우선순위를 가진 구분자들을 설정합니다.
        # 1순위: 단락 구분 (\n\n)
        # 2순위: 줄 바꿈 (\n)
        # 3순위: 공백 ( )
        # 4순위: 문자 단위 ("")
    )
    
    split_docs = text_splitter.split_documents(documents)
    # 문서 리스트를 분할기에 전달하여 청크로 분할합니다.
    # 각 Document 객체의 메타데이터는 보존됩니다.
    
    return split_docs
    # 분할된 문서 청크들의 리스트를 반환합니다.

def validate_api_key(api_key: str) -> bool:
    """
    OpenAI API 키의 기본 유효성을 검증합니다.
    
    Args:
        api_key: 검증할 API 키
    
    Returns:
        bool: 유효성 여부
    """
    # OpenAI API 키의 기본적인 형식을 검증하는 함수입니다.
    # 실제 API 호출 없이 기본 형식만 확인합니다.
    
    if not api_key:
        # API 키가 비어있거나 None인 경우
        return False
        # False를 반환하여 유효하지 않음을 나타냅니다.
    
    if not api_key.startswith('sk-'):
        # API 키가 'sk-'로 시작하지 않는 경우
        # OpenAI API 키는 모두 'sk-'로 시작합니다.
        return False
        # False를 반환하여 유효하지 않음을 나타냅니다.
    
    if len(api_key) < 20:
        # API 키의 길이가 20자 미만인 경우
        # OpenAI API 키는 일반적으로 20자 이상입니다.
        return False
        # False를 반환하여 유효하지 않음을 나타냅니다.
    
    return True
    # 모든 기본 검증을 통과한 경우 True를 반환합니다.
```

## 주요 특징

1. **파일 처리**: 다양한 형식(PDF, TXT)의 파일을 통합 처리
2. **임시 파일 관리**: 안전한 임시 파일 생성 및 정리
3. **텍스트 분할**: 검색 최적화를 위한 스마트 청크 분할
4. **유효성 검증**: API 키의 기본 형식 검증
5. **에러 처리**: 파일 처리 중 발생할 수 있는 예외 상황 대응

## 파일 처리 워크플로우

1. **업로드된 파일 수신**: Streamlit UploadedFile 객체 처리
2. **임시 파일 생성**: 로컬 파일 시스템에 임시 저장
3. **형식별 로딩**: PDF/TXT에 맞는 적절한 로더 선택
4. **메타데이터 추가**: 원본 파일명 등 추가 정보 보관
5. **임시 파일 정리**: 처리 완료 후 임시 파일 삭제

## 텍스트 분할 전략

### 분할 기준
- **청크 크기**: 1000자 (벡터 임베딩에 적합한 크기)
- **중복 영역**: 200자 (컨텍스트 연속성 보장)
- **구분자 우선순위**: 단락 → 줄바꿈 → 공백 → 문자

### 분할의 장점
- 검색 정확도 향상
- 메모리 사용량 최적화
- 관련성 높은 결과 제공
- 긴 문서의 효율적 처리

## API 키 검증 로직

### 검증 항목
1. **존재성**: 키가 비어있지 않은지 확인
2. **접두사**: 'sk-'로 시작하는지 확인
3. **길이**: 최소 20자 이상인지 확인

### 검증의 한계
- 형식적 검증만 수행 (실제 유효성은 API 호출 시 확인)
- 보안상 실제 API 호출을 피함
- 사용자 경험 향상을 위한 기본 피드백 제공

## 보안 고려사항

1. **임시 파일 관리**: 처리 후 즉시 삭제하여 정보 유출 방지
2. **파일 형식 제한**: 지원하는 형식만 허용하여 보안 위험 감소
3. **메모리 효율성**: 대용량 파일 처리 시 메모리 사용량 최적화
4. **에러 핸들링**: 예외 상황에서도 리소스 정리 보장

## 확장 가능성

- 새로운 파일 형식 지원 추가 용이
- 분할 전략 커스터마이징 가능
- 추가 메타데이터 필드 확장 가능
- 다양한 검증 규칙 추가 가능