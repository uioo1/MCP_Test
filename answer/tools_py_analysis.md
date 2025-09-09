# tools.py 코드 상세 해설

`tools.py`는 RAG(Retrieval-Augmented Generation) 시스템의 핵심 도구들을 제공하는 모듈입니다. 문서 처리, 벡터 데이터베이스 생성, 저장/로드, 문서 검색 기능을 담당합니다.

## 코드 라인별 상세 해설

```python
import os
# 운영체제와 상호작용하기 위한 모듈입니다.
# 파일 시스템 조작, 디렉토리 생성, 파일 존재 확인 등에 사용됩니다.

from typing import List, Optional, Any
# Python의 타입 힌팅을 위한 모듈입니다.
# List: 리스트 타입, Optional: 선택적 타입, Any: 모든 타입을 나타냅니다.

from langchain.vectorstores import FAISS
# LangChain의 FAISS 벡터 스토어를 import합니다.
# Facebook AI에서 개발한 고성능 벡터 유사도 검색 라이브러리입니다.

from langchain.embeddings import OpenAIEmbeddings
# LangChain의 OpenAI 임베딩 모델을 import합니다.
# 텍스트를 벡터로 변환하는 기능을 제공합니다.

from langchain.schema import Document
# LangChain의 Document 스키마를 import합니다.
# 문서의 내용과 메타데이터를 구조화하여 저장합니다.

import utils
# 유틸리티 함수들을 제공하는 모듈입니다.
# 문서 로딩, 텍스트 분할 등의 기능을 사용합니다.

import pickle
# Python 객체를 직렬화/역직렬화하는 모듈입니다.
# 객체를 파일로 저장하거나 불러오는데 사용됩니다.

def create_retriever(files, embeddings: Optional[OpenAIEmbeddings] = None) -> Any:
    """
    업로드된 파일들로부터 RAG Retriever를 생성합니다.
    
    Args:
        files: 업로드된 파일 리스트
        embeddings: 임베딩 모델 (없으면 새로 생성)
    
    Returns:
        Any: 검색기 객체
    """
    # RAG 시스템의 핵심인 검색기(retriever)를 생성하는 함수입니다.
    # 업로드된 파일들을 처리하여 벡터 검색이 가능한 형태로 변환합니다.
    
    if embeddings is None:
        # 임베딩 모델이 제공되지 않은 경우
        embeddings = OpenAIEmbeddings()
        # 새로운 OpenAI 임베딩 모델 인스턴스를 생성합니다.
    
    documents = utils.load_documents(files)
    # utils 모듈의 load_documents 함수를 호출하여 파일들을 Document 객체로 변환합니다.
    
    if not documents:
        # 문서가 로드되지 않은 경우 (빈 리스트인 경우)
        raise ValueError("문서를 로드할 수 없습니다.")
        # ValueError 예외를 발생시켜 에러 상황을 알립니다.
    
    split_docs = utils.split_text(documents)
    # utils 모듈의 split_text 함수를 호출하여 문서를 검색에 적합한 크기로 분할합니다.
    
    vector_store = FAISS.from_documents(
        # FAISS 벡터 스토어를 문서들로부터 생성합니다.
        documents=split_docs,
        # 분할된 문서들을 전달합니다.
        embedding=embeddings
        # 사용할 임베딩 모델을 전달합니다.
    )
    
    save_vector_store(vector_store, embeddings)
    # 생성된 벡터 스토어를 로컬 파일로 저장합니다.
    
    retriever = vector_store.as_retriever(
        # 벡터 스토어를 검색기 객체로 변환합니다.
        search_type="similarity",
        # 검색 방식을 유사도 기반으로 설정합니다.
        search_kwargs={"k": 3}
        # 검색 결과로 가장 유사한 3개의 문서를 반환하도록 설정합니다.
    )
    
    return retriever
    # 생성된 검색기 객체를 반환합니다.

def load_existing_retriever(embeddings: Optional[OpenAIEmbeddings] = None) -> Optional[Any]:
    """
    저장된 벡터 스토어가 있으면 로드합니다.
    
    Args:
        embeddings: 임베딩 모델
    
    Returns:
        Any: 검색기 객체 또는 None
    """
    # 이전에 저장된 벡터 스토어가 있으면 로드하여 재사용하는 함수입니다.
    # 매번 새로 생성할 필요없이 기존 지식베이스를 활용할 수 있습니다.
    
    index_path = "faiss_index/index.faiss"
    # FAISS 인덱스 파일의 경로를 설정합니다.
    metadata_path = "faiss_index/index.pkl"
    # 메타데이터 파일의 경로를 설정합니다.
    
    if not os.path.exists(index_path) or not os.path.exists(metadata_path):
        # 인덱스 파일이나 메타데이터 파일이 존재하지 않으면
        return None
        # None을 반환하여 기존 데이터가 없음을 나타냅니다.
    
    if embeddings is None:
        # 임베딩 모델이 제공되지 않은 경우
        embeddings = OpenAIEmbeddings()
        # 새로운 OpenAI 임베딩 모델 인스턴스를 생성합니다.
    
    try:
        # 예외 처리를 위한 try-except 블록을 시작합니다.
        vector_store = FAISS.load_local(
            # 로컬에 저장된 FAISS 벡터 스토어를 로드합니다.
            "faiss_index",
            # 저장된 디렉토리명을 지정합니다.
            embeddings,
            # 사용할 임베딩 모델을 전달합니다.
            allow_dangerous_deserialization=True
            # 역직렬화를 허용합니다 (보안상 주의가 필요한 옵션).
        )
        
        retriever = vector_store.as_retriever(
            # 로드된 벡터 스토어를 검색기 객체로 변환합니다.
            search_type="similarity",
            # 검색 방식을 유사도 기반으로 설정합니다.
            search_kwargs={"k": 3}
            # 검색 결과로 가장 유사한 3개의 문서를 반환하도록 설정합니다.
        )
        
        return retriever
        # 로드된 검색기 객체를 반환합니다.
    except Exception as e:
        # 예외가 발생한 경우 처리합니다.
        print(f"벡터 스토어 로드 실패: {e}")
        # 콘솔에 에러 메시지를 출력합니다.
        return None
        # None을 반환하여 로드 실패를 나타냅니다.

def save_vector_store(vector_store: FAISS, embeddings: OpenAIEmbeddings):
    """
    벡터 스토어를 로컬에 저장합니다.
    
    Args:
        vector_store: FAISS 벡터 스토어
        embeddings: 사용된 임베딩 모델
    """
    # 생성된 벡터 스토어를 로컬 파일 시스템에 저장하는 함수입니다.
    # 다음 실행 시 재사용할 수 있도록 영속성을 제공합니다.
    
    os.makedirs("faiss_index", exist_ok=True)
    # "faiss_index" 디렉토리를 생성합니다.
    # exist_ok=True로 설정하여 이미 존재하는 경우 에러를 발생시키지 않습니다.
    
    vector_store.save_local("faiss_index")
    # FAISS 벡터 스토어를 "faiss_index" 디렉토리에 저장합니다.
    # 인덱스와 메타데이터가 별도 파일로 저장됩니다.
    
    print("벡터 스토어가 저장되었습니다.")
    # 저장 완료 메시지를 콘솔에 출력합니다.

def search_documents(retriever: Any, query: str) -> List[Document]:
    """
    주어진 쿼리로 문서를 검색합니다.
    
    Args:
        retriever: 검색기 객체
        query: 검색 쿼리
    
    Returns:
        List[Document]: 검색된 문서 리스트
    """
    # 사용자의 질의를 받아 관련 문서를 검색하는 함수입니다.
    # RAG 시스템에서 검색 단계를 담당합니다.
    
    try:
        # 예외 처리를 위한 try-except 블록을 시작합니다.
        documents = retriever.get_relevant_documents(query)
        # 검색기의 get_relevant_documents 메서드를 호출하여 관련 문서를 검색합니다.
        # 쿼리와 유사도가 높은 문서들을 반환받습니다.
        return documents
        # 검색된 문서 리스트를 반환합니다.
    except Exception as e:
        # 예외가 발생한 경우 처리합니다.
        print(f"문서 검색 중 오류 발생: {e}")
        # 콘솔에 에러 메시지를 출력합니다.
        return []
        # 빈 리스트를 반환하여 검색 실패를 나타냅니다.
```

## 주요 특징

1. **RAG 시스템 구현**: 문서 기반 검색 증강 생성 시스템의 핵심 기능
2. **벡터 검색**: FAISS를 활용한 고성능 유사도 검색
3. **데이터 영속성**: 생성된 벡터 스토어를 로컬에 저장/로드
4. **에러 처리**: 각 단계별 예외 상황 대응
5. **모듈화**: 각 기능을 독립적인 함수로 분리

## 데이터 플로우

1. **문서 처리**: 업로드된 파일 → Document 객체 변환
2. **텍스트 분할**: 긴 문서를 검색 가능한 청크로 분할
3. **벡터화**: 텍스트 청크를 임베딩 벡터로 변환
4. **인덱싱**: FAISS 인덱스 생성 및 저장
5. **검색**: 쿼리 기반 유사도 검색 수행

## 핵심 기술

- **FAISS**: Facebook AI의 고성능 벡터 검색 라이브러리
- **OpenAI Embeddings**: 텍스트의 의미적 벡터 표현 생성
- **Document Chunking**: 검색 효율성을 위한 텍스트 분할
- **Similarity Search**: 코사인 유사도 기반 문서 검색
- **Persistence**: 벡터 인덱스의 영속적 저장

## 성능 최적화

- 검색 결과를 상위 3개로 제한하여 응답 속도 향상
- 벡터 스토어 캐싱으로 재처리 시간 절약
- 예외 처리로 시스템 안정성 확보
- 메모리 효율적인 문서 처리 방식 채택