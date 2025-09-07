import os
from typing import List, Optional, Any
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
import utils
import pickle


def create_retriever(files, embeddings: Optional[OpenAIEmbeddings] = None) -> Any:
    """
    업로드된 파일들로부터 RAG Retriever를 생성합니다.
    
    Args:
        files: 업로드된 파일 리스트
        embeddings: 임베딩 모델 (없으면 새로 생성)
    
    Returns:
        Any: 검색기 객체
    """
    if embeddings is None:
        embeddings = OpenAIEmbeddings()
    
    documents = utils.load_documents(files)
    
    if not documents:
        raise ValueError("문서를 로드할 수 없습니다.")
    
    split_docs = utils.split_text(documents)
    
    vector_store = FAISS.from_documents(
        documents=split_docs,
        embedding=embeddings
    )
    
    save_vector_store(vector_store, embeddings)
    
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    
    return retriever


def load_existing_retriever(embeddings: Optional[OpenAIEmbeddings] = None) -> Optional[Any]:
    """
    저장된 벡터 스토어가 있으면 로드합니다.
    
    Args:
        embeddings: 임베딩 모델
    
    Returns:
        Any: 검색기 객체 또는 None
    """
    index_path = "faiss_index/index.faiss"
    metadata_path = "faiss_index/index.pkl"
    
    if not os.path.exists(index_path) or not os.path.exists(metadata_path):
        return None
    
    if embeddings is None:
        embeddings = OpenAIEmbeddings()
    
    try:
        vector_store = FAISS.load_local(
            "faiss_index",
            embeddings,
            allow_dangerous_deserialization=True
        )
        
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        
        return retriever
    except Exception as e:
        print(f"벡터 스토어 로드 실패: {e}")
        return None


def save_vector_store(vector_store: FAISS, embeddings: OpenAIEmbeddings):
    """
    벡터 스토어를 로컬에 저장합니다.
    
    Args:
        vector_store: FAISS 벡터 스토어
        embeddings: 사용된 임베딩 모델
    """
    os.makedirs("faiss_index", exist_ok=True)
    
    vector_store.save_local("faiss_index")
    
    print("벡터 스토어가 저장되었습니다.")


def search_documents(retriever: Any, query: str) -> List[Document]:
    """
    주어진 쿼리로 문서를 검색합니다.
    
    Args:
        retriever: 검색기 객체
        query: 검색 쿼리
    
    Returns:
        List[Document]: 검색된 문서 리스트
    """
    try:
        documents = retriever.get_relevant_documents(query)
        return documents
    except Exception as e:
        print(f"문서 검색 중 오류 발생: {e}")
        return []