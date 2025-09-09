# graph.py 코드 상세 해설

`graph.py`는 LangGraph를 활용한 AI 워크플로우를 구현하는 모듈입니다. 질문 분류, 문서 검색, 답변 생성의 전체 파이프라인을 상태 기반 그래프로 관리합니다.

## 코드 라인별 상세 해설

```python
from typing import TypedDict, Literal, List, Optional
# Python의 타입 힌팅을 위한 모듈들을 import합니다.
# TypedDict: 구조화된 딕셔너리 타입을 정의
# Literal: 특정 값들만 허용하는 타입
# List: 리스트 타입, Optional: 선택적 타입

from langchain.schema import BaseMessage, HumanMessage, AIMessage, Document
# LangChain의 기본 스키마들을 import합니다.
# BaseMessage: 메시지의 기본 클래스
# HumanMessage, AIMessage: 사용자와 AI 메시지
# Document: 문서 객체

from langchain_openai import ChatOpenAI
# LangChain의 OpenAI 채팅 모델을 import합니다.
# GPT 모델과 상호작용하기 위해 사용됩니다.

from langgraph.graph import StateGraph, END
# LangGraph의 상태 그래프 관련 클래스들을 import합니다.
# StateGraph: 상태 기반 워크플로우 그래프
# END: 그래프의 종료 노드

import tools
# RAG 시스템의 도구들이 정의된 모듈입니다.
# 문서 검색 기능을 사용합니다.

import streamlit as st
# Streamlit 프레임워크를 import합니다.
# 세션 상태에 접근하기 위해 사용됩니다.

class GraphState(TypedDict):
    """그래프의 상태를 정의하는 TypedDict"""
    # 워크플로우 전체에서 공유되는 상태를 정의하는 클래스입니다.
    # TypedDict를 상속받아 구조화된 데이터를 관리합니다.
    
    question: str
    # 사용자가 입력한 질문을 저장하는 문자열 필드입니다.
    
    classification: Literal["simple", "complex"]
    # 질문의 복잡도 분류 결과를 저장합니다.
    # "simple" 또는 "complex" 값만 허용됩니다.
    
    documents: List[Document]
    # 검색된 관련 문서들을 저장하는 Document 객체 리스트입니다.
    
    answer: str
    # 생성된 AI 응답을 저장하는 문자열 필드입니다.
    
    chat_history: List[BaseMessage]
    # 대화 기록을 저장하는 메시지 객체 리스트입니다.
    
    retriever: Optional[object]
    # RAG 검색기 객체를 저장합니다. None일 수도 있습니다.

def classify_inquiry(state: GraphState) -> GraphState:
    """
    사용자 문의를 simple 또는 complex로 분류합니다.
    
    Args:
        state: 현재 그래프 상태
    
    Returns:
        GraphState: 업데이트된 상태
    """
    # 사용자의 질문을 분석하여 복잡도를 분류하는 함수입니다.
    # 분류 결과에 따라 다른 AI 모델을 사용합니다.
    
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    # GPT-3.5-turbo 모델의 ChatOpenAI 인스턴스를 생성합니다.
    # temperature=0으로 설정하여 일관된 결과를 얻습니다.
    
    classification_prompt = f"""
    다음 고객 문의를 분석하여 'simple' 또는 'complex'로 분류하세요.
    
    분류 기준:
    - simple: 일반적인 질문, FAQ, 간단한 정보 요청, 지식 베이스로 답변 가능한 질문
    - complex: 기술적 문제, 복잡한 요청, 인간 개입이 필요한 문제, 민감한 이슈
    
    문의: {state['question']}
    
    응답은 반드시 'simple' 또는 'complex' 중 하나만 출력하세요.
    """
    # 질문 분류를 위한 프롬프트를 구성합니다.
    # 명확한 분류 기준을 제시하고 정확한 형식의 응답을 요구합니다.
    
    response = llm.predict(classification_prompt)
    # LLM에 프롬프트를 전달하고 응답을 받습니다.
    
    classification = "simple" if "simple" in response.lower() else "complex"
    # 응답에 "simple"이 포함되어 있으면 "simple", 아니면 "complex"로 분류합니다.
    # 대소문자를 구분하지 않기 위해 lower() 메서드를 사용합니다.
    
    state['classification'] = classification
    # 분류 결과를 상태에 저장합니다.
    
    return state
    # 업데이트된 상태를 반환합니다.

def retrieve_documents(state: GraphState) -> GraphState:
    """
    RAG를 사용하여 관련 문서를 검색합니다.
    
    Args:
        state: 현재 그래프 상태
    
    Returns:
        GraphState: 업데이트된 상태
    """
    # 사용자 질문과 관련된 문서를 벡터 데이터베이스에서 검색하는 함수입니다.
    # RAG 시스템의 검색(Retrieval) 단계를 담당합니다.
    
    retriever = state.get('retriever')
    # 상태에서 retriever 객체를 가져옵니다.
    
    if retriever is None:
        # retriever가 없는 경우 (지식베이스가 구축되지 않음)
        state['documents'] = []
        # 빈 문서 리스트를 설정합니다.
        return state
        # 상태를 반환합니다.
    
    try:
        # 예외 처리를 위한 try-except 블록을 시작합니다.
        documents = tools.search_documents(retriever, state['question'])
        # tools 모듈의 search_documents 함수를 호출하여 관련 문서를 검색합니다.
        state['documents'] = documents
        # 검색된 문서들을 상태에 저장합니다.
    except Exception as e:
        # 예외가 발생한 경우 처리합니다.
        print(f"문서 검색 중 오류: {e}")
        # 콘솔에 에러 메시지를 출력합니다.
        state['documents'] = []
        # 빈 문서 리스트를 설정합니다.
    
    return state
    # 업데이트된 상태를 반환합니다.

def generate_rag_answer(state: GraphState) -> GraphState:
    """
    검색된 문서를 바탕으로 답변을 생성합니다.
    
    Args:
        state: 현재 그래프 상태
    
    Returns:
        GraphState: 업데이트된 상태
    """
    # simple 질문에 대해 검색된 문서를 기반으로 답변을 생성하는 함수입니다.
    # RAG 시스템의 생성(Generation) 단계를 담당합니다.
    
    llm = ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo")
    # GPT-3.5-turbo 모델의 ChatOpenAI 인스턴스를 생성합니다.
    # temperature=0.7로 설정하여 창의적인 답변을 생성합니다.
    
    if state['documents']:
        # 검색된 문서가 있는 경우
        context = "\n\n".join([doc.page_content for doc in state['documents'][:3]])
        # 상위 3개 문서의 내용을 컨텍스트로 결합합니다.
        
        prompt = f"""
        다음 컨텍스트를 참고하여 고객 문의에 답변해주세요.
        답변은 친절하고 명확하게 작성해주세요.
        
        컨텍스트:
        {context}
        
        고객 문의: {state['question']}
        
        답변:
        """
        # 컨텍스트를 포함한 프롬프트를 구성합니다.
        # 검색된 문서 내용을 기반으로 답변하도록 지시합니다.
    else:
        # 검색된 문서가 없는 경우
        prompt = f"""
        고객 문의에 대해 일반적인 지식을 바탕으로 답변해주세요.
        답변은 친절하고 명확하게 작성해주세요.
        
        고객 문의: {state['question']}
        
        답변:
        """
        # 일반적인 지식을 기반으로 답변하는 프롬프트를 구성합니다.
    
    answer = llm.predict(prompt)
    # LLM에 프롬프트를 전달하고 답변을 생성받습니다.
    
    state['answer'] = answer
    # 생성된 답변을 상태에 저장합니다.
    
    return state
    # 업데이트된 상태를 반환합니다.

def generate_complex_answer(state: GraphState) -> GraphState:
    """
    복잡한 문의에 대해서도 AI가 답변을 생성합니다.
    
    Args:
        state: 현재 그래프 상태
    
    Returns:
        GraphState: 업데이트된 상태
    """
    # complex 질문에 대해 고급 모델을 사용하여 답변을 생성하는 함수입니다.
    # 더 전문적이고 상세한 답변을 제공합니다.
    
    llm = ChatOpenAI(temperature=0.7, model="gpt-4-turbo-preview")
    # GPT-4-turbo-preview 모델의 ChatOpenAI 인스턴스를 생성합니다.
    # 더 고성능 모델을 사용하여 복잡한 질문에 대응합니다.
    
    if state['documents']:
        # 검색된 문서가 있는 경우
        context = "\n\n".join([doc.page_content for doc in state['documents'][:3]])
        # 상위 3개 문서의 내용을 컨텍스트로 결합합니다.
        
        prompt = f"""
        다음은 복잡한 기술적 문의입니다. 전문적이고 상세한 답변을 제공해주세요.
        가능한 컨텍스트를 참고하되, 없어도 최선을 다해 답변해주세요.
        
        컨텍스트:
        {context}
        
        고객 문의: {state['question']}
        
        답변:
        """
        # 복잡한 질문에 적합한 프롬프트를 구성합니다.
        # 전문적이고 상세한 답변을 요구합니다.
    else:
        # 검색된 문서가 없는 경우
        prompt = f"""
        다음은 복잡한 기술적 문의입니다. 전문적이고 상세한 답변을 제공해주세요.
        
        고객 문의: {state['question']}
        
        답변:
        """
        # 문서 없이도 전문적인 답변을 생성하는 프롬프트를 구성합니다.
    
    answer = llm.predict(prompt)
    # LLM에 프롬프트를 전달하고 답변을 생성받습니다.
    
    state['answer'] = answer
    # 생성된 답변을 상태에 저장합니다.
    
    return state
    # 업데이트된 상태를 반환합니다.

def route_after_classification(state: GraphState) -> str:
    """
    분류 결과에 따라 다음 노드를 결정합니다.
    모든 경우에 문서를 검색하고 답변을 생성합니다.
    
    Args:
        state: 현재 그래프 상태
    
    Returns:
        str: 다음 노드 이름
    """
    # 질문 분류 후 워크플로우의 다음 단계를 결정하는 라우팅 함수입니다.
    # 모든 질문에 대해 문서 검색을 수행합니다.
    
    return 'retrieve_documents'
    # 항상 문서 검색 노드로 진행하도록 설정되어 있습니다.

def build_graph():
    """
    LangGraph 워크플로우를 구축합니다.
    
    Returns:
        CompiledGraph: 컴파일된 그래프 객체
    """
    # 전체 AI 워크플로우를 그래프 형태로 구축하는 함수입니다.
    # 각 단계를 노드로, 흐름을 엣지로 정의합니다.
    
    workflow = StateGraph(GraphState)
    # GraphState를 사용하는 StateGraph 인스턴스를 생성합니다.
    
    workflow.add_node("classify_inquiry", classify_inquiry)
    # 질문 분류 노드를 추가합니다.
    
    workflow.add_node("retrieve_documents", retrieve_documents)
    # 문서 검색 노드를 추가합니다.
    
    workflow.add_node("generate_rag_answer", generate_rag_answer)
    # simple 질문에 대한 답변 생성 노드를 추가합니다.
    
    workflow.add_node("generate_complex_answer", generate_complex_answer)
    # complex 질문에 대한 답변 생성 노드를 추가합니다.
    
    workflow.set_entry_point("classify_inquiry")
    # 워크플로우의 시작점을 질문 분류 노드로 설정합니다.
    
    workflow.add_conditional_edges(
        # 조건부 엣지를 추가합니다.
        "classify_inquiry",
        # 질문 분류 노드에서
        route_after_classification,
        # route_after_classification 함수로 다음 노드를 결정합니다.
        {
            "retrieve_documents": "retrieve_documents"
        }
        # 가능한 다음 노드를 매핑합니다.
    )
    
    def route_after_retrieve(state: GraphState) -> str:
        # 문서 검색 후 다음 단계를 결정하는 내부 함수입니다.
        if state['classification'] == 'simple':
            # simple 질문인 경우
            return 'generate_rag_answer'
            # RAG 답변 생성 노드로 이동합니다.
        else:
            # complex 질문인 경우
            return 'generate_complex_answer'
            # 복잡한 답변 생성 노드로 이동합니다.
    
    workflow.add_conditional_edges(
        # 또 다른 조건부 엣지를 추가합니다.
        "retrieve_documents",
        # 문서 검색 노드에서
        route_after_retrieve,
        # route_after_retrieve 함수로 다음 노드를 결정합니다.
        {
            "generate_rag_answer": "generate_rag_answer",
            "generate_complex_answer": "generate_complex_answer"
        }
        # 가능한 다음 노드들을 매핑합니다.
    )
    
    workflow.add_edge("generate_rag_answer", END)
    # RAG 답변 생성 노드에서 워크플로우를 종료합니다.
    
    workflow.add_edge("generate_complex_answer", END)
    # 복잡한 답변 생성 노드에서 워크플로우를 종료합니다.
    
    return workflow.compile()
    # 정의된 워크플로우를 컴파일하여 실행 가능한 그래프로 변환합니다.

def run_workflow(question: str, retriever=None, chat_history=None):
    """
    워크플로우를 실행합니다.
    
    Args:
        question: 사용자 질문
        retriever: RAG 검색기
        chat_history: 대화 기록
    
    Returns:
        dict: 워크플로우 실행 결과
    """
    # 전체 AI 워크플로우를 실행하는 함수입니다.
    # 사용자 질문을 받아 최종 답변까지 생성하는 전 과정을 관리합니다.
    
    if chat_history is None:
        # 대화 기록이 제공되지 않은 경우
        chat_history = []
        # 빈 리스트로 초기화합니다.
    
    app = build_graph()
    # 위에서 정의한 build_graph 함수를 호출하여 워크플로우 그래프를 생성합니다.
    
    initial_state = {
        # 워크플로우의 초기 상태를 정의합니다.
        "question": question,
        # 사용자 질문을 설정합니다.
        "classification": "simple",
        # 초기 분류를 simple로 설정합니다 (실제로는 classify_inquiry에서 결정).
        "documents": [],
        # 빈 문서 리스트로 초기화합니다.
        "answer": "",
        # 빈 답변으로 초기화합니다.
        "chat_history": chat_history,
        # 제공된 대화 기록을 설정합니다.
        "retriever": retriever
        # 제공된 검색기를 설정합니다.
    }
    
    result = app.invoke(initial_state)
    # 컴파일된 워크플로우 앱에 초기 상태를 전달하여 실행합니다.
    
    return result
    # 워크플로우 실행 결과를 반환합니다.
```

## 주요 특징

1. **상태 기반 워크플로우**: TypedDict를 활용한 구조화된 상태 관리
2. **조건부 라우팅**: 질문 복잡도에 따른 동적 워크플로우 분기
3. **다중 모델 사용**: simple/complex 질문에 대한 차별화된 AI 모델 적용
4. **RAG 통합**: 문서 검색과 답변 생성의 완전한 통합
5. **에러 핸들링**: 각 단계별 예외 상황 대응

## 워크플로우 구조

```
시작 → 질문분류 → 문서검색 → 답변생성 → 종료
         ↓           ↓          ↙    ↘
      simple/    관련문서    RAG답변  복잡답변
      complex      검색      (GPT-3.5) (GPT-4)
```

## 핵심 알고리즘

1. **질문 분류**: GPT-3.5를 사용한 이진 분류 (simple/complex)
2. **문서 검색**: 벡터 유사도 기반 관련 문서 검색
3. **컨텍스트 통합**: 검색된 문서를 프롬프트에 포함
4. **차별화 생성**: 질문 복잡도에 따른 적절한 모델 선택
5. **상태 전파**: 각 단계별 상태 정보의 일관된 전달

## 확장성 고려사항

- 새로운 분류 카테고리 추가 가능
- 추가 AI 모델 통합 용이성
- 커스텀 라우팅 로직 구현 가능
- 다단계 검색 전략 적용 가능