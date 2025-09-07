from typing import TypedDict, Literal, List, Optional
from langchain.schema import BaseMessage, HumanMessage, AIMessage, Document
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
import tools
import streamlit as st


class GraphState(TypedDict):
    """그래프의 상태를 정의하는 TypedDict"""
    question: str
    classification: Literal["simple", "complex"]
    documents: List[Document]
    answer: str
    chat_history: List[BaseMessage]
    retriever: Optional[object]


def classify_inquiry(state: GraphState) -> GraphState:
    """
    사용자 문의를 simple 또는 complex로 분류합니다.
    
    Args:
        state: 현재 그래프 상태
    
    Returns:
        GraphState: 업데이트된 상태
    """
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    
    classification_prompt = f"""
    다음 고객 문의를 분석하여 'simple' 또는 'complex'로 분류하세요.
    
    분류 기준:
    - simple: 일반적인 질문, FAQ, 간단한 정보 요청, 지식 베이스로 답변 가능한 질문
    - complex: 기술적 문제, 복잡한 요청, 인간 개입이 필요한 문제, 민감한 이슈
    
    문의: {state['question']}
    
    응답은 반드시 'simple' 또는 'complex' 중 하나만 출력하세요.
    """
    
    response = llm.predict(classification_prompt)
    classification = "simple" if "simple" in response.lower() else "complex"
    
    state['classification'] = classification
    return state


def retrieve_documents(state: GraphState) -> GraphState:
    """
    RAG를 사용하여 관련 문서를 검색합니다.
    
    Args:
        state: 현재 그래프 상태
    
    Returns:
        GraphState: 업데이트된 상태
    """
    retriever = state.get('retriever')
    
    if retriever is None:
        state['documents'] = []
        return state
    
    try:
        documents = tools.search_documents(retriever, state['question'])
        state['documents'] = documents
    except Exception as e:
        print(f"문서 검색 중 오류: {e}")
        state['documents'] = []
    
    return state


def generate_rag_answer(state: GraphState) -> GraphState:
    """
    검색된 문서를 바탕으로 답변을 생성합니다.
    
    Args:
        state: 현재 그래프 상태
    
    Returns:
        GraphState: 업데이트된 상태
    """
    llm = ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo")
    
    if state['documents']:
        context = "\n\n".join([doc.page_content for doc in state['documents'][:3]])
        
        prompt = f"""
        다음 컨텍스트를 참고하여 고객 문의에 답변해주세요.
        답변은 친절하고 명확하게 작성해주세요.
        
        컨텍스트:
        {context}
        
        고객 문의: {state['question']}
        
        답변:
        """
    else:
        prompt = f"""
        고객 문의에 대해 일반적인 지식을 바탕으로 답변해주세요.
        답변은 친절하고 명확하게 작성해주세요.
        
        고객 문의: {state['question']}
        
        답변:
        """
    
    answer = llm.predict(prompt)
    state['answer'] = answer
    
    return state


def generate_complex_answer(state: GraphState) -> GraphState:
    """
    복잡한 문의에 대해서도 AI가 답변을 생성합니다.
    
    Args:
        state: 현재 그래프 상태
    
    Returns:
        GraphState: 업데이트된 상태
    """
    llm = ChatOpenAI(temperature=0.7, model="gpt-4-turbo-preview")
    
    if state['documents']:
        context = "\n\n".join([doc.page_content for doc in state['documents'][:3]])
        
        prompt = f"""
        다음은 복잡한 기술적 문의입니다. 전문적이고 상세한 답변을 제공해주세요.
        가능한 컨텍스트를 참고하되, 없어도 최선을 다해 답변해주세요.
        
        컨텍스트:
        {context}
        
        고객 문의: {state['question']}
        
        답변:
        """
    else:
        prompt = f"""
        다음은 복잡한 기술적 문의입니다. 전문적이고 상세한 답변을 제공해주세요.
        
        고객 문의: {state['question']}
        
        답변:
        """
    
    answer = llm.predict(prompt)
    state['answer'] = answer
    
    return state


def route_after_classification(state: GraphState) -> str:
    """
    분류 결과에 따라 다음 노드를 결정합니다.
    모든 경우에 문서를 검색하고 답변을 생성합니다.
    
    Args:
        state: 현재 그래프 상태
    
    Returns:
        str: 다음 노드 이름
    """
    return 'retrieve_documents'


def build_graph():
    """
    LangGraph 워크플로우를 구축합니다.
    
    Returns:
        CompiledGraph: 컴파일된 그래프 객체
    """
    workflow = StateGraph(GraphState)
    
    workflow.add_node("classify_inquiry", classify_inquiry)
    workflow.add_node("retrieve_documents", retrieve_documents)
    workflow.add_node("generate_rag_answer", generate_rag_answer)
    workflow.add_node("generate_complex_answer", generate_complex_answer)
    
    workflow.set_entry_point("classify_inquiry")
    
    workflow.add_conditional_edges(
        "classify_inquiry",
        route_after_classification,
        {
            "retrieve_documents": "retrieve_documents"
        }
    )
    
    def route_after_retrieve(state: GraphState) -> str:
        if state['classification'] == 'simple':
            return 'generate_rag_answer'
        else:
            return 'generate_complex_answer'
    
    workflow.add_conditional_edges(
        "retrieve_documents",
        route_after_retrieve,
        {
            "generate_rag_answer": "generate_rag_answer",
            "generate_complex_answer": "generate_complex_answer"
        }
    )
    
    workflow.add_edge("generate_rag_answer", END)
    workflow.add_edge("generate_complex_answer", END)
    
    return workflow.compile()


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
    if chat_history is None:
        chat_history = []
    
    app = build_graph()
    
    initial_state = {
        "question": question,
        "classification": "simple",
        "documents": [],
        "answer": "",
        "chat_history": chat_history,
        "retriever": retriever
    }
    
    result = app.invoke(initial_state)
    
    return result