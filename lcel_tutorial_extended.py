"""
LangChain Expression Language (LCEL) 확장 튜토리얼 - LangGraph 활용
==============================================================

실제 LangGraph StateGraph를 사용하여 LCEL 패턴들을 구현하고 시각화합니다.
단순한 chain_steps 리스트가 아닌, 실제 노드와 엣지가 연결된 그래프를 생성합니다.

필요한 라이브러리:
pip install langchain langchain-openai langchain-core langchain-community langgraph
"""

from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain_core.runnables import RunnableBranch, RunnableParallel
from langchain_core.output_parsers import JsonOutputParser
from operator import itemgetter
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, List, Any, TypedDict, Annotated
import operator
import inspect

# LangGraph 임포트
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
import matplotlib.font_manager as fm
import os

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ==================================================
# 상태 정의 클래스들
# ==================================================

class BasicState(TypedDict):
    """기본 상태 클래스"""
    input: str
    output: str
    step: str

class ParallelState(TypedDict):
    """병렬 처리 상태 클래스"""
    input: str
    branch_a: str
    branch_b: str 
    branch_c: str
    combined_output: Dict[str, str]

class ConditionalState(TypedDict):
    """조건부 처리 상태 클래스"""
    input: str
    condition: bool
    processed_output: str

class ComplexState(TypedDict):
    """복합 처리 상태 클래스"""
    input: Dict[str, Any]
    # 병렬 처리에서 안전한 상태 키들 - 리듀서 사용
    analysis_results: Annotated[List[Dict[str, Any]], operator.add]
    final_output: Dict[str, Any]

# ==================================================
# LangGraph 시각화 유틸리티 함수
# ==================================================

def visualize_langgraph(graph: StateGraph, title: str = "LangGraph Structure", save_path: str = None):
    """LangGraph StateGraph를 PNG 이미지로 저장"""
    try:
        # figure 디렉토리 생성
        figure_dir = "figure"
        if not os.path.exists(figure_dir):
            os.makedirs(figure_dir)
        
        # 파일명 생성 (한글 제목을 영문으로 변환)
        if save_path is None:
            # 제목에서 파일명 생성
            filename = title.replace(" ", "_").replace("===", "").strip()
            # 한글을 영문으로 간단히 변환
            filename = filename.replace("기본", "basic") \
                              .replace("체인", "chain") \
                              .replace("구조", "structure") \
                              .replace("병렬", "parallel") \
                              .replace("처리", "processing") \
                              .replace("조건부", "conditional") \
                              .replace("복합", "complex") \
                              .replace("Lambda", "lambda") \
                              .replace("Passthrough", "passthrough") \
                              .replace("Branch", "branch") \
                              .replace("디버깅", "debug")
            save_path = os.path.join(figure_dir, f"{filename}.png")
        
        # LangGraph 컴파일
        compiled_graph = graph.compile()
        
        # PNG 이미지로 저장 시도
        try:
            png_data = compiled_graph.get_graph().draw_mermaid_png()
            with open(save_path, "wb") as f:
                f.write(png_data)
            print(f"=== {title} ===")
            print(f"그래프 이미지 저장됨: {save_path}")
            print()
        except Exception as png_error:
            # PNG 생성 실패 시 Mermaid 텍스트만 출력
            try:
                mermaid_diagram = compiled_graph.get_graph().draw_mermaid()
                print(f"=== {title} ===")
                print("Mermaid 다이어그램:")
                print(mermaid_diagram)
                print(f"PNG 저장 실패: {png_error}")
                print()
            except:
                print(f"=== {title} ===")
                print("그래프 구조:")
                print(f"노드: {list(compiled_graph.nodes)}")
                print()
        
    except Exception as e:
        print(f"LangGraph 시각화 오류: {e}")
        # 대안: 간단한 텍스트 기반 시각화
        print(f"=== {title} ===")
        print("그래프 구조 (텍스트):")
        try:
            if hasattr(graph, 'nodes'):
                print(f"노드: {list(graph.nodes.keys())}")
            if hasattr(graph, 'edges'):
                print(f"엣지: {list(graph.edges)}")
        except:
            print("그래프 정보를 가져올 수 없습니다.")
        print()

# ==================================================
# 1. 기본 LangGraph 체인 구성
# ==================================================

def basic_langgraph_chain():
    """기본적인 LangGraph 체인 구성법"""
    print("=== 1. 기본 LangGraph 체인 ===")
    
    # 노드 함수들
    def process_input(state: BasicState) -> BasicState:
        """입력 처리 노드"""
        state["step"] = "processed_input"
        return state
    
    def generate_output(state: BasicState) -> BasicState:
        """출력 생성 노드"""
        state["output"] = f"처리 결과: {state['input']}"
        return state
    
    # StateGraph 생성
    workflow = StateGraph(BasicState)
    
    # 노드 추가
    workflow.add_node("process_input", process_input)
    workflow.add_node("generate_output", generate_output)
    
    # 엣지 연결
    workflow.add_edge(START, "process_input")
    workflow.add_edge("process_input", "generate_output")
    workflow.add_edge("generate_output", END)
    
    # 그래프 컴파일
    app = workflow.compile()
    
    # 실행 예시
    result = app.invoke({"input": "LCEL 테스트", "output": "", "step": ""})
    print("입력:", {"input": "LCEL 테스트"})
    print("출력:", result)
    
    # 실제 LangGraph 구조 시각화
    visualize_langgraph(workflow, "기본 LangGraph 체인 구조")
    print()
    
    return app

# ==================================================
# 2. RunnableLambda를 LangGraph로 구현
# ==================================================

def lambda_langgraph_chain():
    """RunnableLambda를 LangGraph로 구현"""
    print("=== 2. Lambda LangGraph 체인 ===")
    
    # 노드 함수들
    def preprocess_node(state: BasicState) -> BasicState:
        """전처리 노드"""
        state["input"] = state["input"].upper()
        state["step"] = "preprocessed"
        return state
    
    def format_node(state: BasicState) -> BasicState:
        """포맷팅 노드"""
        state["input"] = f"처리된 텍스트: {state['input']}"
        state["step"] = "formatted"
        return state
    
    def postprocess_node(state: BasicState) -> BasicState:
        """후처리 노드"""
        state["output"] = f"결과: {state['input']}"
        return state
    
    # StateGraph 생성
    workflow = StateGraph(BasicState)
    
    # 노드 추가
    workflow.add_node("preprocess", preprocess_node)
    workflow.add_node("format", format_node)
    workflow.add_node("postprocess", postprocess_node)
    
    # 엣지 연결
    workflow.add_edge(START, "preprocess")
    workflow.add_edge("preprocess", "format")
    workflow.add_edge("format", "postprocess")
    workflow.add_edge("postprocess", END)
    
    # 그래프 컴파일 및 실행
    app = workflow.compile()
    result = app.invoke({"input": "hello world", "output": "", "step": ""})
    
    print("입력: 'hello world'")
    print("출력:", result)
    
    # LangGraph 구조 시각화
    visualize_langgraph(workflow, "Lambda LangGraph 체인 구조")
    print()
    
    return app

# ==================================================
# 3. RunnablePassthrough를 LangGraph로 구현
# ==================================================

def passthrough_langgraph_chain():
    """RunnablePassthrough를 LangGraph로 구현"""
    print("=== 3. Passthrough LangGraph 체인 ===")
    
    def passthrough_node(state: BasicState) -> BasicState:
        """데이터 통과 노드"""
        state["step"] = "passed_through"
        return state
    
    def add_metadata_node(state: BasicState) -> BasicState:
        """메타데이터 추가 노드"""
        # JSON 형태의 문자열로 처리
        import json
        try:
            data = json.loads(state["input"]) if isinstance(state["input"], str) else {"original": state["input"]}
        except:
            data = {"original": state["input"]}
        
        data["processed"] = True
        state["output"] = json.dumps(data, ensure_ascii=False)
        return state
    
    # StateGraph 생성
    workflow = StateGraph(BasicState)
    
    workflow.add_node("passthrough", passthrough_node)
    workflow.add_node("add_metadata", add_metadata_node)
    
    workflow.add_edge(START, "passthrough")
    workflow.add_edge("passthrough", "add_metadata")
    workflow.add_edge("add_metadata", END)
    
    # 실행
    app = workflow.compile()
    test_input = '{"name": "LCEL", "type": "tutorial"}'
    result = app.invoke({"input": test_input, "output": "", "step": ""})
    
    print("입력:", test_input)
    print("출력:", result["output"])
    
    visualize_langgraph(workflow, "Passthrough LangGraph 체인 구조")
    print()
    
    return app

# ==================================================
# 4. RunnableParallel을 LangGraph로 구현
# ==================================================

def parallel_langgraph_chain():
    """RunnableParallel 개념을 LangGraph로 구현 (실제로는 순차 실행)"""
    print("=== 4. Parallel-style LangGraph 체인 ===")
    
    # 각 처리 노드들 - 실제로는 순차 실행하지만 병렬 개념을 시뮬레이션
    def process_all_branches(state: ParallelState) -> ParallelState:
        """모든 브랜치 처리 (병렬 시뮬레이션)"""
        branch_a = f"A 처리: {state['input']}"
        branch_b = f"B 처리: {state['input']}"
        branch_c = f"C 처리: {state['input']}"
        
        combined_output = {
            "branch_a": branch_a,
            "branch_b": branch_b,
            "branch_c": branch_c
        }
        
        return {
            "branch_a": branch_a,
            "branch_b": branch_b,
            "branch_c": branch_c,
            "combined_output": combined_output
        }
    
    # StateGraph 생성 (단순화)
    workflow = StateGraph(ParallelState)
    workflow.add_node("process_all", process_all_branches)
    workflow.add_edge(START, "process_all")
    workflow.add_edge("process_all", END)
    
    # 실행
    app = workflow.compile()
    result = app.invoke({
        "input": "테스트 데이터",
        "branch_a": "",
        "branch_b": "",
        "branch_c": "",
        "combined_output": {}
    })
    
    print("입력: '테스트 데이터'")
    print("출력:", result["combined_output"])
    
    visualize_langgraph(workflow, "Parallel LangGraph 처리 구조")
    print()
    
    return app

# ==================================================
# 5. RunnableBranch를 LangGraph로 구현
# ==================================================

def branch_langgraph_chain():
    """RunnableBranch를 LangGraph로 구현"""
    print("=== 5. Branch LangGraph 체인 ===")
    
    def condition_check_node(state: ConditionalState) -> ConditionalState:
        """조건 확인 노드"""
        try:
            float(state["input"])
            state["condition"] = True
        except:
            state["condition"] = False
        return state
    
    def process_number_node(state: ConditionalState) -> ConditionalState:
        """숫자 처리 노드"""
        state["processed_output"] = f"숫자 처리: {float(state['input']) * 2}"
        return state
    
    def process_text_node(state: ConditionalState) -> ConditionalState:
        """텍스트 처리 노드"""
        state["processed_output"] = f"텍스트 처리: {state['input'].upper()}"
        return state
    
    # 조건부 라우팅 함수
    def route_condition(state: ConditionalState) -> str:
        """조건에 따라 라우팅"""
        return "process_number" if state["condition"] else "process_text"
    
    # StateGraph 생성
    workflow = StateGraph(ConditionalState)
    
    workflow.add_node("condition_check", condition_check_node)
    workflow.add_node("process_number", process_number_node)
    workflow.add_node("process_text", process_text_node)
    
    workflow.add_edge(START, "condition_check")
    workflow.add_conditional_edges(
        "condition_check",
        route_condition,
        {
            "process_number": "process_number",
            "process_text": "process_text"
        }
    )
    workflow.add_edge("process_number", END)
    workflow.add_edge("process_text", END)
    
    # 실행
    app = workflow.compile()
    
    result1 = app.invoke({"input": "123", "condition": False, "processed_output": ""})
    result2 = app.invoke({"input": "hello", "condition": False, "processed_output": ""})
    
    print("입력1: '123' ->", result1["processed_output"])
    print("입력2: 'hello' ->", result2["processed_output"])
    
    visualize_langgraph(workflow, "Branch LangGraph 조건부 처리")
    print()
    
    return app

# ==================================================
# 6. 복합 LangGraph 체인 구성
# ==================================================

def complex_langgraph_chain():
    """여러 LangGraph 기법을 조합한 복합 체인"""
    print("=== 6. 복합 LangGraph 체인 구성 ===")
    
    # 통합된 분석 노드 - 병렬 처리의 복잡성 없이
    def analyze_text_node(state: ComplexState) -> ComplexState:
        """텍스트 종합 분석 노드"""
        text = state["input"]["text"].strip().lower()
        
        # 길이 분석
        length_analysis = {"length": len(text)}
        
        # 단어 분석  
        word_analysis = {"word_count": len(text.split())}
        
        # 문자 분석
        char_analysis = {"char_types": {
            "alpha": sum(c.isalpha() for c in text),
            "digit": sum(c.isdigit() for c in text),
            "space": sum(c.isspace() for c in text)
        }}
        
        # 모든 분석 결과를 하나의 결과로 통합
        final_output = {"text_analysis": {}}
        final_output["text_analysis"].update(length_analysis)
        final_output["text_analysis"].update(word_analysis) 
        final_output["text_analysis"].update(char_analysis)
        
        return {
            "analysis_results": [final_output],
            "final_output": final_output
        }
    
    # StateGraph 생성 (단순화)
    workflow = StateGraph(ComplexState)
    workflow.add_node("analyze_text", analyze_text_node)
    workflow.add_edge(START, "analyze_text")
    workflow.add_edge("analyze_text", END)
    
    # 실행
    app = workflow.compile()
    
    test_data = {"text": "  Hello World 123!  ", "metadata": {"source": "tutorial"}}
    result = app.invoke({
        "input": test_data,
        "analysis_results": [],
        "final_output": {}
    })
    
    print("입력:", test_data)
    print("출력:", result["final_output"])
    
    visualize_langgraph(workflow, "복합 LangGraph 체인 구조")
    print()
    
    return app

# ==================================================
# 7. 상태 추적 및 디버깅
# ==================================================

def debug_langgraph_execution():
    """LangGraph 실행 과정 추적 및 디버깅"""
    print("=== 7. LangGraph 디버깅 및 상태 추적 ===")
    
    def debug_node(state: BasicState) -> BasicState:
        """디버깅 노드"""
        print(f"현재 상태: {state}")
        state["step"] = f"debug_step_{len(state.get('step', ''))}"
        return state
    
    def processing_node(state: BasicState) -> BasicState:
        """처리 노드"""
        state["output"] = f"처리됨: {state['input']}"
        return state
    
    workflow = StateGraph(BasicState)
    
    workflow.add_node("debug", debug_node)
    workflow.add_node("process", processing_node)
    
    workflow.add_edge(START, "debug")
    workflow.add_edge("debug", "process")
    workflow.add_edge("process", END)
    
    app = workflow.compile()
    
    print("실행 과정 추적:")
    result = app.invoke({"input": "디버그 테스트", "output": "", "step": ""})
    print("최종 결과:", result)
    
    visualize_langgraph(workflow, "디버깅 LangGraph 구조")
    print()
    
    return app

# ==================================================
# 실행 함수
# ==================================================

def run_all_langgraph_examples():
    """모든 LangGraph 예제 실행"""
    print("LangChain LCEL을 LangGraph로 확장한 튜토리얼")
    print("=" * 60)
    print()
    
    basic_langgraph_chain()
    lambda_langgraph_chain()
    passthrough_langgraph_chain()
    parallel_langgraph_chain()
    branch_langgraph_chain()
    complex_langgraph_chain()
    debug_langgraph_execution()
    
    print("=" * 60)
    print("LangGraph 확장 튜토리얼 완료!")
    print("\nLangGraph vs 기존 LCEL의 차이점:")
    print("1. 실제 노드와 엣지가 정의된 상태 기반 그래프")
    print("2. 상태(State)를 통한 데이터 전달 및 관리")
    print("3. 조건부 엣지를 통한 동적 라우팅")
    print("4. 병렬 실행 및 동기화 제어")
    print("5. 실행 과정 추적 및 디버깅 용이성")
    print("6. 더 복잡한 워크플로우 구성 가능")

if __name__ == "__main__":
    run_all_langgraph_examples()