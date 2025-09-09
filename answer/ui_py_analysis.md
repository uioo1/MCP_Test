# ui.py 코드 상세 해설

`ui.py`는 Streamlit 기반 사용자 인터페이스를 담당하는 모듈입니다. 사이드바, 채팅 인터페이스, 지식베이스 구축, 그리고 사용자 쿼리 처리를 위한 UI 컴포넌트들을 제공합니다.

## 코드 라인별 상세 해설

```python
import streamlit as st
# Streamlit 웹 애플리케이션 프레임워크를 import합니다.
# UI 컴포넌트 생성 및 상호작용 처리에 사용됩니다.

from langchain_openai import OpenAIEmbeddings
# LangChain의 OpenAI 임베딩 모델을 import합니다.
# 문서를 벡터로 변환하여 유사도 검색을 가능하게 합니다.

from langchain.schema import HumanMessage, AIMessage
# LangChain의 메시지 스키마를 import합니다.
# 채팅 대화에서 사용자와 AI 메시지를 구분하여 저장합니다.

import tools
# 문서 처리 및 벡터 데이터베이스 관련 기능을 제공하는 모듈입니다.
# retriever 생성 및 관리 기능을 사용합니다.

import utils
# 유틸리티 함수들을 제공하는 모듈입니다.
# API 키 검증 등의 보조 기능을 사용합니다.

def setup_sidebar():
    """
    사이드바를 설정하고 API 키 입력 및 문서 업로드 인터페이스를 생성합니다.
    
    Returns:
        tuple: (api_key, uploaded_files, build_kb_clicked)
    """
    # 사이드바의 모든 UI 요소를 설정하는 함수입니다.
    # API 키 입력, 파일 업로드, 지식베이스 구축 등의 기능을 제공합니다.
    
    with st.sidebar:
        # Streamlit 사이드바 컨텍스트를 생성합니다.
        # 이 블록 내의 모든 UI 요소는 사이드바에 배치됩니다.
        
        st.header("⚙️ 설정")
        # 설정 섹션의 헤더를 표시합니다.
        # 이모지를 사용하여 시각적으로 구분합니다.
        
        api_key = st.text_input(
            # 텍스트 입력 위젯을 생성합니다.
            "OpenAI API Key",
            # 입력 필드의 라벨을 설정합니다.
            type="password",
            # 입력 타입을 비밀번호로 설정하여 입력값을 숨깁니다.
            placeholder="sk-...",
            # 입력 필드에 표시될 힌트 텍스트입니다.
            help="OpenAI API 키를 입력하세요"
            # 도움말 텍스트를 설정합니다 (물음표 아이콘 hover시 표시).
        )
        
        if api_key:
            # API 키가 입력되었으면 유효성을 검증합니다.
            if utils.validate_api_key(api_key):
                # utils 모듈의 validate_api_key 함수로 API 키 형식을 검증합니다.
                st.success("✅ API 키가 설정되었습니다")
                # 유효한 API 키일 경우 성공 메시지를 표시합니다.
            else:
                # 유효하지 않은 API 키일 경우 에러 메시지를 표시합니다.
                st.error("❌ 유효하지 않은 API 키 형식입니다")
        
        st.divider()
        # 시각적 구분선을 추가하여 섹션을 나눕니다.
        
        st.header("📚 지식 베이스")
        # 지식베이스 섹션의 헤더를 표시합니다.
        
        uploaded_files = st.file_uploader(
            # 파일 업로드 위젯을 생성합니다.
            "문서 업로드",
            # 파일 업로더의 라벨을 설정합니다.
            type=['pdf', 'txt'],
            # 허용되는 파일 확장자를 제한합니다.
            accept_multiple_files=True,
            # 여러 파일 동시 업로드를 허용합니다.
            help="PDF 또는 TXT 파일을 업로드하세요"
            # 도움말 텍스트를 설정합니다.
        )
        
        if uploaded_files:
            # 파일이 업로드되었으면 업로드된 파일 정보를 표시합니다.
            st.info(f"📄 {len(uploaded_files)}개 파일이 업로드되었습니다")
            # 업로드된 파일 개수를 정보 메시지로 표시합니다.
            for file in uploaded_files:
                # 각 업로드된 파일에 대해 반복합니다.
                st.text(f"  • {file.name}")
                # 파일명을 텍스트로 표시합니다.
        
        build_kb_clicked = st.button(
            # 버튼 위젯을 생성합니다.
            "🔨 지식 베이스 구축",
            # 버튼의 텍스트를 설정합니다.
            disabled=not uploaded_files or not api_key,
            # 파일이 없거나 API 키가 없으면 버튼을 비활성화합니다.
            use_container_width=True
            # 버튼이 컨테이너의 전체 너비를 사용하도록 설정합니다.
        )
        
        if st.button("🗑️ 대화 기록 초기화", use_container_width=True):
            # 대화 기록 초기화 버튼을 생성합니다.
            st.session_state.chat_history = []
            # 세션 상태의 채팅 기록을 빈 배열로 초기화합니다.
            st.rerun()
            # 앱을 다시 실행하여 UI를 새로고침합니다.
        
        st.divider()
        # 또 다른 시각적 구분선을 추가합니다.
        
        st.header("🤖 모델 정보")
        # 모델 정보 섹션의 헤더를 표시합니다.
        st.info("""
        **Simple 질문**: GPT-3.5-turbo
        **Complex 질문**: GPT-4-turbo-preview
        """)
        # 사용되는 AI 모델에 대한 정보를 표시합니다.
        # 질문 복잡도에 따라 다른 모델이 사용됨을 알립니다.
        
        st.divider()
        # 마지막 구분선을 추가합니다.
        
        st.caption("💡 Tip: 문서를 업로드하고 지식 베이스를 구축하면 더 정확한 답변을 받을 수 있습니다.")
        # 사용자에게 도움이 되는 팁을 캡션으로 표시합니다.
        
        return api_key, uploaded_files, build_kb_clicked
        # 함수의 반환값: API 키, 업로드된 파일들, 지식베이스 구축 버튼 클릭 여부

def render_chat_interface():
    """
    채팅 인터페이스를 렌더링합니다.
    """
    # 메인 화면의 채팅 인터페이스를 구성하는 함수입니다.
    # 대화 기록 표시와 사용자 입력을 처리합니다.
    
    st.header("💬 AI 고객 문의 처리 시스템")
    # 메인 헤더를 표시합니다.
    
    if 'chat_history' not in st.session_state:
        # 채팅 기록이 세션 상태에 없으면 초기화합니다.
        # 이는 방어적 프로그래밍 패턴입니다.
        st.session_state.chat_history = []
        # 빈 배열로 초기화합니다.
    
    chat_container = st.container()
    # 채팅 메시지들을 담을 컨테이너를 생성합니다.
    
    with chat_container:
        # 채팅 컨테이너 컨텍스트 내에서 실행합니다.
        for message in st.session_state.chat_history:
            # 저장된 채팅 기록을 순회합니다.
            if isinstance(message, HumanMessage):
                # 메시지가 사용자 메시지인지 확인합니다.
                with st.chat_message("user"):
                    # 사용자 채팅 메시지 컨테이너를 생성합니다.
                    st.markdown(message.content)
                    # 메시지 내용을 마크다운으로 렌더링합니다.
            elif isinstance(message, AIMessage):
                # 메시지가 AI 메시지인지 확인합니다.
                with st.chat_message("assistant"):
                    # AI 어시스턴트 채팅 메시지 컨테이너를 생성합니다.
                    st.markdown(message.content)
                    # 메시지 내용을 마크다운으로 렌더링합니다.
    
    user_input = st.chat_input("질문을 입력하세요...")
    # 사용자 입력을 받는 채팅 입력 위젯을 생성합니다.
    # 플레이스홀더 텍스트를 설정합니다.
    
    return user_input
    # 사용자가 입력한 텍스트를 반환합니다.

def display_welcome_message():
    """
    환영 메시지를 표시합니다.
    """
    # 첫 방문자에게 표시될 환영 메시지를 구성하는 함수입니다.
    # 시스템 사용법과 특징을 안내합니다.
    
    st.markdown("""
    ### 👋 환영합니다!
    
    이 시스템은 AI를 활용하여 고객 문의를 자동으로 처리합니다.
    
    **사용 방법:**
    1. 왼쪽 사이드바에서 OpenAI API 키를 입력하세요
    2. (선택) 참고할 문서를 업로드하고 지식 베이스를 구축하세요
    3. 아래 입력창에 질문을 입력하세요
    
    **특징:**
    - 🤖 자동 문의 분류 (Simple/Complex)
    - 📚 RAG 기반 정확한 답변 생성
    - 🧠 모든 질문에 대한 AI 답변 제공
    
    **사용 모델:**
    - Simple 질문: GPT-3.5-turbo
    - Complex 질문: GPT-4-turbo-preview
    """)
    # 마크다운 형식으로 환영 메시지를 표시합니다.
    # 시스템 사용법, 특징, 사용 모델에 대한 정보를 제공합니다.

def display_status_message(message: str, type: str = "info"):
    """
    상태 메시지를 표시합니다.
    
    Args:
        message: 표시할 메시지
        type: 메시지 타입 (info, success, warning, error)
    """
    # 다양한 타입의 상태 메시지를 표시하는 범용 함수입니다.
    # 메시지 타입에 따라 다른 스타일로 표시됩니다.
    
    if type == "info":
        # 정보 메시지인 경우
        st.info(message)
        # 파란색 정보 박스로 메시지를 표시합니다.
    elif type == "success":
        # 성공 메시지인 경우
        st.success(message)
        # 녹색 성공 박스로 메시지를 표시합니다.
    elif type == "warning":
        # 경고 메시지인 경우
        st.warning(message)
        # 노란색 경고 박스로 메시지를 표시합니다.
    elif type == "error":
        # 에러 메시지인 경우
        st.error(message)
        # 빨간색 에러 박스로 메시지를 표시합니다.

def build_knowledge_base(uploaded_files, api_key):
    """
    지식 베이스를 구축합니다.
    
    Args:
        uploaded_files: 업로드된 파일들
        api_key: OpenAI API 키
    
    Returns:
        retriever: 생성된 검색기 또는 None
    """
    # 업로드된 문서들로부터 지식베이스를 구축하는 함수입니다.
    # RAG 시스템의 핵심 기능을 제공합니다.
    
    try:
        # 예외 처리를 위한 try-except 블록을 시작합니다.
        with st.spinner("📚 지식 베이스를 구축 중입니다..."):
            # 사용자에게 진행 상황을 알리는 스피너를 표시합니다.
            import os
            # os 모듈을 import합니다 (환경변수 설정용).
            os.environ['OPENAI_API_KEY'] = api_key
            # OpenAI API 키를 환경변수로 설정합니다.
            
            embeddings = OpenAIEmbeddings()
            # OpenAI 임베딩 모델 인스턴스를 생성합니다.
            retriever = tools.create_retriever(uploaded_files, embeddings)
            # tools 모듈의 create_retriever 함수를 호출하여 검색기를 생성합니다.
            
            st.success("✅ 지식 베이스가 성공적으로 구축되었습니다!")
            # 성공 메시지를 표시합니다.
            return retriever
            # 생성된 검색기 객체를 반환합니다.
            
    except Exception as e:
        # 예외가 발생한 경우 처리합니다.
        st.error(f"❌ 지식 베이스 구축 중 오류가 발생했습니다: {str(e)}")
        # 에러 메시지를 표시하고 예외 내용을 포함합니다.
        return None
        # None을 반환하여 실패를 나타냅니다.

def process_user_query(question: str, retriever, api_key: str):
    """
    사용자 질문을 처리하고 응답을 생성합니다.
    
    Args:
        question: 사용자 질문
        retriever: RAG 검색기
        api_key: OpenAI API 키
    
    Returns:
        str: AI 응답
    """
    # 사용자의 질문을 받아 AI 응답을 생성하는 핵심 함수입니다.
    # LangGraph 워크플로우를 실행하여 응답을 생성합니다.
    
    import os
    # os 모듈을 import합니다.
    os.environ['OPENAI_API_KEY'] = api_key
    # OpenAI API 키를 환경변수로 설정합니다.
    
    with st.spinner("🤔 답변을 생성 중입니다..."):
        # 사용자에게 AI가 답변을 생성 중임을 알리는 스피너를 표시합니다.
        import graph
        # graph 모듈을 import합니다 (LangGraph 워크플로우).
        
        result = graph.run_workflow(
            # graph 모듈의 run_workflow 함수를 호출합니다.
            question=question,
            # 사용자 질문을 전달합니다.
            retriever=retriever,
            # RAG 검색기를 전달합니다.
            chat_history=st.session_state.chat_history
            # 현재 채팅 기록을 전달합니다.
        )
        
        classification = result.get('classification', 'unknown')
        # 워크플로우 결과에서 질문 분류 결과를 가져옵니다.
        # 기본값으로 'unknown'을 사용합니다.
        
        if classification == 'complex':
            # 복잡한 질문으로 분류된 경우
            model_used = "GPT-4-turbo-preview"
            # 사용된 모델명을 설정합니다.
            st.info(f"🧠 복잡한 문의로 분류 - 사용 모델: {model_used}")
            # 분류 결과와 사용된 모델을 사용자에게 알립니다.
        else:
            # 일반 질문으로 분류된 경우
            model_used = "GPT-3.5-turbo"
            # 사용된 모델명을 설정합니다.
            st.info(f"✅ 일반 문의로 분류 - 사용 모델: {model_used}")
            # 분류 결과와 사용된 모델을 사용자에게 알립니다.
        
        return result.get('answer', '답변을 생성할 수 없습니다.')
        # 워크플로우 결과에서 답변을 가져와 반환합니다.
        # 답변이 없는 경우 기본 메시지를 반환합니다.
```

## 주요 특징

1. **모듈화된 UI**: 각 기능별로 독립적인 함수로 구성
2. **실시간 피드백**: 스피너, 상태 메시지를 통한 사용자 경험 개선
3. **조건부 UI**: 상태에 따른 동적 UI 요소 활성화/비활성화
4. **에러 처리**: 예외 상황에 대한 적절한 에러 메시지 제공
5. **상태 관리**: Streamlit 세션 상태를 활용한 데이터 유지

## UI 컴포넌트 구조

1. **사이드바**: API 키 입력, 파일 업로드, 지식베이스 구축
2. **메인 화면**: 채팅 인터페이스, 대화 기록 표시
3. **상태 표시**: 진행 상황, 성공/에러 메시지
4. **정보 제공**: 환영 메시지, 사용법 안내, 모델 정보

## 사용자 경험 최적화

- 직관적인 아이콘과 이모지 사용
- 실시간 상태 피드백
- 명확한 에러 메시지
- 도움말 텍스트 제공
- 반응형 UI 디자인