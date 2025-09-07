import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain.schema import HumanMessage, AIMessage
import tools
import utils


def setup_sidebar():
    """
    사이드바를 설정하고 API 키 입력 및 문서 업로드 인터페이스를 생성합니다.
    
    Returns:
        tuple: (api_key, uploaded_files, build_kb_clicked)
    """
    with st.sidebar:
        st.header("⚙️ 설정")
        
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="sk-...",
            help="OpenAI API 키를 입력하세요"
        )
        
        if api_key:
            if utils.validate_api_key(api_key):
                st.success("✅ API 키가 설정되었습니다")
            else:
                st.error("❌ 유효하지 않은 API 키 형식입니다")
        
        st.divider()
        
        st.header("📚 지식 베이스")
        
        uploaded_files = st.file_uploader(
            "문서 업로드",
            type=['pdf', 'txt'],
            accept_multiple_files=True,
            help="PDF 또는 TXT 파일을 업로드하세요"
        )
        
        if uploaded_files:
            st.info(f"📄 {len(uploaded_files)}개 파일이 업로드되었습니다")
            for file in uploaded_files:
                st.text(f"  • {file.name}")
        
        build_kb_clicked = st.button(
            "🔨 지식 베이스 구축",
            disabled=not uploaded_files or not api_key,
            use_container_width=True
        )
        
        if st.button("🗑️ 대화 기록 초기화", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
        
        st.divider()
        
        st.header("🤖 모델 정보")
        st.info("""
        **Simple 질문**: GPT-3.5-turbo
        **Complex 질문**: GPT-4-turbo-preview
        """)
        
        st.divider()
        
        st.caption("💡 Tip: 문서를 업로드하고 지식 베이스를 구축하면 더 정확한 답변을 받을 수 있습니다.")
        
        return api_key, uploaded_files, build_kb_clicked


def render_chat_interface():
    """
    채팅 인터페이스를 렌더링합니다.
    """
    st.header("💬 AI 고객 문의 처리 시스템")
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.chat_history:
            if isinstance(message, HumanMessage):
                with st.chat_message("user"):
                    st.markdown(message.content)
            elif isinstance(message, AIMessage):
                with st.chat_message("assistant"):
                    st.markdown(message.content)
    
    user_input = st.chat_input("질문을 입력하세요...")
    
    return user_input


def display_welcome_message():
    """
    환영 메시지를 표시합니다.
    """
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


def display_status_message(message: str, type: str = "info"):
    """
    상태 메시지를 표시합니다.
    
    Args:
        message: 표시할 메시지
        type: 메시지 타입 (info, success, warning, error)
    """
    if type == "info":
        st.info(message)
    elif type == "success":
        st.success(message)
    elif type == "warning":
        st.warning(message)
    elif type == "error":
        st.error(message)


def build_knowledge_base(uploaded_files, api_key):
    """
    지식 베이스를 구축합니다.
    
    Args:
        uploaded_files: 업로드된 파일들
        api_key: OpenAI API 키
    
    Returns:
        retriever: 생성된 검색기 또는 None
    """
    try:
        with st.spinner("📚 지식 베이스를 구축 중입니다..."):
            import os
            os.environ['OPENAI_API_KEY'] = api_key
            
            embeddings = OpenAIEmbeddings()
            retriever = tools.create_retriever(uploaded_files, embeddings)
            
            st.success("✅ 지식 베이스가 성공적으로 구축되었습니다!")
            return retriever
            
    except Exception as e:
        st.error(f"❌ 지식 베이스 구축 중 오류가 발생했습니다: {str(e)}")
        return None


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
    import os
    os.environ['OPENAI_API_KEY'] = api_key
    
    with st.spinner("🤔 답변을 생성 중입니다..."):
        import graph
        
        result = graph.run_workflow(
            question=question,
            retriever=retriever,
            chat_history=st.session_state.chat_history
        )
        
        classification = result.get('classification', 'unknown')
        
        if classification == 'complex':
            model_used = "GPT-4-turbo-preview"
            st.info(f"🧠 복잡한 문의로 분류 - 사용 모델: {model_used}")
        else:
            model_used = "GPT-3.5-turbo"
            st.info(f"✅ 일반 문의로 분류 - 사용 모델: {model_used}")
        
        return result.get('answer', '답변을 생성할 수 없습니다.')