import streamlit as st
import os
from dotenv import load_dotenv
from langchain.schema import HumanMessage, AIMessage
import ui
import tools
import graph

load_dotenv()

def initialize_session_state():
    """
    세션 상태를 초기화합니다.
    """
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'retriever' not in st.session_state:
        st.session_state.retriever = None
    
    if 'api_key' not in st.session_state:
        st.session_state.api_key = os.getenv('OPENAI_API_KEY', '')
    
    if 'kb_built' not in st.session_state:
        st.session_state.kb_built = False


def main():
    """
    메인 애플리케이션 실행 함수
    """
    st.set_page_config(
        page_title="AI 고객 문의 처리 시스템",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    initialize_session_state()
    
    api_key, uploaded_files, build_kb_clicked = ui.setup_sidebar()
    
    if api_key:
        st.session_state.api_key = api_key
    
    if build_kb_clicked and uploaded_files and api_key:
        retriever = ui.build_knowledge_base(uploaded_files, api_key)
        if retriever:
            st.session_state.retriever = retriever
            st.session_state.kb_built = True
    
    if not api_key:
        existing_retriever = tools.load_existing_retriever()
        if existing_retriever:
            st.session_state.retriever = existing_retriever
            st.info("📚 기존 지식 베이스를 로드했습니다.")
    
    if len(st.session_state.chat_history) == 0:
        ui.display_welcome_message()
    
    user_input = ui.render_chat_interface()
    
    if user_input and st.session_state.api_key:
        st.session_state.chat_history.append(HumanMessage(content=user_input))
        
        with st.chat_message("user"):
            st.markdown(user_input)
        
        response = ui.process_user_query(
            question=user_input,
            retriever=st.session_state.retriever,
            api_key=st.session_state.api_key
        )
        
        st.session_state.chat_history.append(AIMessage(content=response))
        
        with st.chat_message("assistant"):
            st.markdown(response)
    
    elif user_input and not st.session_state.api_key:
        st.error("⚠️ OpenAI API 키를 먼저 입력해주세요.")
    
    with st.sidebar:
        st.divider()
        st.caption("📊 상태 정보")
        st.text(f"대화 수: {len(st.session_state.chat_history)//2}")
        st.text(f"지식 베이스: {'✅ 활성' if st.session_state.retriever else '❌ 비활성'}")
        st.text(f"API 키: {'✅ 설정됨' if st.session_state.api_key else '❌ 미설정'}")


if __name__ == "__main__":
    main()