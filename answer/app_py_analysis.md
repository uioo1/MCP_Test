# app.py 코드 상세 해설

`app.py`는 Streamlit 기반 AI 고객 문의 처리 시스템의 메인 애플리케이션 파일입니다. 전체 애플리케이션의 진입점 역할을 하며, 사용자 인터페이스와 비즈니스 로직을 연결합니다.

## 코드 라인별 상세 해설

```python
import streamlit as st
# Streamlit 웹 애플리케이션 프레임워크를 import합니다. 
# st 별칭으로 사용하여 UI 컴포넌트를 생성할 수 있습니다.

import os
# 운영체제와 상호작용하기 위한 모듈입니다.
# 환경 변수 접근 등에 사용됩니다.

from dotenv import load_dotenv
# .env 파일에서 환경 변수를 로드하기 위한 모듈입니다.
# API 키 등의 민감한 정보를 안전하게 관리할 수 있습니다.

from langchain.schema import HumanMessage, AIMessage
# LangChain의 메시지 스키마를 import합니다.
# 채팅 대화 기록을 구조화된 형태로 저장하기 위해 사용됩니다.

import ui
# 사용자 인터페이스 관련 함수들이 정의된 모듈입니다.
# UI 컴포넌트 렌더링과 상호작용을 처리합니다.

import tools
# 문서 처리, 벡터 데이터베이스 관련 도구들이 정의된 모듈입니다.
# RAG 시스템의 핵심 기능을 제공합니다.

import graph
# LangGraph를 활용한 워크플로우가 정의된 모듈입니다.
# AI 응답 생성 파이프라인을 관리합니다.

load_dotenv()
# .env 파일에서 환경 변수들을 시스템 환경변수로 로드합니다.
# 이를 통해 API 키 등을 안전하게 관리할 수 있습니다.

def initialize_session_state():
    """
    세션 상태를 초기화합니다.
    """
    # Streamlit 세션 상태를 초기화하는 함수입니다.
    # 웹 애플리케이션의 상태를 브라우저 세션 동안 유지하기 위해 사용됩니다.
    
    if 'chat_history' not in st.session_state:
        # 'chat_history' 키가 세션 상태에 없으면 실행됩니다.
        # 사용자와 AI의 대화 기록을 저장할 빈 리스트를 초기화합니다.
        st.session_state.chat_history = []
        # 채팅 기록을 빈 배열로 초기화합니다.
    
    if 'retriever' not in st.session_state:
        # 'retriever' 키가 세션 상태에 없으면 실행됩니다.
        # RAG 시스템에서 문서 검색을 담당하는 retriever 객체를 저장합니다.
        st.session_state.retriever = None
        # retriever를 None으로 초기화합니다 (아직 지식베이스가 구축되지 않음).
    
    if 'api_key' not in st.session_state:
        # 'api_key' 키가 세션 상태에 없으면 실행됩니다.
        # OpenAI API 키를 저장하기 위한 변수입니다.
        st.session_state.api_key = os.getenv('OPENAI_API_KEY', '')
        # 환경 변수에서 OpenAI API 키를 가져오거나, 없으면 빈 문자열로 설정합니다.
    
    if 'kb_built' not in st.session_state:
        # 'kb_built' 키가 세션 상태에 없으면 실행됩니다.
        # 지식 베이스 구축 완료 여부를 추적하는 불린 변수입니다.
        st.session_state.kb_built = False
        # 초기값으로 False를 설정합니다 (아직 지식베이스가 구축되지 않음).

def main():
    """
    메인 애플리케이션 실행 함수
    """
    # 애플리케이션의 메인 로직을 담당하는 함수입니다.
    # 전체 UI를 설정하고 사용자 상호작용을 처리합니다.
    
    st.set_page_config(
        # Streamlit 페이지의 기본 설정을 구성합니다.
        page_title="AI 고객 문의 처리 시스템",
        # 브라우저 탭에 표시될 페이지 제목을 설정합니다.
        page_icon="🤖",
        # 브라우저 탭에 표시될 파비콘 이모지를 설정합니다.
        layout="wide",
        # 페이지 레이아웃을 넓게(wide) 설정하여 화면을 효율적으로 사용합니다.
        initial_sidebar_state="expanded"
        # 사이드바를 초기에 확장된 상태로 표시합니다.
    )
    
    initialize_session_state()
    # 위에서 정의한 세션 상태 초기화 함수를 호출합니다.
    
    api_key, uploaded_files, build_kb_clicked = ui.setup_sidebar()
    # ui 모듈의 setup_sidebar 함수를 호출하여 사이드바를 설정합니다.
    # 반환값: API 키, 업로드된 파일들, 지식베이스 구축 버튼 클릭 여부
    
    if api_key:
        # API 키가 입력되었으면 실행됩니다.
        st.session_state.api_key = api_key
        # 세션 상태에 API 키를 저장합니다.
    
    if build_kb_clicked and uploaded_files and api_key:
        # 지식베이스 구축 버튼이 클릭되고, 파일이 업로드되고, API 키가 있으면 실행됩니다.
        # 모든 조건이 만족될 때만 지식베이스를 구축합니다.
        retriever = ui.build_knowledge_base(uploaded_files, api_key)
        # ui 모듈의 build_knowledge_base 함수를 호출하여 지식베이스를 구축합니다.
        if retriever:
            # retriever 객체가 성공적으로 생성되었으면 실행됩니다.
            st.session_state.retriever = retriever
            # 세션 상태에 retriever를 저장합니다.
            st.session_state.kb_built = True
            # 지식베이스 구축 완료 상태를 True로 설정합니다.
    
    if not api_key:
        # API 키가 없으면 실행됩니다.
        # 기존에 저장된 지식베이스가 있는지 확인하여 로드합니다.
        existing_retriever = tools.load_existing_retriever()
        # tools 모듈의 load_existing_retriever 함수를 호출하여 기존 retriever를 로드합니다.
        if existing_retriever:
            # 기존 retriever가 존재하면 실행됩니다.
            st.session_state.retriever = existing_retriever
            # 세션 상태에 기존 retriever를 저장합니다.
            st.info("📚 기존 지식 베이스를 로드했습니다.")
            # 사용자에게 기존 지식베이스 로드 완료 메시지를 표시합니다.
    
    if len(st.session_state.chat_history) == 0:
        # 채팅 기록이 비어있으면 (첫 방문) 실행됩니다.
        ui.display_welcome_message()
        # ui 모듈의 display_welcome_message 함수를 호출하여 환영 메시지를 표시합니다.
    
    user_input = ui.render_chat_interface()
    # ui 모듈의 render_chat_interface 함수를 호출하여 채팅 인터페이스를 렌더링합니다.
    # 사용자가 입력한 메시지를 반환받습니다.
    
    if user_input and st.session_state.api_key:
        # 사용자 입력이 있고 API 키가 설정되어 있으면 실행됩니다.
        # AI 응답 생성을 위한 모든 조건이 충족된 상태입니다.
        st.session_state.chat_history.append(HumanMessage(content=user_input))
        # 사용자 메시지를 HumanMessage 객체로 래핑하여 채팅 기록에 추가합니다.
        
        with st.chat_message("user"):
            # Streamlit의 채팅 메시지 컨테이너를 사용자 모드로 생성합니다.
            st.markdown(user_input)
            # 사용자 입력을 마크다운 형식으로 표시합니다.
        
        response = ui.process_user_query(
            # ui 모듈의 process_user_query 함수를 호출하여 사용자 질문을 처리합니다.
            question=user_input,
            # 사용자가 입력한 질문을 전달합니다.
            retriever=st.session_state.retriever,
            # 세션에 저장된 retriever 객체를 전달합니다.
            api_key=st.session_state.api_key
            # 세션에 저장된 API 키를 전달합니다.
        )
        
        st.session_state.chat_history.append(AIMessage(content=response))
        # AI 응답을 AIMessage 객체로 래핑하여 채팅 기록에 추가합니다.
        
        with st.chat_message("assistant"):
            # Streamlit의 채팅 메시지 컨테이너를 어시스턴트 모드로 생성합니다.
            st.markdown(response)
            # AI 응답을 마크다운 형식으로 표시합니다.
    
    elif user_input and not st.session_state.api_key:
        # 사용자 입력은 있지만 API 키가 없으면 실행됩니다.
        st.error("⚠️ OpenAI API 키를 먼저 입력해주세요.")
        # 사용자에게 API 키 입력이 필요하다는 에러 메시지를 표시합니다.
    
    with st.sidebar:
        # 사이드바 영역에 추가 정보를 표시합니다.
        st.divider()
        # 시각적 구분선을 추가합니다.
        st.caption("📊 상태 정보")
        # 상태 정보 섹션의 제목을 표시합니다.
        st.text(f"대화 수: {len(st.session_state.chat_history)//2}")
        # 총 대화 수를 표시합니다 (사용자+AI 메시지 쌍이므로 2로 나눕니다).
        st.text(f"지식 베이스: {'✅ 활성' if st.session_state.retriever else '❌ 비활성'}")
        # retriever 존재 여부에 따라 지식베이스 활성화 상태를 표시합니다.
        st.text(f"API 키: {'✅ 설정됨' if st.session_state.api_key else '❌ 미설정'}")
        # API 키 설정 여부를 표시합니다.

if __name__ == "__main__":
    # 스크립트가 직접 실행될 때만 실행되는 블록입니다.
    # 모듈로 import될 때는 실행되지 않습니다.
    main()
    # main 함수를 호출하여 애플리케이션을 시작합니다.
```

## 주요 특징

1. **세션 상태 관리**: Streamlit의 세션 상태를 활용하여 사용자별 데이터 유지
2. **모듈화**: UI, 도구, 그래프 처리를 각각 별도 모듈로 분리
3. **조건부 실행**: API 키 존재 여부에 따른 기능 제한
4. **사용자 친화적**: 상태 정보 표시 및 적절한 에러 메시지 제공
5. **RAG 시스템**: 문서 기반 검색 증강 생성 지원

## 워크플로우

1. 페이지 설정 및 세션 상태 초기화
2. 사이드바를 통한 API 키 입력 및 파일 업로드
3. 지식베이스 구축 (선택적)
4. 사용자 질문 입력 및 AI 응답 생성
5. 대화 기록 관리 및 상태 정보 표시