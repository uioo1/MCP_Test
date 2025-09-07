프로젝트 명세서: AI 에이전트를 활용한 고객 문의 처리 자동화 워크플로우
1. 프로젝트 개요 (Overview)
본 프로젝트는 LangGraph를 사용하여 고객 문의를 자동으로 분류하고, 지식 베이스(RAG)를 활용해 답변을 생성하거나, 복잡한 문제일 경우 담당팀에 에스컬레이션하는 에이전트 워크플로우(Agentic Workflow) MVP를 구축하는 것을 목표로 합니다.

사용자는 Streamlit으로 제작된 웹 UI를 통해 질문을 제출하고, AI 에이전트는 사전에 정의된 워크플로우에 따라 지능적으로 요청을 처리합니다. 이 과정에서 MCP(Model-Controller-Parser) 디자인 패턴을 적용하여 그래프의 상태(State)를 관리하고, 조건에 따라 흐름을 제어합니다.

2. 핵심 기술 스택 (Core Technologies)
Web Framework: Streamlit

Workflow Engine: LangGraph

LLM Orchestration: LangChain

LLM Provider: OpenAI

Knowledge Base (RAG): FAISS (Vector Store), LangChain Document Loaders

3. 시스템 아키텍처 (System Architecture)
```
+--------------------------+
|     Streamlit Web UI     | (main.py, ui.py)
+-------------+------------+
              | (User Query)
              v
+-------------+------------+
|  LangGraph Workflow App  | (graph.py)
+-------------+------------+
              |
              v (Start)
+-------------+------------+
| Node 1: Classify Inquiry |
| (문의 유형 분류)         |
+-------------+------------+
              |
              v (Conditional Routing)
+--------------------------+---------------------------+
| Route A: Simple Question |  Route B: Complex Question  |
+--------------------------+---------------------------+
              |                            |
              v                            v
+-------------+------------+ +-------------+-------------+
| Node 2A: Retrieve Docs | | Node 2B: Prep Escalation|
| (RAG - 문서 검색)      | | (에스컬레이션 준비)     |
+-------------+------------+ +-------------+-------------+
              |                            |
              v                            v
+-------------+------------+ +-------------+-------------+
| Node 3A: Generate Answer | | Node 3B: Format Message   |
| (RAG - 답변 생성)      | | (팀 전달 메시지 포맷팅) |
+-------------+------------+ +-------------+-------------+
              |                            |
              +-------------+--------------+
                            |
                            v
+--------------------------+
|          End             |
|  (결과를 UI에 반환)      |
+--------------------------+
```

4. 프로젝트 파일 구조 (File Structure)
```
.
├── 📁 faiss_index/      # 벡터 데이터베이스 저장소
├── 📄 .env               # API 키 등 환경 변수 파일
├── 📄 app.py              # Streamlit 앱 실행을 위한 메인 진입점
├── 📄 ui.py                # Streamlit UI 컴포넌트 관리 모듈
├── 📄 graph.py             # LangGraph 워크플로우(상태, 노드, 엣지) 정의 모듈
├── 📄 tools.py             # RAG Retriever 등 AI 에이전트가 사용할 도구 모음
├── 📄 utils.py             # 문서 로딩, 텍스트 분할 등 유틸리티 함수 모듈
└── 📄 requirements.txt     # 프로젝트 의존성 패키지 목록
```

5. 파일별 상세 명세 (Detailed File Specifications)
📄 app.py
목적: Streamlit 웹 애플리케이션의 메인 실행 파일.

핵심 로직:

필요한 라이브러리와 모듈(streamlit, ui, graph, utils)을 임포트합니다.

dotenv를 사용하여 .env 파일의 환경 변수(특히 OPENAI_API_KEY)를 로드합니다.

st.session_state를 사용하여 앱의 상태(대화 기록, 그래프 객체 등)를 초기화하고 관리합니다.

ui.py에 정의된 함수들을 호출하여 사이드바와 메인 채팅 인터페이스를 렌더링합니다.

사용자로부터 파일 업로드 및 텍스트 입력을 받아, 이를 처리할 utils.py 및 graph.py의 함수로 전달합니다.

그래프로부터 반환된 최종 결과를 받아 ui.py를 통해 화면에 출력합니다.

📄 ui.py
목적: Streamlit UI 컴포넌트의 생성과 관리를 담당. app.py의 코드를 간결하게 유지.

주요 함수:

setup_sidebar():

OpenAI API 키를 입력받는 st.text_input (type="password") 생성.

지식 베이스로 사용할 문서를 업로드하는 st.file_uploader (PDF, TXT) 생성.

업로드된 문서를 처리하고 RAG를 준비하는 st.button("지식 베이스 구축") 생성.

render_chat_interface():

st.session_state에 저장된 대화 기록(chat_history)을 순회하며 st.chat_message로 기존 대화를 표시.

st.chat_input을 통해 사용자로부터 새로운 질문을 입력받음.

새로운 질문이 입력되면, 이를 처리할 로직(그래프 실행)을 호출하고 st.spinner로 로딩 상태를 표시.

📄 graph.py
목적: LangGraph 워크플로우의 모든 로직(상태, 노드, 엣지)을 정의. MCP 패턴의 핵심 구현부.

주요 구성 요소:

State (Model): TypedDict를 사용하여 그래프의 상태(GraphState)를 정의. 포함될 key:

question: 사용자의 원본 질문 (str)

classification: 문의 분류 결과 (Literal["simple", "complex"])

documents: RAG 검색 결과 (List[Document])

answer: LLM이 생성한 최종 답변 (str)

chat_history: 전체 대화 기록 (List[BaseMessage])

Nodes (Parser/Processors): 각 단계에 해당하는 Python 함수들.

classify_inquiry(state): question을 받아 LLM을 이용해 "simple" 또는 "complex"로 분류. 결과를 classification에 저장.

retrieve_documents(state): question을 받아 tools.py의 RAG Retriever로 문서를 검색. 결과를 documents에 저장.

generate_rag_answer(state): question과 documents를 프롬프트에 담아 LLM으로 답변 생성. 결과를 answer에 저장.

prepare_escalation(state): question을 바탕으로 "담당팀에 전달할 문의 내용입니다: [질문]" 형식의 메시지를 생성. 결과를 answer에 저장.

Edges (Controller): 노드 간의 흐름을 제어하는 조건부 엣지.

route_after_classification(state): classification state 값을 확인하여 retrieve_documents로 갈지, prepare_escalation으로 갈지 결정.

Graph Builder:

build_graph():

StateGraph(GraphState)로 그래프 객체 생성.

정의된 노드들을 graph.add_node()로 추가.

graph.set_entry_point()로 classify_inquiry 노드를 시작점으로 설정.

graph.add_conditional_edges()로 classify_inquiry 노드 이후의 분기 로직을 설정.

나머지 흐름을 graph.add_edge()로 연결.

graph.compile()로 실행 가능한 앱(app)을 반환.

📄 tools.py
목적: LangGraph 노드에서 호출하여 사용할 수 있는 재사용 가능한 도구들을 정의.

주요 함수:

create_retriever(files, embeddings):

utils.py의 헬퍼 함수를 호출하여 업로드된 파일(files)을 로드하고 텍스트를 분할.

분할된 텍스트를 embeddings를 사용해 벡터로 변환하고 FAISS 벡터 스토어를 생성.

생성된 벡터 스토어로부터 as_retriever()를 호출하여 검색기(retriever) 객체를 반환.

📄 utils.py
목적: 특정 도메인에 종속되지 않는 범용 헬퍼 함수들을 모아놓은 모듈.

주요 함수:

load_documents(uploaded_files):

streamlit.UploadedFile 객체 리스트를 입력받음.

파일 확장자(.pdf, .txt)를 확인하여 적절한 LangChain의 DocumentLoader(PyPDFLoader, TextLoader)를 사용해 문서를 로드하고 List[Document] 형태로 반환.

임시 파일 저장이 필요할 경우 tempfile 모듈 활용.

split_text(documents):

List[Document]를 입력받아 RecursiveCharacterTextSplitter를 사용해 적절한 크기의 청크(chunk)로 분할하여 반환.

📄 requirements.txt
목적: 프로젝트 실행에 필요한 모든 Python 패키지와 버전을 명시.

내용:
```
streamlit
langchain
langgraph
langchain-openai
langchain-community
faiss-cpu
pypdf
tiktoken
python-dotenv
```
