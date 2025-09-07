# AI 고객 문의 처리 시스템

Streamlit 기반의 AI 고객 문의 처리 시스템입니다. 문서 업로드를 통해 지식 베이스를 구축하고, RAG(Retrieval-Augmented Generation) 기술을 활용하여 고객 문의에 정확한 답변을 제공합니다.

## 🚀 주요 기능

- **문서 기반 지식 베이스**: PDF 문서를 업로드하여 자동으로 지식 베이스 구축
- **RAG 기반 질의응답**: 업로드된 문서를 기반으로 정확한 답변 생성
- **실시간 채팅 인터페이스**: 사용자 친화적인 채팅 UI
- **FAISS 벡터 검색**: 고성능 벡터 유사도 검색
- **지식 베이스 영속성**: 구축된 지식 베이스 자동 저장 및 로드

## 🛠️ 기술 스택

- **Frontend**: Streamlit
- **LLM Framework**: LangChain, LangGraph
- **Vector Database**: FAISS
- **LLM Provider**: OpenAI GPT
- **Document Processing**: PyPDF

## 📦 설치 방법

1. 저장소 클론
```bash
git clone <repository-url>
cd MCP
```

2. 가상환경 생성 및 활성화
```bash
python -m venv env
source env/bin/activate  # Windows: env\Scripts\activate
```

3. 의존성 설치
```bash
pip install -r requirements.txt
```

4. 환경 변수 설정
`.env` 파일을 생성하고 OpenAI API 키를 설정합니다:
```env
OPENAI_API_KEY=your_openai_api_key_here
```

## 🚀 실행 방법

```bash
streamlit run app.py
```

브라우저에서 `http://localhost:8501`로 접속하여 애플리케이션을 사용할 수 있습니다.

## 📁 프로젝트 구조

```
MCP/
├── app.py              # 메인 애플리케이션
├── ui.py               # UI 컴포넌트 및 인터페이스
├── tools.py            # 도구 및 유틸리티 함수
├── graph.py            # LangGraph 워크플로우
├── utils.py            # 공통 유틸리티 함수
├── requirements.txt    # 프로젝트 의존성
├── .env               # 환경 변수 (생성 필요)
├── docs/              # 문서 폴더
└── faiss_index/       # FAISS 인덱스 저장 폴더
```

## 💡 사용 방법

1. **API 키 설정**: 사이드바에서 OpenAI API 키를 입력합니다.

2. **문서 업로드**: PDF 파일을 업로드하여 지식 베이스를 구축합니다.

3. **질문하기**: 채팅 인터페이스를 통해 업로드한 문서에 관련된 질문을 입력합니다.

4. **답변 확인**: AI가 문서 내용을 기반으로 정확한 답변을 제공합니다.

## 🔧 주요 모듈

### app.py
- 메인 애플리케이션 실행 파일
- Streamlit 설정 및 세션 상태 관리

### ui.py
- 사용자 인터페이스 컴포넌트
- 채팅 인터페이스 및 사이드바 구성

### tools.py
- 문서 처리 및 벡터 데이터베이스 관련 도구
- FAISS 인덱스 저장/로드 기능

### graph.py
- LangGraph를 활용한 워크플로우 구성
- RAG 파이프라인 구현

## 🤝 기여 방법

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 라이센스

이 프로젝트는 MIT 라이센스를 따릅니다.

## 📧 연락처

프로젝트에 대한 질문이나 제안이 있으시면 이슈를 등록해주세요.