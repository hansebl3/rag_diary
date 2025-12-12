### 🛠️ 1. 확정된 프로그램 구동 시나리오

이 프로그램은 \*\*"작성 -\> AI 변환 -\> 검토(수정) -\> 청킹 확인 -\> 이중 저장(SQL+Vector)"\*\*의 흐름을 따릅니다.

1.  **DB 선택 (Setup):**
      * 사이드바에서 저장할 \*\*카테고리(Vector DB 종류)\*\*를 선택합니다. (예: `공장_설비_매뉴얼`, `개인_일기`, `개발_노트`)
2.  **일기 작성 (Input):**
      * 사용자가 텍스트 창에 날짜와 원문 내용을 작성합니다.
3.  **AI 변환 (Transform):**
      * [변환] 버튼 클릭 -\> LLM이 원문을 읽고 \*\*'요약문'\*\*과 \*\*'키워드(태그)'\*\*를 자동으로 생성합니다.
4.  **검토 및 수정 (Review & Edit):**
      * 화면에 **[날짜 | 카테고리 | 원문 | 요약 | 키워드]** 입력란이 모두 표시됩니다.
      * 사용자는 LLM이 만든 요약이 마음에 안 들면 직접 수정합니다.
      * [컨펌(확인)] 버튼 클릭 -\> 이 데이터들이 하나의 \*\*객체(Object)\*\*로 메모리에 확정됩니다.
5.  **분해 결과 미리보기 (Chunk Preview):**
      * LangChain이 데이터를 벡터 DB에 넣기 전, 실제로 어떻게 쪼개는지(Chunking) 눈으로 보여줍니다. (예: 500자씩 3개로 나뉨 등)
6.  **최종 저장 (Commit):**
      * [DB Insert] 버튼 클릭.
      * **동작 A (MariaDB):** `tb_knowledge_log` 테이블에 원본, 요약, 키워드를 영구 저장 (백업용).
      * **동작 B (Vector DB):** 선택한 카테고리의 ChromaDB에 임베딩하여 저장 (검색용).
      * "저장이 완료되었습니다." 메시지와 함께 입력창 초기화.

-----

### 📊 2. MariaDB 테이블 설계 (백업용)

백업을 위해 미니서버의 MariaDB에 들어갈 테이블 구조입니다.

  * **테이블명:** `tb_daily_log` (또는 `tb_knowledge_base`)
  * **컬럼 구성:**
      * `id` (INT, PK): 고유 번호
      * `created_at` (DATETIME): 작성 시간
      * `category` (VARCHAR): 선택한 DB 종류 (공장용/개인용 등)
      * `content_body` (TEXT): 원문 내용
      * `summary` (TEXT): AI가 만든(또는 수정된) 요약
      * `keywords` (VARCHAR): 추출된 키워드

-----

### 📋 3. IDE AI에게 줄 상세 구현 프롬프트

이 내용을 복사해서 IDE(Cursor/Copilot)에 입력하세요. MariaDB 연결과 시나리오가 완벽하게 포함되어 있습니다.

**[지시 사항]**
너는 **Senior Python Developer**야.
Streamlit, LangChain, MariaDB, ChromaDB를 연동하여 \*\*"데이터 생성 및 이중 저장 파이프라인"\*\*을 갖춘 RAG 시스템을 구축해줘.

**1. 프로젝트 목표**

  - 사용자가 작성한 문서를 AI가 요약/태깅하고, 사용자가 검토한 뒤, \*\*MariaDB(영구보존)\*\*와 \*\*ChromaDB(벡터검색)\*\*에 동시 저장하는 시스템.

**2. 기술 스택**

  - **App:** Streamlit
  - **Logic:** LangChain Community, OpenAI (ChatOpenAI)
  - **DB (Structured):** MariaDB (pymysql 라이브러리 사용)
  - **DB (Vector):** ChromaDB (Local Persist)
  - **Deployment:** Docker

**3. 핵심 기능 명세 (Step-by-Step Scenario)**

**(A) 사이드바: 카테고리 관리**

  - `st.sidebar.selectbox`로 **DB 카테고리(폴더명)** 선택 (예: `Factory`, `Personal`, `Dev`).
  - 선택된 값은 데이터 저장 시 `category` 태그로 사용됨.

**(B) 메인 화면: 작성 및 변환 프로세스 (Session State 필수)**

1.  **Step 1: 입력**
      - `st.date_input`: 날짜 선택.
      - `st.text_area`: 일기(원문) 작성.
      - `[AI 변환 실행]` 버튼.
2.  **Step 2: AI 처리 및 검토 (수정 가능)**
      - 버튼 클릭 시 LLM에게 프롬프트를 보내 \*\*요약문(Summary)\*\*과 \*\*키워드(Keywords)\*\*를 추출.
      - 결과를 `st.text_input` (키워드), `st.text_area` (요약문)에 보여주어 **사용자가 직접 수정**할 수 있게 함.
      - `[컨펌(Confirm)]` 버튼 생성.
3.  **Step 3: 객체 생성 및 청킹 미리보기**
      - 컨펌 버튼 클릭 시, 입력된 데이터(날짜, 카테고리, 원문, 수정된 요약, 키워드)를 `session_state`에 딕셔너리 객체로 저장.
      - LangChain `RecursiveCharacterTextSplitter`를 사용하여 원문을 어떻게 쪼갤지 시뮬레이션하고, 그 결과(Chunk List)를 화면에 리스트로 보여줌.
      - `[DB 저장(Insert)]` 버튼 생성.
4.  **Step 4: 이중 저장 (Dual Save)**
      - 저장 버튼 클릭 시:
          - **MariaDB:** `tb_daily_log` 테이블에 INSERT (컬럼: date, category, content, summary, keywords).
          - **ChromaDB:** 선택된 카테고리 경로(`./vector_dbs/{category}`)에 문서 저장. 이때 메타데이터에 `summary`, `keywords`를 반드시 포함.
      - 성공 시 `st.success` 메시지와 함께 상태 초기화.

**4. 데이터베이스 설정 (MariaDB)**

  - 도커 환경에서 실행되므로 MariaDB 호스트는 `host.docker.internal` 또는 환경변수로 받도록 설정.
  - 코드 실행 시 테이블이 없으면 자동 생성하는 `CREATE TABLE IF NOT EXISTS` 로직 포함.
    ```sql
    CREATE TABLE IF NOT EXISTS tb_daily_log (
        id INT AUTO_INCREMENT PRIMARY KEY,
        log_date DATE,
        category VARCHAR(50),
        content TEXT,
        summary TEXT,
        keywords VARCHAR(255),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    ```

**5. 파일 요구사항**

  - `app.py`: 위 로직이 구현된 메인 코드.
  - `db_utils.py`: MariaDB 연결 및 쿼리 실행을 담당하는 헬퍼 함수 모음.
  - `requirements.txt`: `pymysql`, `langchain`, `streamlit` 등 포함.
  - `docker-compose.yml`: MariaDB 서비스와 Streamlit 앱을 함께 띄우거나, 외부 MariaDB 연결 설정 포함.

위 명세에 맞춰 **에러 핸들링이 포함된 완성된 코드**를 작성해줘.

-----

### 💡 사용자님을 위한 팁 (도커 네트워크)

미니서버(우분투)에 이미 MariaDB가 설치되어 돌고 있다면, 도커 컨테이너에서 그 MariaDB에 접속하기 위해 `docker-compose.yml` 파일의 환경변수 부분에 **DB 호스트 주소**를 잘 적어야 합니다.

  * **같은 도커 컴포즈 내에 있다면:** 서비스 이름 (예: `mariadb`)
  * **호스트(우분투)에 깔린 DB라면:** `host.docker.internal` (Linux Docker에서는 추가 설정 필요할 수 있음) 또는 **서버의 내부 IP** (예: `192.168.0.x`)를 쓰는 것이 가장 확실합니다.

이대로 진행하면 나중에 벡터 DB를 실수로 지워도, MariaDB에 있는 데이터를 불러와서 **"벡터 DB 재구축(Re-indexing)"** 버튼 하나만 만들면 1분 만에 복구되는 강력한 시스템이 됩니다.