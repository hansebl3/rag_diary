
---

### ✅ Phase 1: 핵심 구조 변경 (UUID & 스키마)
**"데이터 무결성 확보를 위한 기초 공사"**입니다. 가장 먼저 해야 합니다.

* [ ] **스키마 업데이트 (`category_config.py`)**
    * 모든 카테고리의 `table_schema`에 `uuid CHAR(36) NOT NULL UNIQUE` 컬럼 추가.
* [ ] **UUID 생성 로직 추가 (`app.py`)**
    * `import uuid` 추가.
    * **Step 3 (Chunk Preview)** 단계에서 `uuid.uuid4()`로 고유 ID 생성.
    * `session_state.final_data` 및 메타데이터에 `uuid` 포함시키기.
* [ ] **DB 초기화 (`reset_db_schema.py`)**
    * 변경된 스키마 반영을 위해 스크립트 실행하여 기존 테이블 Drop & Re-create.



### 🛠️ Phase 2: 관리자 페이지 (CMS) 구축
**"수정 및 재색인(Re-indexing) 기능"**을 만드는 단계입니다.

* [ ] **멀티 페이지 구조 생성**
    * 프로젝트 루트에 `pages/` 폴더 생성.
    * `pages/admin.py` 파일 생성.
* [ ] **관리자 UI 구현 (`pages/admin.py`)**
    * **조회:** 카테고리/날짜 선택 → 일기 목록 표시.
    * **수정:** 일기 선택 시 기존 내용(`content`, `keywords` 등) 불러와서 에디터에 표시.
* [ ] **동기화 저장 로직 구현**
    * **[저장]** 버튼 클릭 시:
        1.  **MariaDB:** `UPDATE` 문으로 내용 갱신 (UUID 유지).
        2.  **ChromaDB:** 해당 `source_uuid`를 가진 벡터 **삭제(Delete)** 후, 수정된 내용으로 **재등록(Add)**.



### 🚀 Phase 3: 배포 및 테스트
**"실전 가동 준비"** 단계입니다.

* [ ] **도커 환경 점검 (`docker-compose.yml`)**
    * `network_mode: "host"` 상태에서 MariaDB 및 Ollama 접속 테스트.
* [ ] **통합 테스트 (E2E Test)**
    * **Case 1:** 신규 일기 작성 → MariaDB/ChromaDB 저장 확인 → UUID 생성 확인.
    * **Case 2:** 관리자 페이지 접속 → 기존 일기 수정 → ChromaDB에서 검색 시 수정된 내용 반영되는지 확인.

