"""
RAG Diary Configuration Module
------------------------------
This module defines the configuration for different diary categories.
It centrally manages:
1.  **Display Names**: UI labels.
2.  **Database Schemas**: Table definitions, specifically supporting bilingual (English/Korean) fields.
3.  **LLM Prompts**: Instructions for the AI to extract metadata in both languages.
4.  **Enriched Templates**: How data is formatted before being embedded into the Vector DB.

Key Features:
- Bilingual Metadata Support (`_en` and `_ko` suffixes).
- Dynamic Table Generation.
"""

# Category-specific configurations
# Keys match the selection in app.py
# Unified Database Configuration
COMMON_TABLE_NAME = "tb_knowledge_base"

# Hybrid Schema: SQL (Core) + JSON (Flexible Metadata)
COMMON_SCHEMA = """
    uuid CHAR(36) PRIMARY KEY,
    log_date DATE,
    category VARCHAR(50),
    subject VARCHAR(150),
    content TEXT,
    metadata JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
"""

CATEGORY_CONFIG = {
    "Factory_Manuals": {
        "display_name": "🏭 Factory Manuals",
        "description": "For equipment troubleshooting and maintenance logs.",
        "table_name": COMMON_TABLE_NAME,
        "subject_key": "equipment_en",
        "table_schema": COMMON_SCHEMA,
        "prompt_template": """
        너는 산업 현장 데이터 분석 전문가야. 아래 [작업 일지]를 분석해서 정보를 추출해줘.
        모든 항목은 **영문(en)과 한글(ko)을 각각 분리해서** 작성해줘.
        응답은 반드시 아래 포맷(JSON 형식)을 지켜줘.
        
        {{
            "equipment_en": "Equipment Name (English)",
            "equipment_ko": "장비명 (한글)",
            "symptoms_en": "Symptoms Summary (English)",
            "symptoms_ko": "증상 요약 (한글)",
            "keywords_en": "Keywords (English, comma separated)",
            "keywords_ko": "핵심 키워드 (한글, 콤마 분리)",
            "severity_en": "Severity (Normal/Warning/Critical)",
            "severity_ko": "심각도 (정상/경고/위험)",
            "log_type_en": "Log Type (Troubleshooting/Routine/Maintenance)",
            "log_type_ko": "유형 (문제해결/점검/유지보수)",
            "summary_en": "Summary (English)",
            "summary_ko": "요약 (한글)"
        }}

        [작업 일지]:
        {text}
        """,
        "default_values": {
            "equipment_en": "Unknown", "equipment_ko": "알수없음",
            "symptoms_en": "None", "symptoms_ko": "없음",
            "keywords_en": "", "keywords_ko": "",
            "severity_en": "Normal", "severity_ko": "정상",
            "log_type_en": "Routine", "log_type_ko": "점검",
            "summary_en": "", "summary_ko": ""
        },
        "metadata_keys": ["equipment_en", "equipment_ko", "symptoms_en", "symptoms_ko", "severity_en", "severity_ko", "log_type_en", "log_type_ko", "keywords_en", "keywords_ko", "summary_en", "summary_ko"],
        "enriched_template": """[작업 일지 / {log_type_en}]
- 날짜: {date}
- 장비: {equipment_en} ({equipment_ko})
- 증상: {symptoms_en} ({symptoms_ko})
- 심각도: {severity_en} ({severity_ko})
- 내용: {content}
- 키워드: {keywords_en} / {keywords_ko}
- 요약: {summary_en}
       {summary_ko}
"""
    },
    "Personal_Diaries": {
        "display_name": "📔 Personal Diaries",
        "description": "For daily thoughts and personal records.",
        "table_name": COMMON_TABLE_NAME,
        "subject_key": "topic_en",
        "table_schema": COMMON_SCHEMA,
        "prompt_template": """
        너는 심리 상담가이자 작가야. 아래 [일기]를 분석해서 정보를 추출해줘.
        모든 항목은 **영문(en)과 한글(ko)을 각각 분리해서** 작성해줘.
        응답은 반드시 아래 포맷(JSON 형식)을 지켜줘.
        
        {{
            "emotion_en": "Emotion (e.g. Joy, Sadness)",
            "emotion_ko": "감정 (기쁨, 우울 등)",
            "topic_en": "Main Topic (English)",
            "topic_ko": "주제 (한글)",
            "keywords_en": "Keywords (English)",
            "keywords_ko": "키워드 (한글)",
            "weather_en": "Weather (English)",
            "weather_ko": "날씨 (한글)",
            "summary_en": "One-line Summary (English)",
            "summary_ko": "한 줄 요약 (한글)"
        }}

        [일기]:
        {text}
        """,
        "default_values": {
            "emotion_en": "Neutral", "emotion_ko": "평온",
            "topic_en": "Daily Life", "topic_ko": "일상",
            "keywords_en": "", "keywords_ko": "",
            "weather_en": "Unknown", "weather_ko": "알수없음",
            "summary_en": "", "summary_ko": ""
        },
        "metadata_keys": ["emotion_en", "emotion_ko", "topic_en", "topic_ko", "keywords_en", "keywords_ko", "weather_en", "weather_ko", "summary_en", "summary_ko"],
        "enriched_template": """[일기 / {emotion_en}]
- 날짜: {date}
- 주제: {topic_en} ({topic_ko})
- 날씨: {weather_en} ({weather_ko})
- 내용: {content}
- 키워드: {keywords_en} / {keywords_ko}
- 요약: {summary_en}
       {summary_ko}
"""
    },
    "Dev_Logs": {
        "display_name": "💻 Dev Logs",
        "description": "For software development and debugging notes.",
        "table_name": COMMON_TABLE_NAME,
        "subject_key": "project_en",
        "table_schema": COMMON_SCHEMA,
        "prompt_template": """
        너는 시니어 개발자야. 아래 [개발 일지]를 분석해서 정보를 추출해줘.
        모든 항목은 **영문(en)과 한글(ko)을 각각 분리해서** 작성해줘.
        응답은 반드시 아래 포맷(JSON 형식)을 지켜줘.
        
        {{
            "project_en": "Project Name (English)",
            "project_ko": "프로젝트명 (한글)",
            "task_type_en": "Task Type (Feature/Bugfix/etc)",
            "task_type_ko": "작업 유형 (기능/수정/등)",
            "tech_stack_en": "Tech Stack (English)",
            "tech_stack_ko": "기술 스택 (한글)",
            "status_en": "Status (Done/InProgress)",
            "status_ko": "상태 (완료/진행중)",
            "summary_en": "Summary (English)",
            "summary_ko": "요약 (한글)"
        }}

        [개발 일지]:
        {text}
        """,
        "default_values": {
            "project_en": "General", "project_ko": "공통",
            "task_type_en": "Feature", "task_type_ko": "기능",
            "tech_stack_en": "", "tech_stack_ko": "",
            "status_en": "Done", "status_ko": "완료",
            "summary_en": "", "summary_ko": ""
        },
        "metadata_keys": ["project_en", "project_ko", "task_type_en", "task_type_ko", "tech_stack_en", "tech_stack_ko", "status_en", "status_ko", "summary_en", "summary_ko"],
        "enriched_template": """[개발 일지 / {task_type_en}]
- 날짜: {date}
- 프로젝트: {project_en} ({project_ko})
- 상태: {status_en} ({status_ko})
- 기술 스택: {tech_stack_en}
- 내용: {content}
- 요약: {summary_en}
       {summary_ko}
"""
    },
    # Fallback / Generic
    "Ideas": {
        "display_name": "💡 Ideas",
        "description": "For general ideas and notes.",
        "table_name": COMMON_TABLE_NAME,
        "subject_key": "topic_en",
        "table_schema": COMMON_SCHEMA,
        "prompt_template": """
        너는 아이디어 뱅크야. 아래 [메모]를 분석해서 정리해줘.
        모든 항목은 **영문(en)과 한글(ko)을 각각 분리해서** 작성해줘.
        JSON 포맷 준수.
        
        {{
            "topic_en": "Topic (English)",
            "topic_ko": "주제 (한글)",
            "keywords_en": "Keywords (English)",
            "keywords_ko": "키워드 (한글)",
            "priority_en": "Priority (High/Medium/Low)",
            "priority_ko": "중요도 (상/중/하)",
            "summary_en": "Summary (English)",
            "summary_ko": "요약 (한글)"
        }}

        [메모]:
        {text}
        """,
        "default_values": {
            "topic_en": "General", "topic_ko": "공통",
            "keywords_en": "", "keywords_ko": "",
            "priority_en": "Medium", "priority_ko": "중",
            "summary_en": "", "summary_ko": ""
        },
        "metadata_keys": ["topic_en", "topic_ko", "keywords_en", "keywords_ko", "priority_en", "priority_ko", "summary_en", "summary_ko"],
        "enriched_template": """[아이디어 / {topic_en}]
- 날짜: {date}
- 중요도: {priority_en} ({priority_ko})
- 내용: {content}
- 키워드: {keywords_en} / {keywords_ko}
- 요약: {summary_en}
       {summary_ko}
"""
    }
}

def get_config(category_name):
    return CATEGORY_CONFIG.get(category_name, CATEGORY_CONFIG["Ideas"])
