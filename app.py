"""
RAG Diary Application
---------------------
This is the main Streamlit application for the RAG Diary system.

Key functionalities:
1.  **Sidebar**:
    - LLM Model Selection (Persistent).
    - Category Configuration (Persistent).
2.  **Main Interface**:
    - **Step 1**: User inputs date, content, and selects category.
    - **Step 2**: LLM analyzes content (Bilingual extraction) -> User reviews/edits metadata.
    - **Step 3**: Data is enriched with metadata -> Chunked -> Saved to MariaDB (Full Text) and ChromaDB (Index).
3.  **Data Flow**:
    - Input -> LLM -> Enriched JSON -> MariaDB (for Archival/Context) + ChromaDB (for Search Index).

"""

# Imports
import streamlit as st
import os
import datetime
import requests
import json
import uuid 
import chromadb
from sentence_transformers import SentenceTransformer

import db_utils
import category_config # Import the new config 

# --- Configuration ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://100.65.53.9:11434")
CHROMA_HOST = os.getenv("CHROMA_HOST", "100.65.53.9")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", 8001))
EMBEDDING_MODEL_NAME = "jhgan/ko-sroberta-multitask"
# VECTOR_DB_DIR = "./vector_dbs" # Removed local dir

# --- Setup Page ---
st.set_page_config(page_title="RAG Diary", page_icon="📝", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .stTextArea textarea { font-size: 16px; }
    .stButton button { width: 100%; border-radius: 5px; }
    .main-header { text-align: center; color: #4A90E2; }
    .status-box { padding: 10px; border-radius: 5px; margin-bottom: 10px;}
</style>
""", unsafe_allow_html=True)

# Initialization
if "processed_data" not in st.session_state:
    st.session_state.processed_data = None
if "step" not in st.session_state:
    st.session_state.step = 1

# Initialize DB
db_utils.init_db()

# --- Functions ---

def get_ollama_models(base_url):
    """Fetches available models from Ollama."""
    try:
        if not base_url.endswith('/'):
            base_url += '/'
        url = f"{base_url.rstrip('/')}/api/tags"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            return [model['name'] for model in data.get('models', [])]
    except Exception as e:
        st.error(f"Failed to fetch models: {e}")
    return ["llama3:latest", "mistral:latest"] # Fallback

def analyze_log_content(text, model_name, config):
    """Analyzes text to extract structured fields using Ollama (Native Requests)."""
    try:
        # Prompt Construction
        prompt_tmpl = config["prompt_template"]
        full_prompt = prompt_tmpl.format(text=text) # Assuming template uses {text}

        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": full_prompt}],
            "stream": False,
            "format": "json" # Request JSON output mode if supported
        }
        
        url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/chat"
        response = requests.post(url, json=payload, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            content = result.get('message', {}).get('content', '')
            
            # Parsing logic logic
            try:
                data = json.loads(content)
                return data
            except json.JSONDecodeError:
                # Fallback extraction if markdown code blocks exist
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0]
                    return json.loads(content)
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0]
                    return json.loads(content)
                else:
                    return config["default_values"] # Simple Failover
        else:
             st.error(f"Ollama API Error: {response.text}")
             return config["default_values"]
        
    except Exception as e:
        st.error(f"LLM Analysis Error: {e}")
        # Return default values from config on error
        return config["default_values"]

@st.cache_resource
def get_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)

def get_chroma_collection():
    """Returns the Native Chroma Collection."""
    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    return client.get_or_create_collection(name=category_config.COMMON_TABLE_NAME)

# Text Splitter Helper (Simple Regex or Native Splitter)
def simple_text_split(text, chunk_size=1000, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start += (chunk_size - overlap)
    return chunks

# Persistence File
SETTINGS_FILE = "last_settings.json"

def load_settings():
    """Loads last used settings from JSON file."""
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r") as f:
                return json.load(f)
        except:
            pass
    return {}

def save_settings(model, category):
    """Saves current settings to JSON file."""
    try:
        with open(SETTINGS_FILE, "w") as f:
            json.dump({"selected_model": model, "selected_category": category}, f)
    except Exception as e:
        print(f"Failed to save settings: {e}")

# --- UI ---

st.title("📝 RAG Diary System")
st.markdown("---")

# Sidebar
st.sidebar.title("🗂️ Settings")

# Load Settings Once
settings = load_settings()

# 2. Model Selection
st.sidebar.subheader("🤖 LLM Model")
available_models = get_ollama_models(OLLAMA_BASE_URL)

# Initialize Session State from File if not present
if "selected_model" not in st.session_state:
    st.session_state.selected_model = settings.get("selected_model", available_models[0] if available_models else "llama3:latest")

# Determine index safely
default_index = 0
if st.session_state.selected_model in available_models:
    default_index = available_models.index(st.session_state.selected_model)

selected_model = st.sidebar.selectbox(
    "Select Model",
    available_models,
    index=default_index,
    key="selected_model"
)

# 1. Category Selection (Moved to Main Area logic, but we handle init here)
category_keys = list(category_config.CATEGORY_CONFIG.keys())

if "selected_category" not in st.session_state:
    st.session_state.selected_category = settings.get("selected_category", category_keys[0])

# Main Logic
# Select Category (Main Area)
# Ensure clean category selection
current_cat = st.session_state.selected_category
cat_index = category_keys.index(current_cat) if current_cat in category_keys else 0

category = st.selectbox(
    "📂 Select Category",
    category_keys,
    index=cat_index,
    key="selected_category"
)

# Auto-Save Settings whenever selections change
if selected_model != settings.get("selected_model") or category != settings.get("selected_category"):
    save_settings(selected_model, category)

# Load Config for selected category
current_config = category_config.get_config(category)
st.info(f"**{current_config['display_name']}**\n\n{current_config['description']}")
st.divider()



# Main Logic

# Step 1: Input
st.subheader("1. Write Diary")

# Single Column Layout for Mobile Optimization
log_date = st.date_input("Date", datetime.date.today())

if "original_content" not in st.session_state:
    st.session_state.original_content = ""
content = st.text_area("Content", height=300, placeholder="Write your daily log here...")

if st.button("✨ Run AI Analysis"):
    if not content:
        st.warning("Please write some content first.")
    else:
        with st.spinner(f"AI ({selected_model}) is analyzing your text..."):
            # Pass config to analysis function
            analysis_result = analyze_log_content(content, selected_model, current_config)
            st.session_state.analysis_result = analysis_result
            st.session_state.step = 2

# Step 2: Review & Edit
if st.session_state.step >= 2:
    st.divider()
    st.subheader("2. Review & Enrichment")
    
    ar = st.session_state.analysis_result
    
    # Dynamic Form Generation based on Config Metadata Keys
    metadata_keys = current_config["metadata_keys"]
    edited_metadata = {}
    
    # Create columns dynamically (max 2 columns per row)
    cols = st.columns(2)
    for i, key in enumerate(metadata_keys):
        col = cols[i % 2]
        with col:
            # Check if key is long text (like summary or keywords) to use text_area
            if key in ["summary", "keywords", "content"]:
                 edited_metadata[key] = st.text_area(key.capitalize(), value=str(ar.get(key, '')))
            else:
                 edited_metadata[key] = st.text_input(key.capitalize(), value=str(ar.get(key, '')))
    
    # Construct Enriched Content Preview using Template
    # We need to pass all metadata + date + content to `format`
    format_data = {
        "date": log_date.strftime('%Y-%m-%d'),
        "content": content.strip(),
        **edited_metadata
    }
    
    try:
        enriched_content_preview = current_config["enriched_template"].format(**format_data)
    except KeyError as e:
        enriched_content_preview = f"Error generating preview: Missing key {e}"
        st.error(f"Template Error: {e}")

    st.info("💡 **Preview of Content to be Embedded (Enriched):**")
    st.code(enriched_content_preview, language="text")

    if st.button("✅ Confirm & Preview Chunking"):
        st.session_state.final_data = {
            "date": log_date,
            "category": category,
            "content": content, # Raw content
            "enriched_content": enriched_content_preview, # For Vector DB
            "metadata": edited_metadata # Dynamic Metadata
        }
        st.session_state.step = 3
        st.rerun()

# Step 3: Chunking Preview
if st.session_state.step >= 3:
    st.divider()
    st.subheader("3. Chunk Preview -> Final Save")
    
    data = st.session_state.final_data
    
    # Display Final Object
    with st.expander("📦 View Final Data Object", expanded=False):
        st.json(data, expanded=True)

    # Chunking - Native Splitter
    # text_splitter = RecursiveCharacterTextSplitter(...) -> REMOVED
    
    # Important: Chunk the ENRICHED content (for context in text)
    chunks = simple_text_split(data['enriched_content'], chunk_size=1000, overlap=100)
    
    # 2. Preview Metadata (User Request Assurance)
    # Construct the metadata that will be attached to every chunk
    preview_metadata = {
        "date": str(data['date']),
        "category": data['category'],
        "author": "User",
        "source_id": "(Generated after Save)",
        "table_name": current_config.get("table_name")
    }
    
    st.markdown(f"**Total Chunks:** `{len(chunks)}`")
    
    # Show Metadata Assurance
    with st.expander("🛡️ Metadata Assurance (Optimized)", expanded=True):
        st.info("We now use **ID Backtracking**. Metadata is minimized to just the ID link. The Full Context is retrieved from MariaDB.")
        st.json(preview_metadata)

    cols = st.columns(3)
    for i, chunk in enumerate(chunks):
        with cols[i % 3]:
            st.info(f"**Chunk {i+1}**\n\n{chunk}")

    if st.button("💾 Save to Database (MariaDB + ChromaDB)", type="primary"):
        with st.spinner("Saving to Databases..."):
            
            # 1. Prepare Data for Hybrid Schema
            record_uuid = str(uuid.uuid4())
            subject_key = current_config.get("subject_key", "topic_en")
            subject_value = data['metadata'].get(subject_key, "General") 
            
            # Bundle dynamic fields into JSON
            metadata_json = json.dumps(data['metadata'], ensure_ascii=False)
            
            mariadb_data = {
                "uuid": record_uuid,
                "log_date": str(data['date']),
                "category": data['category'],
                "subject": subject_value,
                "content": data['content'], # Use original_content for MariaDB
                "metadata": metadata_json,
                # created_at is handled by DB default
            }

            # 2. MariaDB Save
            table_name = current_config.get("table_name")
            db_success = False
            
            if table_name:
                # db_utils.save_to_mariadb now expects the hybrid structure
                result_id = db_utils.save_to_mariadb(table_name, mariadb_data)
                if result_id:
                    db_success = True
            else:
                st.error("Table name not found in config.")
            
            # 3. ChromaDB Save (Native)
            chroma_success = False
            if db_success:
                try:
                    collection = get_chroma_collection()
                    
                    # Prepare Embeddings
                    model = get_embedding_model()
                    embeddings = model.encode(chunks).tolist()
                    
                    # Prepare IDs (UUID + chunk index)
                    ids = [f"{record_uuid}_{i}" for i in range(len(chunks))]
                    
                    # Prepare Metadatas
                    basic_metadata = {
                        "date": str(data['date']),
                        "category": data['category'],
                        "author": "User",
                        "source_id": record_uuid,    # UUID
                        "table_name": table_name,
                        **data['metadata'] # Flattened for Chroma query capability
                    }
                    metadatas = []
                    for i in range(len(chunks)):
                        meta = basic_metadata.copy()
                        meta["chunk_index"] = i
                        metadatas.append(meta)
                    
                    # Upsert to Chroma
                    collection.upsert(
                        ids=ids,
                        embeddings=embeddings,
                        documents=chunks,
                        metadatas=metadatas
                    )
                    chroma_success = True
                except Exception as e:
                    st.error(f"ChromaDB Error: {e}")
            
            # Result
            if db_success and chroma_success:
                st.success(f"🎉 Successfully saved to BOTH MariaDB (UUID: {record_uuid}) and ChromaDB!")
                if st.button("Start New Entry"):
                    # Clear session state but PRESERVE user settings
                    # We explicitly remove data-related keys to force widget reset
                    keys_to_clear = ["processed_data", "analysis_result", "final_data", "original_content"]
                    for key in keys_to_clear:
                        if key in st.session_state:
                            del st.session_state[key]
                    
                    # Reset Step
                    st.session_state.step = 1
                    st.rerun()
            elif inserted_id:
                st.warning(f"Saved to MariaDB (ID: {inserted_id}) but FAILED ChromaDB.")
            elif chroma_success:
                st.warning("Saved to ChromaDB but FAILED MariaDB (Critical Data Sync Issue).") # Should not happen with if logic
            else:
                st.error("Failed to save to both databases.")
