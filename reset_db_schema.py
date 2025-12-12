import pymysql
import os
import category_config
import db_utils
import chromadb

# DB Config (matching docker-compose/app env)
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "lavita!978")
DB_NAME = os.getenv("DB_NAME", "rag_diary_db")

# Chroma Config
CHROMA_HOST = os.getenv("CHROMA_HOST", "100.65.53.9")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", 8001))

def reset_schema():
    print("=== 🛠️ RAG Diary System Reset Tool 🛠️ ===")
    
    # 1. MariaDB Reset
    print(f"\n[1/3] 🔌 Connecting to MariaDB ({DB_NAME})...")
    try:
        conn = pymysql.connect(
            host=DB_HOST, user=DB_USER, password=DB_PASSWORD, charset='utf8mb4'
        )
        with conn.cursor() as cursor:
            cursor.execute(f"USE {DB_NAME}")
            
            # Drop Legacy Tables
            LEGACY_TABLES = ["tb_factory_manuals", "tb_personal_diaries", "tb_dev_logs", "tb_ideas"]
            for table in LEGACY_TABLES:
                cursor.execute(f"DROP TABLE IF EXISTS {table}")
            
            # Drop Configured Tables
            target_tables = set(config.get("table_name") for config in category_config.CATEGORY_CONFIG.values())
            for table_name in target_tables:
                if table_name:
                    cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
            
        conn.commit()
        print("✅ MariaDB tables cleared.")
    except Exception as e:
        print(f"❌ MariaDB Error: {e}")
    finally:
        if 'conn' in locals() and conn.open:
            conn.close()

    # 2. ChromaDB Reset
    print(f"\n[2/3] 🔌 Connecting to ChromaDB ({CHROMA_HOST}:{CHROMA_PORT})...")
    try:
        client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        
        # We need to clear collections for all known categories
        # Collections are named after the dictionary keys in CATEGORY_CONFIG
        target_collections = list(category_config.CATEGORY_CONFIG.keys())
        
        for col_name in target_collections:
            try:
                client.delete_collection(col_name)
                print(f"   -> Deleted collection: {col_name}")
            except ValueError:
                # Collection doesn't exist
                pass
            except Exception as e:
                print(f"   Warning deleting {col_name}: {e}")
                
        print("✅ ChromaDB collections cleared.")
    except Exception as e:
        print(f"❌ ChromaDB Error: {e}")

    # 3. Re-initialize
    print("\n[3/3] 🔄 Re-initializing Database Schema...")
    try:
        db_utils.init_db()
        print("🚀 System Reset & Re-initialized for Unified Structure!")
    except Exception as e:
        print(f"❌ Re-init Error: {e}")

if __name__ == "__main__":
    reset_schema()
