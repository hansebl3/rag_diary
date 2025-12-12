"""
RAG Diary Database Utilities
----------------------------
This module handles all interactions with MariaDB (SQL Database).
It provides functions to:
1.  Connect to the database.
2.  Initialize tables based on the category configuration.
3.  Save diary entries dynamically to the correct table.
"""

import pymysql
import os
import streamlit as st
import category_config

DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "lavita!978")
DB_NAME = os.getenv("DB_NAME", "rag_diary_db")

def get_db_connection():
    """Establishes a connection to the MariaDB database."""
    try:
        connection = pymysql.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )
        return connection
    except pymysql.MySQLError as e:
        st.error(f"Error connecting to MariaDB: {e}")
        return None

def init_db():
    """Initializes the database and creates tables based on config."""
    conn = get_db_connection()
    if conn:
        try:
            with conn.cursor() as cursor:
                # Create Database if not exists
                cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_NAME}")
                cursor.execute(f"USE {DB_NAME}")
                
                # Iterate through all categories and create their specific tables
                for cat_key, config in category_config.CATEGORY_CONFIG.items():
                    table_name = config.get("table_name")
                    table_schema = config.get("table_schema")
                    
                    if table_name and table_schema:
                        create_query = f"CREATE TABLE IF NOT EXISTS {table_name} ({table_schema})"
                        # st.write(f"Creating table: {table_name}") # Debug
                        cursor.execute(create_query)
                        
            conn.commit()
        except pymysql.MySQLError as e:
            st.error(f"Error initializing DB: {e}")
            st.error(f"Query was: {create_query}")
        finally:
            conn.close()

def save_to_mariadb(table_name, data_dict):
    """
    Saves the log entry to the specific MariaDB table dynamically.
    
    Args:
        table_name (str): Name of the table to insert into.
        data_dict (dict): Dictionary of column_name: value.

    Returns:
        int or None: The `id` of the inserted row if successful, None otherwise.
    """
    conn = get_db_connection()
    if conn:
        try:
            with conn.cursor() as cursor:
                cursor.execute(f"USE {DB_NAME}")
                
                # Dynamic Insert Construction
                
                # [Safety Feature] Fetch valid columns to prevent "Unknown column" errors
                cursor.execute(f"SHOW COLUMNS FROM {table_name}")
                valid_columns = {row['Field'] for row in cursor.fetchall()}
                
                # Filter data_dict to only include valid columns
                filtered_data = {k: v for k, v in data_dict.items() if k in valid_columns}
                
                if not filtered_data:
                    st.warning(f"No valid data fields found for table {table_name}.")
                    return None

                columns = list(filtered_data.keys())
                placeholders = ["%s"] * len(columns)
                values = list(filtered_data.values())
                
                col_str = ", ".join(columns)
                val_str = ", ".join(placeholders)
                
                insert_query = f"INSERT INTO {table_name} ({col_str}) VALUES ({val_str})"
                
                cursor.execute(insert_query, values)
                
                # For Hybrid Schema with UUID, lastrowid might be 0.
                if 'uuid' in filtered_data:
                    inserted_id = filtered_data['uuid']
                else:
                    inserted_id = cursor.lastrowid 
            
            conn.commit()
            return inserted_id # Return ID/UUID
        except pymysql.MySQLError as e:
            st.error(f"Error saving to MariaDB Table '{table_name}': {e}")
            return None
        finally:
            conn.close()
    return None
