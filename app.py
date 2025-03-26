import os
import sqlite3
import streamlit as st
import pandas as pd
import google.generativeai as genai
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
import re
import speech_recognition as sr
import sqlparse

# Load API Key
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Streamlit App Configuration
st.set_page_config(page_title="Advanced CSV to SQL Query Analysis", layout="wide")

def clean_column_name(name):
    """Sanitize column names: replace spaces/dots with underscores & remove special characters."""
    name = name.strip().replace(" ", "_").replace(".", "_")
    name = re.sub(r'\W+', '', name)  # Remove non-alphanumeric characters (except _)
    return name
# Function to Create Database Table from CSV
def create_table_from_csv(file, db_name, table_name):
    df = pd.read_csv(file)

    # Sanitize column names
    df.columns = [clean_column_name(col) for col in df.columns]

    # Auto-detect column types
    dtype_mapping = {
        "int64": "INTEGER",
        "float64": "REAL",
        "object": "TEXT"
    }
    column_types = ", ".join([f'"{col}" {dtype_mapping[str(df[col].dtype)]}' for col in df.columns])

    conn = sqlite3.connect(db_name)
    cur = conn.cursor()

    # Drop table if exists (to avoid duplication issues)
    cur.execute(f'DROP TABLE IF EXISTS "{table_name}"')

    # Create Table
    cur.execute(f'CREATE TABLE "{table_name}" ({column_types})')

    # Insert Data
    df.to_sql(table_name, conn, if_exists="append", index=False)
    conn.commit()
    conn.close()
    
    return df, table_name
# Function to Generate SQL Query from Natural Language
def get_sql_query(nl_question, table_name, columns):
    prompt_template = f"""
    You are an expert in converting English questions to SQL queries.
    The SQL database contains a table '{table_name}' with the following columns: {', '.join(columns)}.

    Example 1: "How many rows are there?"
    SQL Query: SELECT COUNT(*) FROM {table_name};

    Example 2: "Show me all records where column 'X' is greater than 10."
    SQL Query: SELECT * FROM {table_name} WHERE X > 10;

    Important: Do not include '```sql' or '```' in your output.
    """
    
    model = genai.GenerativeModel('gemini-2.0-flash')
    response = model.generate_content([prompt_template, nl_question])
    return response.text.strip()

def format_sql_query(sql_query):
    try:
        formatted_query = sqlparse.format(sql_query, reindent=True, keyword_case="upper")
        
        # Ensure WHERE and other clauses are not misplaced
        formatted_query = formatted_query.replace("\n\n", "\n")  # Remove excessive newlines
        
        return formatted_query
    except Exception as e:
        return sql_query  # Return original query if formatting fails



# Function to Execute SQL Query
def execute_sql_query(db_name, sql_query):
    conn = sqlite3.connect(db_name)
    cur = conn.cursor()
    
    try:
        cur.execute(sql_query)
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description]
        conn.close()
        return rows, columns
    except Exception as e:
        conn.close()
        return str(e), None

def generate_ai_summary(query, data):
    summary_prompt = f"""
    You are an AI assistant that summarizes SQL query results.
    The query executed was: {query}.
    The result of the query is: {data}.
    
    Provide a human-friendly summary in 2-3 sentences.
    """
    model = genai.GenerativeModel('gemini-2.0-flash')
    response = model.generate_content(summary_prompt)
    return response.text.strip()


def voice_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎙️ Speak Now...")
        audio = recognizer.listen(source)
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return "Sorry, I could not understand your voice."
    except sr.RequestError:
        return "Error connecting to speech service."

# Function to Generate Charts
def generate_chart(df):
    st.subheader("📊 Data Visualization")
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns

    if not numeric_columns.empty:
        chart_type = st.selectbox("Select Chart Type", ["Bar", "Line", "Scatter"])
        x_axis = st.selectbox("Select X-axis", numeric_columns)
        y_axis = st.selectbox("Select Y-axis", numeric_columns)

        fig, ax = plt.subplots()
        if chart_type == "Bar":
            sns.barplot(x=df[x_axis], y=df[y_axis], ax=ax)
        elif chart_type == "Line":
            sns.lineplot(x=df[x_axis], y=df[y_axis], ax=ax)
        elif chart_type == "Scatter":
            sns.scatterplot(x=df[x_axis], y=df[y_axis], ax=ax)

        st.pyplot(fig)
    else:
        st.warning("No numeric columns available for visualization.")

# Streamlit UI
st.title("📊 Advanced CSV to SQL Query Analysis")

# Step 1: Upload Multiple CSV Files
st.sidebar.header("Upload CSV Files")
uploaded_files = st.sidebar.file_uploader("Upload one or more CSV files", type=["csv"], accept_multiple_files=True)

if uploaded_files:
    db_name = "user_data.db"
    st.sidebar.subheader("Select Table to Query")
    table_mapping = {}

    for i, file in enumerate(uploaded_files):
        table_name = f"table_{i+1}"
        df, table_name = create_table_from_csv(file, db_name, table_name)
        table_mapping[file.name] = table_name

    selected_table = st.sidebar.selectbox("Choose a Table", list(table_mapping.keys()))
    table_name = table_mapping[selected_table]

    st.sidebar.success(f"Table '{table_name}' ready for queries! ✅")

    # Display Data Preview
    st.subheader("📂 Uploaded Data Preview")
    st.write(df.head())

    # Step 2: User Asks a Question in Natural Language
    st.subheader("💬 Ask a Question or Write SQL Query")
    query_option = st.radio(
        "Choose how you want to run a query:",
        ("Use Natural Language", "Write SQL Manually" , "Speak"),
    )
    sql_query = ""
    
    if query_option == "Use Natural Language":
        user_question = st.text_input("Enter your question:")
        if user_question:
            st.button("Generate Query")
            generated_query = get_sql_query(user_question, table_name, df.columns)
            edit_sql_query = st.text_area("Edit SQL Query:", value=generated_query, height=150)
            sql_query = format_sql_query(edit_sql_query)
            st.code(f"📝 SQL Query: {sql_query}")

    elif query_option == "Write SQL Manually":
        write_sql_query = st.text_area("Write your SQL Query here:")
        sql_query = format_sql_query(write_sql_query)
    
    elif query_option == "Speak":
        st.button("🎙️ Speak Your Query")

        user_question = voice_input()
        st.success(f"Recognized: {user_question}")
        sql_query = get_sql_query(user_question, table_name, df.columns)


    if sql_query and st.button("Execute Query"):
        st.code(f"📝 SQL Query: {sql_query}")
        with st.spinner("Executing Query..."):
            
            results, columns = execute_sql_query(db_name, sql_query)

            if columns:
                df_results = pd.DataFrame(results, columns=columns)
                st.subheader("🔍 Query Results")
                st.dataframe(df_results)

                st.subheader("🤖 AI-Generated Summary")
                summary = generate_ai_summary(sql_query, df_results.to_dict())
                st.write(summary)

                # Generate Charts
                generate_chart(df_results)
            else:
                st.error(f"❌ SQL Execution Error: {results}")

