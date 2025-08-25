# app.py
# Import necessary libraries
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import re
import traceback
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.sql import SQLDatabaseChain
from langchain_community.utilities import SQLDatabase
import json
from sqlalchemy import text

# Load environment variables from a .env file
load_dotenv()

# Initialize the Flask application
app = Flask(__name__, template_folder='templates') # Ensures Flask looks in the 'templates' folder
CORS(app)

# --- Helper Functions ---

def clean_sql_query(query):
    """Removes markdown formatting from the SQL query returned by the LLM."""
    query = re.sub(r"```sql\n?", "", query)
    query = re.sub(r"```\n?", "", query)
    return query.strip()

def is_sql_query(llm, question):
    """Uses the LLM to determine if a question is asking for data or is conversational."""
    # This prompt asks the LLM for a simple Yes/No answer.
    prompt = f"""
Is the following question asking for data from a database, or is it a general conversational question (like a greeting or a random query)?
Respond with only 'Yes' if it is asking for data, or 'No' if it is conversational.

Question: "{question}"
Answer:"""
    
    try:
        response = llm.invoke(prompt).content.strip().lower()
        print(f"Intent check for '{question}': AI responded '{response}'")
        return "yes" in response
    except Exception as e:
        print(f"Error during intent check: {e}")
        # Default to assuming it's a SQL query if the check fails
        return True

def generate_natural_response(llm, question, sql_query, result_data):
    """Generates a conversational explanation of the query results using an LLM."""
    if not result_data or not result_data.get('rows'):
        return "I couldn't find any data matching your query. It's possible the table is empty or your criteria didn't match any records."

    headers = result_data['headers']
    rows = result_data['rows'][:3]
    preview = ", ".join(headers) + "\n"
    for row in rows:
        preview += ", ".join(map(str, row)) + "\n"

    response_prompt = f"""
You are a helpful Data Analyst Assistant. Your task is to provide a friendly, conversational summary of the provided data.

Original Question: "{question}"
SQL Query Used: "{sql_query}"
Data Preview:
{preview}

Please provide a concise, easy-to-understand summary of the results. Do not mention technical terms like 'SQL'.
Natural Language Response:"""
    
    try:
        natural_response = llm.invoke(response_prompt)
        return natural_response.content.strip()
    except Exception as e:
        print(f"Error generating natural response: {e}")
        return "Here is the data I found."

# --- Global Variables for Core Components ---
llm = None
db = None

def initialize_components():
    """Initializes the LLM and database connection from environment variables."""
    global llm, db
    try:
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            raise ValueError("GOOGLE_API_KEY not found in .env file")
        os.environ["GOOGLE_API_KEY"] = google_api_key
        
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1)
        
        db_user = os.getenv("DB_USER", "root")
        db_password = os.getenv("DB_PASSWORD", "")
        db_host = os.getenv("DB_HOST", "localhost")
        db_port = os.getenv("DB_PORT", "3306")
        db_name = os.getenv("DB_NAME", "your_database")
        
        db_uri = f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        print(f"Connecting to database: {db_user}@{db_host}:{db_port}/{db_name}")
        
        db = SQLDatabase.from_uri(db_uri)
        print("✅ Components initialized successfully!")

    except Exception as e:
        print(f"❌ Error initializing components: {e}")
        print("Please ensure your .env file is correctly configured and the database is running.")

# --- Flask Routes ---

@app.route('/')
def index():
    """Serves the main HTML page from the 'templates' folder."""
    return render_template('index.html')

@app.route('/api/query', methods=['POST'])
def process_query():
    """API endpoint to handle both SQL and conversational queries."""
    try:
        data = request.get_json()
        question = data.get('query')
        if not question:
            return jsonify({'error': 'No query provided in the request'}), 400
        
        if not all([llm, db]):
            return jsonify({'error': 'Backend components are not initialized. Check server logs.'}), 500

        # --- NEW: Intent Detection Step ---
        if not is_sql_query(llm, question):
            # Handle as a general conversational question
            print("Handling as a conversational query.")
            chat_response = llm.invoke(f"You are a helpful assistant. Answer the following question concisely: {question}").content
            return jsonify({
                'is_chat': True,
                'explanation': chat_response
            })

        # --- Existing Logic for SQL Queries ---
        print("Handling as a SQL query.")
        table_info = db.get_table_info()
        sql_prompt = f"""You are an expert MySQL developer. Given a user question and the database schema, generate a syntactically correct MySQL query.
Only output the SQL query. Do not include any markdown or explanatory text.

User Question: "{question}"
Database Schema:
{table_info}
SQL Query:"""
        
        sql_query_raw = llm.invoke(sql_prompt).content
        sql_query = clean_sql_query(sql_query_raw)
        print(f"Generated SQL: {sql_query}")
        
        structured_result = {"headers": [], "rows": []}
        with db._engine.connect() as connection:
            result_proxy = connection.execute(text(sql_query))
            if result_proxy.returns_rows:
                structured_result["headers"] = list(result_proxy.keys())
                structured_result["rows"] = [list(row) for row in result_proxy.fetchall()]
            else:
                structured_result["headers"] = ["status"]
                structured_result["rows"] = [[f"Query executed. Rows affected: {result_proxy.rowcount}"]]

        natural_answer = generate_natural_response(llm, question, sql_query, structured_result)
        
        return jsonify({
            'is_chat': False,
            'sql': sql_query,
            'results': structured_result,
            'explanation': natural_answer
        })
        
    except Exception as e:
        print(f"Error during query processing: {e}")
        print(traceback.format_exc())
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/api/health')
def health_check():
    """Health check endpoint for the frontend to verify API status."""
    is_initialized = bool(llm and db)
    status = {
        'status': 'healthy' if is_initialized else 'degraded',
        'components_initialized': is_initialized,
        'google_api_key_set': bool(os.getenv("GOOGLE_API_KEY"))
    }
    return jsonify(status)

# --- Main Execution ---
if __name__ == '__main__':
    print("🚀 Initializing backend components...")
    initialize_components()
    print("🚀 Starting Flask server on http://0.0.0.0:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)