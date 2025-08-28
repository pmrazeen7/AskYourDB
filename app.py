# app.py
# Import necessary libraries
from flask import Flask, request, jsonify, render_template, session
from flask_cors import CORS
import re
import traceback
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine, text

# Load environment variables from a .env file
load_dotenv()

# Initialize the Flask application
# This now uses the standard 'templates' folder, which is best practice.
app = Flask(__name__, template_folder='templates') 
CORS(app, supports_credentials=True) # supports_credentials is required for sessions

# A secret key is mandatory for Flask sessions to work securely
app.secret_key = os.getenv("FLASK_SECRET_KEY", os.urandom(24))

# --- Global Variable for the LLM ---
llm = None

def initialize_llm():
    """Initializes the Language Model from environment variables."""
    global llm
    try:
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            raise ValueError("GOOGLE_API_KEY not found in .env file")
        os.environ["GOOGLE_API_KEY"] = google_api_key
        
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1)
        print("✅ LLM initialized successfully!")

    except Exception as e:
        print(f"❌ Error initializing LLM: {e}")
        print("Please ensure your GOOGLE_API_KEY is correctly configured in the .env file.")

# --- Helper Functions ---

def clean_sql_query(query):
    """Removes markdown formatting from the SQL query."""
    query = re.sub(r"```sql\n?", "", query)
    query = re.sub(r"```\n?", "", query)
    return query.strip()

def is_sql_query(question):
    """Uses the LLM to determine if a question is asking for data."""
    if not llm: return True # Default to SQL if LLM is not available
    prompt = f"""Is the following question asking for data from a database, or is it a general conversational question (like a greeting)?
Respond with only 'Yes' if it is asking for data, or 'No' if it is conversational.
Question: "{question}"
Answer:"""
    try:
        response = llm.invoke(prompt).content.strip().lower()
        print(f"Intent check for '{question}': AI responded '{response}'")
        return "yes" in response
    except Exception as e:
        print(f"Error during intent check: {e}")
        return True

def generate_natural_response(question, sql_query, result_data):
    """Generates a conversational explanation of the query results."""
    if not llm: return "Here is the data I found."
    if not result_data or not result_data.get('rows'):
        return "I couldn't find any data matching your query. It's possible the table is empty or your criteria didn't match any records."

    headers = result_data['headers']
    rows = result_data['rows'][:3] # Preview first 3 rows
    preview = ", ".join(headers) + "\n"
    for row in rows:
        preview += ", ".join(map(str, row)) + "\n"

    response_prompt = f"""You are a helpful Data Analyst Assistant. Provide a friendly, conversational summary of the provided data.
Original Question: "{question}"
SQL Query Used: "{sql_query}"
Data Preview (first 3 rows):
{preview}
Please provide a concise, easy-to-understand summary. Do not mention technical terms like 'SQL'.
Natural Language Response:"""
    try:
        natural_response = llm.invoke(response_prompt)
        return natural_response.content.strip()
    except Exception as e:
        print(f"Error generating natural response: {e}")
        return "Here is the data I found."

# --- Flask Routes ---

@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/api/connect', methods=['POST'])
def connect_db():
    """Establishes and tests a database connection, saving it to the session."""
    data = request.get_json()
    db_user = data.get("user", "")
    db_password = data.get("password", "")
    db_host = data.get("host", "localhost")
    db_port = data.get("port", "3306")
    db_name = data.get("name", "")

    if not db_name:
        return jsonify({'status': 'error', 'message': 'Database name is required.'}), 400

    db_uri = f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    
    try:
        print(f"Attempting to connect to: {db_user}@{db_host}/{db_name}")
        engine = create_engine(db_uri)
        with engine.connect() as connection:
            connection.execute(text("SELECT 1")) # Simple query to test connection
        
        session['db_uri'] = db_uri
        session['db_info'] = f"{db_user}@{db_host}/{db_name}"
        
        print(f"✅ Connection successful. DB URI stored in session.")
        
        return jsonify({'status': 'success', 'message': f"Successfully connected to {session['db_info']}"})
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return jsonify({'status': 'error', 'message': f"Connection failed: {str(e)}"}), 400

@app.route('/api/status', methods=['GET'])
def connection_status():
    """Checks if a database connection exists for the current session."""
    if 'db_uri' in session:
        return jsonify({'connected': True, 'db_info': session.get('db_info')})
    else:
        return jsonify({'connected': False})

@app.route('/api/disconnect', methods=['POST'])
def disconnect_db():
    """Removes database connection details from the session."""
    session.pop('db_uri', None)
    session.pop('db_info', None)
    return jsonify({'status': 'success', 'message': 'Disconnected.'})

@app.route('/api/query', methods=['POST'])
def process_query():
    """API endpoint to handle queries using the session-specific database."""
    if 'db_uri' not in session:
        return jsonify({'error': 'Not connected to a database. Please connect first.'}), 403
    
    if not llm:
        return jsonify({'error': 'Backend LLM is not initialized. Check server logs.'}), 500

    try:
        db = SQLDatabase.from_uri(session['db_uri'])
        data = request.get_json()
        question = data.get('query')
        if not question:
            return jsonify({'error': 'No query provided'}), 400
        
        if not is_sql_query(question):
            chat_response = llm.invoke(f"You are a helpful assistant. Answer concisely: {question}").content
            return jsonify({'is_chat': True, 'explanation': chat_response})

        table_info = db.get_table_info()
        sql_prompt = f"""You are an expert MySQL developer. Given a user question and schema, generate a correct MySQL query.
Only output the SQL query. No markdown or text.
Question: "{question}"
Schema: {table_info}
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

        natural_answer = generate_natural_response(question, sql_query, structured_result)
        
        return jsonify({
            'is_chat': False,
            'sql': sql_query,
            'results': structured_result,
            'explanation': natural_answer
        })
        
    except Exception as e:
        print(f"Error during query processing: {e}\n{traceback.format_exc()}")
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/api/health')
def health_check():
    """Health check for the LLM component."""
    return jsonify({'status': 'healthy' if llm else 'degraded'})

# --- Main Execution ---
if __name__ == '__main__':
    print("🚀 Initializing LLM component...")
    initialize_llm()
    print("🚀 Starting Flask server on http://127.0.0.1:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)