import sys, os
sys.path.insert(0, 'src')
sys.path.insert(0, '.')

from schema_retriever import SchemaRetriever
from sql_agent import SQLAgent
from few_shot_examples import get_relevant_few_shot_examples
import config

SQLITE_DB_DIR = "data/spider2/Spider2/spider2-lite/resource/databases/sqlite"

def find_sqlite_db(db_name):
    for folder in os.listdir(SQLITE_DB_DIR):
        if folder.lower() == db_name.lower():
            db_folder = os.path.join(SQLITE_DB_DIR, folder)
            for f in os.listdir(db_folder):
                if f.endswith('.sqlite') or f.endswith('.db'):
                    return os.path.join(db_folder, f)
    return None

db_path = find_sqlite_db("E_commerce")
retriever = SchemaRetriever(db_path)
agent = SQLAgent(
    db_path=db_path,
    schema_retriever=retriever,
    api_key=config.ACTIVE_API_KEY,
    model_name=config.MODEL_SPIDER2,
    max_iterations=3,
    provider=config.PROVIDER,
)

question = "What is the total number of orders?"
schema_context, _ = retriever.get_scoped_context(question, top_k=3)

prompt = f"""You are an expert SQL agent. Given a database schema and question,
generate a SQL query AND rate your confidence.

Database Schema:
{schema_context}

Question: {question}

Respond in EXACTLY this format (no extra text):
CONFIDENCE: <number between 0.0 and 1.0>
SQL: <your sql query here>"""

raw = agent._call_llm(prompt)
print("=== RAW LLM RESPONSE ===")
print(raw)
print("========================")
