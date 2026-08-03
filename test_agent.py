import sys
sys.path.insert(0, 'src')

from schema_retriever import SchemaRetriever
from sql_agent import SQLAgent
import config

print("Step 1: Schema retriever banate hain...")
retriever = SchemaRetriever('data/sample_dbs/company_db.sqlite')

print("Step 2: Agent banate hain (Groq provider)...")
agent = SQLAgent(
    db_path='data/sample_dbs/company_db.sqlite',
    schema_retriever=retriever,
    api_key=config.GROQ_API_KEY,
    model_name=config.MODEL_SPIDER1,
    provider="groq",
)

print("Step 3: Question poochte hain...")
question = (
    "List the names of employees who are leading a project with budget "
    "greater than 300000, along with their department name."
)
result = agent.answer_question(question, verbose=True)

print("\n=== FINAL RESULT ===")
print(result)