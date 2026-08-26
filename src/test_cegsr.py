import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))  # project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # src folder
from schema_retriever import SchemaRetriever
from sql_agent import SQLAgent
import config

db_path = 'data/spider2/Spider2/spider2-lite/resource/databases/sqlite/local003/local003.sqlite'
retriever = SchemaRetriever(db_path)
agent = SQLAgent(
    db_path=db_path,
    schema_retriever=retriever,
    api_key=config.ACTIVE_API_KEY,
    model_name=config.MODEL_SPIDER2,
    provider=config.PROVIDER,
)
result = agent.answer_with_cegsr('list all records', verbose=True)
print(result)