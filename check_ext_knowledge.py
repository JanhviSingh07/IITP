import json, os

SPIDER2_BASE = "data/spider2/Spider2/spider2-lite"
QUESTIONS_FILE = f"{SPIDER2_BASE}/spider2-lite.jsonl"
GOLD_SQL_DIR = f"{SPIDER2_BASE}/evaluation_suite/gold/sql"
SQLITE_DB_DIR = f"{SPIDER2_BASE}/resource/databases/sqlite"
DOCS_DIR = f"{SPIDER2_BASE}/resource/documents"

# Load all questions
with open(QUESTIONS_FILE) as f:
    all_data = {json.loads(l)['instance_id']: json.loads(l) for l in f}

# Find evaluatable local questions
gold_sql_files = [f for f in os.listdir(GOLD_SQL_DIR) if f.startswith('local')]

print("Evaluatable questions with external knowledge:\n")
with_docs = []
without_docs = []

for sql_file in gold_sql_files:
    instance_id = sql_file.replace('.sql', '')
    item = all_data.get(instance_id, {})
    ext_knowledge = item.get('external_knowledge')

    if ext_knowledge:
        doc_path = os.path.join(DOCS_DIR, ext_knowledge)
        doc_exists = os.path.exists(doc_path)
        with_docs.append((instance_id, ext_knowledge, doc_exists))
        print(f"  {instance_id}: {ext_knowledge} (exists: {doc_exists})")
    else:
        without_docs.append(instance_id)

print(f"\nWith external knowledge: {len(with_docs)}")
print(f"Without external knowledge: {len(without_docs)}")
print(f"Without docs: {without_docs}")
