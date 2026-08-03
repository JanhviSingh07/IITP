"""
retry_failed.py
----------------
Currently failed cases ko naye improved prompt ke saath retry karta hai.
Sirf failed questions pe LLM call karta hai - poora 98 dobara nahi.
"""

import sys, os, json, sqlite3
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, os.path.dirname(__file__))

from schema_retriever import SchemaRetriever
from sql_agent import SQLAgent
from few_shot_examples import get_relevant_few_shot_examples
import config


def execution_match(result_a, result_b):
    if result_a is None or result_b is None:
        return False
    rows_a = result_a.get("rows") if isinstance(result_a, dict) else result_a
    rows_b = result_b.get("rows") if isinstance(result_b, dict) else result_b
    if rows_a is None or rows_b is None:
        return False
    try:
        set_a = set(tuple(row) for row in rows_a)
        set_b = set(tuple(row) for row in rows_b)
    except TypeError:
        return rows_a == rows_b
    # Set match - handles DISTINCT vs non-DISTINCT
    if set_a == set_b:
        return True
    # Subset check - handles extra helpful columns
    if rows_a and rows_b:
        len_a = len(rows_a[0]) if rows_a[0] else 0
        len_b = len(rows_b[0]) if rows_b[0] else 0
        min_len = min(len_a, len_b)
        if min_len > 0 and len_a != len_b:
            trimmed_a = set(tuple(row[:min_len]) for row in rows_a)
            trimmed_b = set(tuple(row[:min_len]) for row in rows_b)
            if trimmed_a == trimmed_b:
                return True
    return False


def get_gold_result(db_path, gold_sql):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    try:
        cur.execute(gold_sql)
        rows = cur.fetchall()
        conn.close()
        return {"rows": rows}
    except sqlite3.Error:
        conn.close()
        return None


with open(f"{config.SPIDER1_DATA_DIR}/dev.json") as f:
    full_data = json.load(f)
q_to_item = {item["question"]: item for item in full_data}

with open("results/spider1_results.json") as f:
    results = json.load(f)

failed_indices = [i for i, r in enumerate(results) if not r.get("result_matched")]
print(f"Found {len(failed_indices)} failed cases to retry.\n")

few_shot = get_relevant_few_shot_examples(num_examples=3)
schema_cache = {}

for idx in failed_indices:
    r = results[idx]
    question = r["question"]
    item = q_to_item.get(question)
    if not item:
        print(f"Skipping (not found in dev.json): {question[:60]}")
        continue

    db_id = item["db_id"]
    gold_sql = item["query"]
    db_path = os.path.join(config.SPIDER1_DATA_DIR, "database", db_id, f"{db_id}.sqlite")

    if db_id not in schema_cache:
        schema_cache[db_id] = SchemaRetriever(db_path)
    retriever = schema_cache[db_id]

    agent = SQLAgent(
        db_path=db_path,
        schema_retriever=retriever,
        api_key=config.ACTIVE_API_KEY,
        model_name=config.MODEL_SPIDER1,
        max_iterations=config.MAX_REACT_ITERATIONS,
        few_shot_examples=few_shot,
        provider=config.PROVIDER,
    )

    print(f"Retrying: {question[:70]}")
    try:
        outcome = agent.answer_question(question, verbose=False)
    except Exception as e:
        outcome = {"success": False, "sql": None, "result": str(e)}

    gold_result = get_gold_result(db_path, gold_sql)
    gen_result = outcome.get("result") if outcome.get("success") else None
    matched = execution_match(gen_result, gold_result) if outcome.get("success") else False

    results[idx].update({
        "generated_sql": outcome.get("sql"),
        "execution_success": outcome.get("success", False),
        "error_message": outcome.get("result") if not outcome.get("success") else None,
        "result_matched": matched,
        "iterations_used": outcome.get("iterations_used"),
    })
    print(f"  -> {'✓ MATCHED' if matched else '✗ NOT MATCHED'}\n")

with open("results/spider1_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

total = sum(1 for r in results if r.get("result_matched"))
print(f"=== FINAL ACCURACY: {total}/{len(results)} = {round(total/len(results)*100, 2)}% ===")
