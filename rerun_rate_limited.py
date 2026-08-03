"""
rerun_rate_limited.py
----------------------
Pichle run mein jo questions sirf RATE LIMIT (429) ki wajah se fail hue
the (genuine SQL error nahi the), unko dobara try karta hai aur results
file ko update karta hai - taaki poora 98-question result clean ho,
bina poora dataset dobara chalaye.
"""

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, os.path.dirname(__file__))

from schema_retriever import SchemaRetriever
from sql_agent import SQLAgent
from few_shot_examples import get_relevant_few_shot_examples
import config
import sqlite3


def execution_match(result_a, result_b):
    if result_a is None or result_b is None:
        return False
    rows_a = result_a.get("rows") if isinstance(result_a, dict) else result_a
    rows_b = result_b.get("rows") if isinstance(result_b, dict) else result_b
    if rows_a is None or rows_b is None:
        return False
    if len(rows_a) != len(rows_b):
        return False
    try:
        set_a = set(tuple(row) for row in rows_a)
        set_b = set(tuple(row) for row in rows_b)
    except TypeError:
        return rows_a == rows_b
    if set_a == set_b:
        return True
    # Column-order independent check
    try:
        sorted_a = set(tuple(sorted(str(v) for v in row)) for row in rows_a)
        sorted_b = set(tuple(sorted(str(v) for v in row)) for row in rows_b)
        if sorted_a == sorted_b:
            return True
    except Exception:
        pass
    # Subset check
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


# Load original Spider data to get db_id for each question
with open(f"{config.SPIDER1_DATA_DIR}/dev.json", "r") as f:
    full_data = json.load(f)

# Load previous results
with open("results/spider1_results.json", "r") as f:
    results = json.load(f)

# Find rate-limited entries (by matching question text)
rate_limited_indices = [
    i for i, r in enumerate(results)
    if r.get("error_message") and "429" in str(r.get("error_message"))
]

print(f"Found {len(rate_limited_indices)} rate-limited questions to retry.\n")

few_shot = get_relevant_few_shot_examples(num_examples=3)
schema_cache = {}

for idx in rate_limited_indices:
    question_text = results[idx]["question"]

    # Find matching item in full_data to get db_id
    matching_item = next((item for item in full_data if item["question"] == question_text), None)
    if not matching_item:
        print(f"Could not find db_id for: {question_text}, skipping")
        continue

    db_id = matching_item["db_id"]
    gold_sql = matching_item["query"]
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

    print(f"Retrying: {question_text}")
    try:
        outcome = agent.answer_question(question_text, verbose=False)
    except Exception as e:
        outcome = {"success": False, "sql": None, "result": str(e)}

    gold_result = get_gold_result(db_path, gold_sql)
    generated_result = outcome.get("result") if outcome.get("success") else None
    matched = execution_match(generated_result, gold_result) if outcome.get("success") else False

    results[idx] = {
        "question": question_text,
        "gold_sql": gold_sql,
        "generated_sql": outcome.get("sql"),
        "execution_success": outcome.get("success", False),
        "error_message": outcome.get("result") if not outcome.get("success") else None,
        "result_matched": matched,
        "hardness": results[idx]["hardness"],
        "iterations_used": outcome.get("iterations_used"),
    }
    print(f"  -> {'MATCHED' if matched else 'NOT MATCHED'}\n")

# Save updated results
with open("results/spider1_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

# Final accuracy
total_matched = sum(1 for r in results if r.get("result_matched"))
print(f"\n=== UPDATED FINAL ACCURACY: {total_matched}/{len(results)} = {round(total_matched/len(results)*100, 2)}% ===")
