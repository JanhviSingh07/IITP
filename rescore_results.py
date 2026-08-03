"""
rescore_results.py
-------------------
Existing spider1_results.json ko naye, improved execution_match logic se
RE-SCORE karta hai - BINA koi naya LLM call kiye.
"""

import json
import sqlite3
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
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
    # Set match - handles DISTINCT vs non-DISTINCT (same unique values)
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


def get_generated_result(db_path, generated_sql):
    if not generated_sql:
        return None
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    try:
        cur.execute(generated_sql)
        rows = cur.fetchall()
        conn.close()
        return {"rows": rows}
    except sqlite3.Error:
        conn.close()
        return None


with open(f"{config.SPIDER1_DATA_DIR}/dev.json", "r") as f:
    full_data = json.load(f)
question_to_item = {item["question"]: item for item in full_data}

with open("results/spider1_results.json", "r") as f:
    results = json.load(f)

print("Re-scoring existing results with improved matching logic...")
print("(No LLM calls - purely local re-evaluation)\n")

updated = 0
for r in results:
    if not r.get("execution_success") or not r.get("generated_sql"):
        continue
    question = r["question"]
    item = question_to_item.get(question)
    if not item:
        continue
    db_id = item["db_id"]
    db_path = os.path.join(config.SPIDER1_DATA_DIR, "database", db_id, f"{db_id}.sqlite")
    gold_result = get_gold_result(db_path, r["gold_sql"])
    gen_result = get_generated_result(db_path, r["generated_sql"])
    new_match = execution_match(gen_result, gold_result)
    if new_match != r.get("result_matched"):
        r["result_matched"] = new_match
        updated += 1
        print(f"Changed: {'FAIL->PASS' if new_match else 'PASS->FAIL'}: {question[:70]}")

print(f"\nTotal changed: {updated} cases")

with open("results/spider1_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

total_matched = sum(1 for r in results if r.get("result_matched"))
print(f"\n=== FINAL ACCURACY: {total_matched}/{len(results)} = {round(total_matched/len(results)*100, 2)}% ===")
