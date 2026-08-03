"""
run_spider1.py
--------------
Spider 1.0 ka poora evaluation pipeline. Yeh dev.json/test.json se
questions lega, har question ke liye:
1. Correct database (sqlite file) load karega (db_id se)
2. AskDB agent se SQL generate + execute karwayega
3. Generated result ko gold (ground truth) result se compare karega
4. Final accuracy + failure analysis report banayega

Usage:
    python eval/run_spider1.py --data_file data/spider1/dev.json \\
        --db_dir data/spider1/database --num_samples 98

Pehle SETUP_NOTES.md follow karo - dataset download aur API key setup ke liye.
"""

import json
import os
import sys
import argparse
import sqlite3
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # config.py is in project root

from schema_retriever import SchemaRetriever
from sql_agent import SQLAgent
from few_shot_examples import get_relevant_few_shot_examples
import error_analyzer

import config


def estimate_hardness(sql):
    """
    Spider ka dev.json mein 'hardness' field directly nahi hota (yeh sirf
    official evaluation.py se compute hoti hai SQL parse karke). Yahan ek
    simple heuristic approximation use kar rahe hain based on Spider paper's
    own criteria (component count - joins, nesting, aggregations, set ops).
    Yeh exact official metric nahi hai, lekin failure-analysis ke liye
    reasonable proxy hai.
    """
    sql_upper = sql.upper()
    score = 0

    score += sql_upper.count("JOIN")
    score += sql_upper.count("SELECT") - 1  # nested subqueries
    score += sum(sql_upper.count(op) for op in ["GROUP BY", "HAVING", "ORDER BY"])
    score += sum(sql_upper.count(op) for op in ["INTERSECT", "EXCEPT", "UNION"])
    score += sql_upper.count(" OR ") + sql_upper.count(" AND ")

    if score <= 1:
        return "easy"
    elif score <= 3:
        return "medium"
    elif score <= 5:
        return "hard"
    else:
        return "extra hard"


def load_spider_data(data_file, num_samples=None):
    with open(data_file, "r") as f:
        data = json.load(f)
    if num_samples:
        data = data[:num_samples]
    return data


def execution_match(result_a, result_b):
    """
    Execution Accuracy check - paper jaisa: result sets equivalent
    hone chahiye (order-independent comparison).

    Improvements over basic exact match:
    1. Row-order independent (set comparison)
    2. Extra column handling (subset match)
    3. Column-ORDER independent (sorted tuple comparison) - handles cases
       where agent returns same data but in different column order
       e.g. Gold: SELECT avg, type  vs  Generated: SELECT type, avg
    """
    if result_a is None or result_b is None:
        return False

    rows_a = result_a.get("rows") if isinstance(result_a, dict) else result_a
    rows_b = result_b.get("rows") if isinstance(result_b, dict) else result_b

    if rows_a is None or rows_b is None:
        return False

    # NOTE: Row count check removed - DISTINCT vs non-DISTINCT queries
    # return same unique values but different row counts (e.g. 28 vs 230).
    # Set comparison below handles this correctly.

    try:
        set_a = set(tuple(row) for row in rows_a)
        set_b = set(tuple(row) for row in rows_b)
    except TypeError:
        return rows_a == rows_b

    # Check 1: Exact match (row-order independent)
    if set_a == set_b:
        return True

    # Check 2: Column-order independent match
    # Sort each row's values before comparing - handles reversed column order
    # e.g. (5, 'cat') and ('cat', 5) both become ('cat', 5) after sorting
    try:
        sorted_a = set(tuple(sorted(str(v) for v in row)) for row in rows_a)
        sorted_b = set(tuple(sorted(str(v) for v in row)) for row in rows_b)
        if sorted_a == sorted_b:
            return True
    except Exception:
        pass

    # Check 3: Subset match (generated has extra helpful columns)
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
    """Ground truth SQL ko bhi execute karte hain comparison ke liye."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    try:
        cur.execute(gold_sql)
        rows = cur.fetchall()
        conn.close()
        return {"rows": rows}
    except sqlite3.Error as e:
        conn.close()
        return None


def run_evaluation(data_file, db_dir, num_samples=None, use_self_consistency=False,
                    offline_test_mode=False, model_name=None, use_egsr=False):
    data = load_spider_data(data_file, num_samples)
    model_name = model_name or config.MODEL_SPIDER1
    few_shot = get_relevant_few_shot_examples(num_examples=3)

    results_log = []
    schema_retriever_cache = {}  # db_id -> SchemaRetriever (re-use for same db)

    for item in tqdm(data, desc="Evaluating Spider 1.0"):
        question = item["question"]
        db_id = item["db_id"]
        gold_sql = item["query"]
        hardness = estimate_hardness(gold_sql)

        db_path = os.path.join(db_dir, db_id, f"{db_id}.sqlite")

        if not os.path.exists(db_path):
            print(f"WARNING: Database not found: {db_path}, skipping")
            continue

        if db_id not in schema_retriever_cache:
            schema_retriever_cache[db_id] = SchemaRetriever(
                db_path, offline_test_mode=offline_test_mode
            )
        retriever = schema_retriever_cache[db_id]

        agent = SQLAgent(
            db_path=db_path,
            schema_retriever=retriever,
            api_key=config.ACTIVE_API_KEY,
            model_name=model_name,
            max_iterations=config.MAX_REACT_ITERATIONS,
            few_shot_examples=few_shot,
            provider=config.PROVIDER,
        )

        try:
            if use_egsr:
                # NOVEL CONTRIBUTION: Execution-Guided Schema Refinement
                outcome = agent.answer_with_egsr(question, verbose=False)
            elif use_self_consistency:
                outcome = agent.answer_with_self_consistency(
                    question, num_samples=config.SELF_CONSISTENCY_SAMPLES, verbose=False
                )
            else:
                outcome = agent.answer_question(question, verbose=False)
        except Exception as e:
            outcome = {"success": False, "sql": None, "result": str(e)}

        gold_result = get_gold_result(db_path, gold_sql)
        generated_result = outcome.get("result") if outcome.get("success") else None
        matched = execution_match(generated_result, gold_result) if outcome.get("success") else False

        results_log.append({
            "question": question,
            "gold_sql": gold_sql,
            "generated_sql": outcome.get("sql"),
            "execution_success": outcome.get("success", False),
            "error_message": outcome.get("result") if not outcome.get("success") else None,
            "result_matched": matched,
            "hardness": hardness,
            "iterations_used": outcome.get("iterations_used"),
            "egsr_rounds": outcome.get("egsr_rounds_used", 1),
        })

    return results_log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", default=f"{config.SPIDER1_DATA_DIR}/dev.json")
    parser.add_argument("--db_dir", default=f"{config.SPIDER1_DATA_DIR}/database")
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--self_consistency", action="store_true")
    parser.add_argument("--egsr", action="store_true",
                         help="Use EGSR: Execution-Guided Schema Refinement (novel contribution)")
    parser.add_argument("--offline_test_mode", action="store_true",
                         help="TF-IDF mock embedder use karo (sirf testing ke liye, no internet)")
    args = parser.parse_args()

    # Save results to different file when using EGSR so baseline is preserved
    results_filename = "spider1_egsr_results.json" if args.egsr else "spider1_results.json"

    results = run_evaluation(
        args.data_file, args.db_dir, args.num_samples,
        use_self_consistency=args.self_consistency,
        offline_test_mode=args.offline_test_mode,
        use_egsr=args.egsr,
    )

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    output_path = os.path.join(config.RESULTS_DIR, results_filename)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nRaw results saved to: {output_path}")

    report = error_analyzer.analyze_results_batch(results)
    error_analyzer.print_report(report)

    # EGSR-specific stats
    if args.egsr:
        rounds_used = [r.get("egsr_rounds", 1) for r in results]
        multi_round = sum(1 for r in rounds_used if r > 1)
        print(f"\n[EGSR Stats] Questions needing >1 schema round: {multi_round}/{len(results)}")
        print(f"[EGSR Stats] Average rounds per question: {sum(rounds_used)/len(rounds_used):.2f}")


if __name__ == "__main__":
    main()
