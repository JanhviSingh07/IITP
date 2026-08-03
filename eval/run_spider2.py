"""
run_spider2.py
--------------
Spider 2.0-lite evaluation - sirf local SQLite questions (21-24 questions)
Gold SQL .sql files se load hota hai.

Usage:
    python eval/run_spider2.py
    python eval/run_spider2.py --egsr
"""

import json, os, sys, argparse, sqlite3
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from schema_retriever import SchemaRetriever
from sql_agent import SQLAgent
from few_shot_examples import get_relevant_few_shot_examples
import error_analyzer
import config

SPIDER2_BASE = "data/spider2/Spider2/spider2-lite"
QUESTIONS_FILE = f"{SPIDER2_BASE}/spider2-lite.jsonl"
GOLD_SQL_DIR = f"{SPIDER2_BASE}/evaluation_suite/gold/sql"
SQLITE_DB_DIR = f"{SPIDER2_BASE}/resource/databases/sqlite"


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
    if set_a == set_b:
        return True
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


def find_sqlite_db(db_name):
    """Case-insensitive DB folder search."""
    for folder in os.listdir(SQLITE_DB_DIR):
        if folder.lower() == db_name.lower():
            db_folder = os.path.join(SQLITE_DB_DIR, folder)
            for f in os.listdir(db_folder):
                if f.endswith('.sqlite') or f.endswith('.db'):
                    return os.path.join(db_folder, f)
    return None


def load_evaluatable_questions(num_samples=None):
    with open(QUESTIONS_FILE) as f:
        all_data = {json.loads(l)['instance_id']: json.loads(l) for l in f}

    gold_sql_files = [f for f in os.listdir(GOLD_SQL_DIR) if f.startswith('local')]
    evaluatable = []

    for sql_file in gold_sql_files:
        instance_id = sql_file.replace('.sql', '')
        item = all_data.get(instance_id)
        if not item:
            continue
        db_path = find_sqlite_db(item['db'])
        if not db_path:
            continue
        with open(os.path.join(GOLD_SQL_DIR, sql_file)) as f:
            gold_sql = f.read().strip()
        evaluatable.append({
            'instance_id': instance_id,
            'question': item['question'],
            'db': item['db'],
            'db_path': db_path,
            'gold_sql': gold_sql,
        })

    print(f"Found {len(evaluatable)} evaluatable local SQLite questions")
    if num_samples:
        evaluatable = evaluatable[:num_samples]
    return evaluatable


def get_gold_result(db_path, gold_sql):
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


def estimate_hardness(sql):
    sql_upper = sql.upper()
    score = 0
    score += sql_upper.count("JOIN")
    score += sql_upper.count("SELECT") - 1
    score += sum(sql_upper.count(op) for op in ["GROUP BY", "HAVING", "ORDER BY"])
    score += sum(sql_upper.count(op) for op in ["INTERSECT", "EXCEPT", "UNION"])
    score += sql_upper.count("WITH ")
    if score <= 1:
        return "easy"
    elif score <= 3:
        return "medium"
    elif score <= 6:
        return "hard"
    else:
        return "extra hard"


def run_evaluation(num_samples=None, use_egsr=False, use_self_consistency=False):
    data = load_evaluatable_questions(num_samples)
    few_shot = get_relevant_few_shot_examples(num_examples=3)
    schema_cache = {}
    results_log = []

    for item in tqdm(data, desc="Evaluating Spider 2.0"):
        question = item['question']
        db_path = item['db_path']
        gold_sql = item['gold_sql']
        db_id = item['db']

        if db_id not in schema_cache:
            try:
                schema_cache[db_id] = SchemaRetriever(db_path)
            except Exception as e:
                print(f"Schema error for {db_id}: {e}")
                continue

        retriever = schema_cache[db_id]
        agent = SQLAgent(
            db_path=db_path,
            schema_retriever=retriever,
            api_key=config.ACTIVE_API_KEY,
            model_name=config.MODEL_SPIDER2,
            max_iterations=config.MAX_REACT_ITERATIONS,
            few_shot_examples=few_shot,
            provider=config.PROVIDER,
        )

        try:
            if use_egsr:
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
            "instance_id": item['instance_id'],
            "question": question,
            "db": db_id,
            "gold_sql": gold_sql[:300],
            "generated_sql": outcome.get("sql"),
            "execution_success": outcome.get("success", False),
            "error_message": str(outcome.get("result", ""))[:200] if not outcome.get("success") else None,
            "result_matched": matched,
            "hardness": estimate_hardness(gold_sql),
            "egsr_rounds": outcome.get("egsr_rounds_used", 1),
        })

    return results_log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--egsr", action="store_true")
    parser.add_argument("--self_consistency", action="store_true")
    args = parser.parse_args()

    suffix = "_egsr" if args.egsr else "_baseline"
    results_filename = f"spider2{suffix}_results.json"

    results = run_evaluation(
        num_samples=args.num_samples,
        use_egsr=args.egsr,
        use_self_consistency=args.self_consistency,
    )

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    output_path = os.path.join(config.RESULTS_DIR, results_filename)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

    report = error_analyzer.analyze_results_batch(results)
    error_analyzer.print_report(report)

    if args.egsr and results:
        rounds = [r.get("egsr_rounds", 1) for r in results]
        multi = sum(1 for r in rounds if r > 1)
        print(f"\n[EGSR Stats] Questions needing >1 schema round: {multi}/{len(results)}")
        print(f"[EGSR Stats] Average rounds: {sum(rounds)/len(rounds):.2f}")


if __name__ == "__main__":
    main()