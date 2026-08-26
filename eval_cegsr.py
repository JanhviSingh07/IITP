"""
eval_cegsr.py
--------------
Baseline vs EGSR vs C-EGSR comparison on Spider 1.0 and Spider 2.0.

Run:
    python eval_cegsr.py --benchmark spider1 --num_samples 98
    python eval_cegsr.py --benchmark spider2
"""

import os, sys, json, sqlite3, argparse
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, os.path.dirname(__file__))

from schema_retriever import SchemaRetriever
from sql_agent import SQLAgent
from few_shot_examples import get_relevant_few_shot_examples
import config

# Paths
SPIDER1_DATA = "data/spider1/dev.json"
SPIDER1_DB_DIR = "data/spider1/database"
SPIDER2_BASE = "data/spider2/Spider2/spider2-lite"
SPIDER2_QUESTIONS = f"{SPIDER2_BASE}/spider2-lite.jsonl"
SPIDER2_GOLD_DIR = f"{SPIDER2_BASE}/evaluation_suite/gold/sql"
SPIDER2_DB_DIR = f"{SPIDER2_BASE}/resource/databases/sqlite"


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
        return set_a == set_b
    except:
        return False


def get_gold_result(db_path, gold_sql):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    try:
        cur.execute(gold_sql)
        rows = cur.fetchall()
        conn.close()
        return {"rows": rows}
    except:
        conn.close()
        return None


def find_spider2_db(db_name):
    for folder in os.listdir(SPIDER2_DB_DIR):
        if folder.lower() == db_name.lower():
            db_folder = os.path.join(SPIDER2_DB_DIR, folder)
            for f in os.listdir(db_folder):
                if f.endswith('.sqlite') or f.endswith('.db'):
                    return os.path.join(db_folder, f)
    return None


def load_spider1(num_samples=None):
    with open(SPIDER1_DATA) as f:
        data = json.load(f)
    if num_samples:
        data = data[:num_samples]
    return [{"question": d["question"], "db_id": d["db_id"],
             "gold_sql": d["query"],
             "db_path": os.path.join(SPIDER1_DB_DIR, d["db_id"], f"{d['db_id']}.sqlite")}
            for d in data if os.path.exists(
                os.path.join(SPIDER1_DB_DIR, d["db_id"], f"{d['db_id']}.sqlite"))]


def load_spider2():
    with open(SPIDER2_QUESTIONS) as f:
        all_data = {json.loads(l)['instance_id']: json.loads(l) for l in f}
    gold_files = [f for f in os.listdir(SPIDER2_GOLD_DIR) if f.startswith('local')]
    data = []
    for sql_file in gold_files:
        instance_id = sql_file.replace('.sql', '')
        item = all_data.get(instance_id)
        if not item:
            continue
        db_path = find_spider2_db(item['db'])
        if not db_path:
            continue
        with open(os.path.join(SPIDER2_GOLD_DIR, sql_file)) as f:
            gold_sql = f.read().strip()
        data.append({"question": item['question'], "db_id": item['db'],
                     "gold_sql": gold_sql, "db_path": db_path})
    return data


def run_method(data, method, model_name, few_shot, desc=""):
    schema_cache = {}
    results = []

    for item in tqdm(data, desc=f"{desc} [{method}]"):
        question = item['question']
        db_path = item['db_path']
        db_id = item['db_id']
        gold_sql = item['gold_sql']

        if db_id not in schema_cache:
            try:
                schema_cache[db_id] = SchemaRetriever(db_path)
            except:
                continue

        agent = SQLAgent(
            db_path=db_path,
            schema_retriever=schema_cache[db_id],
            api_key=config.ACTIVE_API_KEY,
            model_name=model_name,
            max_iterations=config.MAX_REACT_ITERATIONS,
            few_shot_examples=few_shot,
            provider=config.PROVIDER,
        )

        try:
            if method == "baseline":
                outcome = agent.answer_question(question, verbose=False)
            elif method == "egsr":
                outcome = agent.answer_with_egsr(question, verbose=False)
            elif method == "cegsr":
                outcome = agent.answer_with_cegsr(question, verbose=False)
        except Exception as e:
            outcome = {"success": False, "sql": None, "result": str(e)}

        gold_result = get_gold_result(db_path, gold_sql)
        gen_result = outcome.get("result") if outcome.get("success") else None
        matched = execution_match(gen_result, gold_result) if outcome.get("success") else False

        results.append({
            "question": question,
            "db_id": db_id,
            "matched": matched,
            "success": outcome.get("success", False),
            "error": str(outcome.get("result", ""))[:100] if not outcome.get("success") else None,
            "cegsr_expanded": outcome.get("cegsr_used_expanded"),
            "cegsr_top_k": outcome.get("cegsr_top_k"),
        })

    return results


def summarize(results, method):
    total = len(results)
    rate_limited = sum(1 for r in results if r.get("error") and "429" in str(r.get("error", "")))
    correct = sum(1 for r in results if r["matched"])
    properly_eval = total - rate_limited
    real_acc = round(correct / properly_eval * 100, 2) if properly_eval > 0 else 0

    print(f"\n  {method}:")
    print(f"    Total: {total}, Rate-limited: {rate_limited}, Properly evaluated: {properly_eval}")
    print(f"    Correct: {correct}, Raw accuracy: {round(correct/total*100,2)}%")
    print(f"    Real accuracy (excl. rate-limited): {correct}/{properly_eval} = {real_acc}%")

    if method == "cegsr":
        expanded = sum(1 for r in results if r.get("cegsr_expanded"))
        print(f"    C-EGSR expanded schema: {expanded}/{total} ({round(expanded/total*100,1)}%)")

    return real_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=["spider1", "spider2"], default="spider1")
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--methods", nargs="+",
                        choices=["baseline", "egsr", "cegsr"],
                        default=["baseline", "egsr", "cegsr"])
    args = parser.parse_args()

    few_shot = get_relevant_few_shot_examples(num_examples=3)

    if args.benchmark == "spider1":
        data = load_spider1(args.num_samples)
        model = config.MODEL_SPIDER1
        desc = "Spider 1.0"
    else:
        data = load_spider2()
        if args.num_samples:
            data = data[:args.num_samples]
        model = config.MODEL_SPIDER2
        desc = "Spider 2.0"

    print(f"\n{'='*60}")
    print(f"Benchmark: {desc} | Questions: {len(data)} | Methods: {args.methods}")
    print(f"Model: {model}")
    print(f"{'='*60}")

    all_results = {}
    for method in args.methods:
        results = run_method(data, method, model, few_shot, desc)
        all_results[method] = results
        os.makedirs("results", exist_ok=True)
        with open(f"results/cegsr_{args.benchmark}_{method}.json", "w") as f:
            json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print("FINAL COMPARISON")
    print(f"{'='*60}")
    accs = {}
    for method in args.methods:
        accs[method] = summarize(all_results[method], method)

    print(f"\n{'='*60}")
    print("SUMMARY TABLE")
    print(f"{'='*60}")
    print(f"{'Method':<12} {'Accuracy':>10}")
    for method, acc in accs.items():
        print(f"{method:<12} {acc:>9.2f}%")


if __name__ == "__main__":
    main()
