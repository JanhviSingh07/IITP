"""
collect_signals_spider1.py
---------------------------
Spider 1.0 ke liye signals collect karta hai.
Spider 2.0 ke signal_data.json ke saath combine karke
cost model ka training data banayega.
"""

import os, sys, json, sqlite3, re
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, os.path.dirname(__file__))

from schema_retriever import SchemaRetriever
from sql_agent import SQLAgent
from few_shot_examples import get_relevant_few_shot_examples
import config

SPIDER1_DATA_FILE = "data/spider1/dev.json"
SPIDER1_DB_DIR = "data/spider1/database"


def find_sqlite_db(db_dir, db_id):
    path = os.path.join(db_dir, db_id, f"{db_id}.sqlite")
    return path if os.path.exists(path) else None


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


def compute_sql_complexity(sql):
    sql_upper = sql.upper() if sql else ""
    score = 0
    score += sql_upper.count("JOIN")
    score += sql_upper.count("SELECT") - 1
    score += sql_upper.count("WITH ")
    score += sql_upper.count("GROUP BY")
    score += sql_upper.count("HAVING")
    score += sql_upper.count("ORDER BY")
    score += sql_upper.count("INTERSECT") + sql_upper.count("EXCEPT") + sql_upper.count("UNION")
    return max(0, score)


def get_llm_confidence(agent, question, schema_context):
    prompt = f"""You are an expert SQL agent. Given a database schema and question,
generate a SQL query AND rate your confidence.

Database Schema:
{schema_context}

Question: {question}

Respond in EXACTLY this format:
CONFIDENCE: <number between 0.0 and 1.0>
SQL: <your sql query here>"""

    try:
        raw = agent._call_llm(prompt)
        lines = raw.strip().split('\n')
        confidence = 0.5
        sql_lines = []
        in_sql = False

        for line in lines:
            if line.startswith("CONFIDENCE:"):
                try:
                    confidence = float(line.replace("CONFIDENCE:", "").strip())
                    confidence = max(0.0, min(1.0, confidence))
                except:
                    confidence = 0.5
            elif line.startswith("SQL:"):
                in_sql = True
                sql_part = line.replace("SQL:", "").strip()
                if sql_part:
                    sql_lines.append(sql_part)
            elif in_sql:
                sql_lines.append(line)

        sql = "\n".join(sql_lines).strip()
        sql = re.sub(r"^```sql\s*", "", sql, flags=re.IGNORECASE)
        sql = re.sub(r"^```\s*", "", sql)
        sql = re.sub(r"\s*```$", "", sql)
        return sql.strip(), confidence
    except:
        return "", 0.5


def collect_spider1_signals(num_samples=98):
    with open(SPIDER1_DATA_FILE) as f:
        data = json.load(f)[:num_samples]

    few_shot = get_relevant_few_shot_examples(num_examples=3)
    schema_cache = {}
    signal_data = []

    print(f"Collecting signals for {len(data)} Spider 1.0 questions...\n")

    for item in tqdm(data, desc="Spider 1.0 signals"):
        question = item["question"]
        db_id = item["db_id"]
        gold_sql = item["query"]

        db_path = find_sqlite_db(SPIDER1_DB_DIR, db_id)
        if not db_path:
            continue

        if db_id not in schema_cache:
            try:
                schema_cache[db_id] = SchemaRetriever(db_path)
            except:
                continue

        retriever = schema_cache[db_id]

        # Cosine similarity signals
        from sklearn.metrics.pairwise import cosine_similarity
        query_emb = retriever.embedder.encode([question])
        sims = cosine_similarity(query_emb, retriever.table_embeddings)[0]
        sorted_sims = sorted(sims, reverse=True)

        cos_sim_top1 = float(sorted_sims[0]) if len(sorted_sims) > 0 else 0.0
        cos_sim_top2 = float(sorted_sims[1]) if len(sorted_sims) > 1 else 0.0
        cos_sim_gap = cos_sim_top1 - cos_sim_top2
        cos_sim_variance = float(np.var(sims))
        num_tables = len(retriever.table_names)

        schema_context_r1, _ = retriever.get_scoped_context(question, top_k=3)

        agent = SQLAgent(
            db_path=db_path,
            schema_retriever=retriever,
            api_key=config.ACTIVE_API_KEY,
            model_name=config.MODEL_SPIDER1,
            max_iterations=3,
            few_shot_examples=few_shot,
            provider=config.PROVIDER,
        )

        sql_r1, llm_confidence = get_llm_confidence(agent, question, schema_context_r1)
        sql_complexity = compute_sql_complexity(sql_r1)
        question_length = len(question.split())

        # Execute Round 1
        if sql_r1:
            success_r1, result_r1 = agent.execute_query(sql_r1)
            result_rows_r1 = len(result_r1.get("rows", [])) if success_r1 else -1
        else:
            success_r1, result_r1 = False, None
            result_rows_r1 = -1

        gold_result = get_gold_result(db_path, gold_sql)
        r1_correct = execution_match(result_r1, gold_result) if success_r1 else False

        # Execute Round 2
        schema_context_r2, _ = retriever.get_scoped_context(question, top_k=6)
        sql_r2, _ = get_llm_confidence(agent, question, schema_context_r2)

        if sql_r2:
            success_r2, result_r2 = agent.execute_query(sql_r2)
            result_rows_r2 = len(result_r2.get("rows", [])) if success_r2 else -1
        else:
            success_r2, result_r2 = False, None
            result_rows_r2 = -1

        r2_correct = execution_match(result_r2, gold_result) if success_r2 else False
        egsr_helped = (not r1_correct) and r2_correct

        record = {
            "instance_id": f"spider1_{db_id}_{item.get('question_id', question[:20])}",
            "question": question,
            "db": db_id,
            "benchmark": "spider1",
            "cos_sim_top1": round(cos_sim_top1, 4),
            "cos_sim_top2": round(cos_sim_top2, 4),
            "cos_sim_gap": round(cos_sim_gap, 4),
            "cos_sim_variance": round(cos_sim_variance, 6),
            "llm_confidence": round(llm_confidence, 4),
            "sql_complexity_r1": sql_complexity,
            "question_length": question_length,
            "num_tables_in_db": num_tables,
            "result_rows_r1": result_rows_r1,
            "r1_correct": r1_correct,
            "r2_correct": r2_correct,
            "egsr_helped": egsr_helped,
            "sql_r1": sql_r1[:200] if sql_r1 else "",
            "sql_r2": sql_r2[:200] if sql_r2 else "",
        }

        signal_data.append(record)
        print(f"\n  {db_id}: conf={llm_confidence:.2f}, cos={cos_sim_top1:.3f}, "
              f"complexity={sql_complexity}, r1={'✓' if r1_correct else '✗'}, "
              f"r2={'✓' if r2_correct else '✗'}, egsr_helped={egsr_helped}")

    # Save Spider 1.0 signals
    os.makedirs("results", exist_ok=True)
    with open("results/signal_data_spider1.json", "w") as f:
        json.dump(signal_data, f, indent=2)

    print(f"\n\nSpider 1.0 signals saved to results/signal_data_spider1.json")
    print(f"Total records: {len(signal_data)}")
    print(f"EGSR helped: {sum(1 for r in signal_data if r['egsr_helped'])}")
    print(f"EGSR did NOT help: {sum(1 for r in signal_data if not r['egsr_helped'])}")

    # Combine with Spider 2.0 data
    spider2_file = "results/signal_data.json"
    if os.path.exists(spider2_file):
        with open(spider2_file) as f:
            spider2_data = json.load(f)
        # Add benchmark label
        for d in spider2_data:
            d["benchmark"] = "spider2"

        combined = signal_data + spider2_data
        with open("results/signal_data_combined.json", "w") as f:
            json.dump(combined, f, indent=2)

        print(f"\nCombined dataset saved to results/signal_data_combined.json")
        print(f"Total combined records: {len(combined)}")
        print(f"Total EGSR helped: {sum(1 for r in combined if r['egsr_helped'])}")
        print(f"Total EGSR did NOT help: {sum(1 for r in combined if not r['egsr_helped'])}")


if __name__ == "__main__":
    collect_spider1_signals()
