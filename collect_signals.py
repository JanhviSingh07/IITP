"""
collect_signals.py
-------------------
Cost-Model EGSR ke liye training data collect karta hai.

Har query ke liye yeh signals log karta hai:
1. cosine_similarity_top1    - top table ka similarity score
2. cosine_similarity_variance - scores ka spread
3. cosine_similarity_gap     - top1 aur top2 ka difference
4. llm_confidence            - LLM se explicitly maanga score
5. sql_complexity            - JOINs + subqueries count
6. question_length           - words count
7. num_tables_retrieved      - kitni tables retrieve hui
8. result_rows_round1        - Round 1 ka result (rows count)
9. egsr_helped               - LABEL: True agar Round 2 ne better result diya

Output: results/signal_data.json (training data for cost model)
"""

import os, sys, json, sqlite3, re, numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, os.path.dirname(__file__))

from schema_retriever import SchemaRetriever
from sql_agent import SQLAgent
from few_shot_examples import get_relevant_few_shot_examples
import config

SPIDER2_BASE = "data/spider2/Spider2/spider2-lite"
QUESTIONS_FILE = f"{SPIDER2_BASE}/spider2-lite.jsonl"
GOLD_SQL_DIR = f"{SPIDER2_BASE}/evaluation_suite/gold/sql"
SQLITE_DB_DIR = f"{SPIDER2_BASE}/resource/databases/sqlite"


def find_sqlite_db(db_name):
    for folder in os.listdir(SQLITE_DB_DIR):
        if folder.lower() == db_name.lower():
            db_folder = os.path.join(SQLITE_DB_DIR, folder)
            for f in os.listdir(db_folder):
                if f.endswith('.sqlite') or f.endswith('.db'):
                    return os.path.join(db_folder, f)
    return None


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
    """SQL complexity score - number of JOINs, subqueries, CTEs etc."""
    sql_upper = sql.upper() if sql else ""
    score = 0
    score += sql_upper.count("JOIN")
    score += sql_upper.count("SELECT") - 1  # nested selects
    score += sql_upper.count("WITH ")       # CTEs
    score += sql_upper.count("GROUP BY")
    score += sql_upper.count("HAVING")
    score += sql_upper.count("ORDER BY")
    score += sql_upper.count("INTERSECT") + sql_upper.count("EXCEPT") + sql_upper.count("UNION")
    return max(0, score)


def get_llm_confidence(agent, question, schema_context):
    """
    LLM se explicitly SQL + confidence score maango.
    Returns: (sql, confidence_score)
    """
    prompt = f"""You are an expert SQL agent. Given a database schema and question,
generate a SQL query AND rate your confidence.

Database Schema:
{schema_context}

Question: {question}

Respond in EXACTLY this format (no extra text):
CONFIDENCE: <number between 0.0 and 1.0>
SQL: <your sql query here>"""

    try:
        raw = agent._call_llm(prompt)
        lines = raw.strip().split('\n')
        confidence = 0.5  # default
        sql = ""

        for line in lines:
            if line.startswith("CONFIDENCE:"):
                try:
                    confidence = float(line.replace("CONFIDENCE:", "").strip())
                    confidence = max(0.0, min(1.0, confidence))
                except:
                    confidence = 0.5
            elif line.startswith("SQL:"):
                sql = line.replace("SQL:", "").strip()
            elif sql and not line.startswith("CONFIDENCE:"):
                sql += "\n" + line

        # Clean SQL
        sql = re.sub(r"^```sql\s*", "", sql, flags=re.IGNORECASE)
        sql = re.sub(r"^```\s*", "", sql)
        sql = re.sub(r"\s*```$", "", sql)
        return sql.strip(), confidence

    except Exception as e:
        return "", 0.5


def load_evaluatable_questions():
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

    return evaluatable


def collect_signals():
    data = load_evaluatable_questions()
    few_shot = get_relevant_few_shot_examples(num_examples=3)
    schema_cache = {}
    signal_data = []

    print(f"Collecting signals for {len(data)} questions...\n")

    for item in tqdm(data, desc="Collecting signals"):
        question = item['question']
        db_path = item['db_path']
        gold_sql = item['gold_sql']
        db_id = item['db']
        instance_id = item['instance_id']

        if db_id not in schema_cache:
            try:
                schema_cache[db_id] = SchemaRetriever(db_path)
            except Exception as e:
                print(f"Schema error for {db_id}: {e}")
                continue

        retriever = schema_cache[db_id]

        # ---- Signal 1,2,3: Cosine similarity stats ----
        query_embedding = retriever.embedder.encode([question])
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(query_embedding, retriever.table_embeddings)[0]
        sorted_sims = sorted(similarities, reverse=True)

        cos_sim_top1 = float(sorted_sims[0]) if len(sorted_sims) > 0 else 0.0
        cos_sim_top2 = float(sorted_sims[1]) if len(sorted_sims) > 1 else 0.0
        cos_sim_gap = cos_sim_top1 - cos_sim_top2
        cos_sim_variance = float(np.var(similarities))
        num_tables = len(retriever.table_names)

        # ---- Signal 4: Question length ----
        question_length = len(question.split())

        # Get schema context (top-3, Round 1)
        schema_context_r1, ranked_r1 = retriever.get_scoped_context(question, top_k=3)

        # ---- Signal 5: LLM Confidence + SQL ----
        agent = SQLAgent(
            db_path=db_path,
            schema_retriever=retriever,
            api_key=config.ACTIVE_API_KEY,
            model_name=config.MODEL_SPIDER2,
            max_iterations=3,
            few_shot_examples=few_shot,
            provider=config.PROVIDER,
        )

        sql_r1, llm_confidence = get_llm_confidence(agent, question, schema_context_r1)

        # ---- Signal 6: SQL Complexity ----
        sql_complexity = compute_sql_complexity(sql_r1)

        # ---- Execute Round 1 ----
        if sql_r1:
            success_r1, result_r1 = agent.execute_query(sql_r1)
            if success_r1:
                result_rows_r1 = len(result_r1.get("rows", []))
            else:
                result_rows_r1 = -1  # error
        else:
            success_r1, result_r1 = False, None
            result_rows_r1 = -1

        # ---- Round 1 correct? ----
        gold_result = get_gold_result(db_path, gold_sql)
        r1_correct = execution_match(result_r1, gold_result) if success_r1 else False

        # ---- Round 2 (EGSR expansion - top-6) ----
        schema_context_r2, _ = retriever.get_scoped_context(question, top_k=6)
        sql_r2, _ = get_llm_confidence(agent, question, schema_context_r2)

        if sql_r2:
            success_r2, result_r2 = agent.execute_query(sql_r2)
            result_rows_r2 = len(result_r2.get("rows", [])) if success_r2 else -1
        else:
            success_r2, result_r2 = False, None
            result_rows_r2 = -1

        r2_correct = execution_match(result_r2, gold_result) if success_r2 else False

        # ---- LABEL: Did EGSR help? ----
        # egsr_helped = True agar Round 2 correct tha but Round 1 nahi tha
        egsr_helped = (not r1_correct) and r2_correct

        record = {
            "instance_id": instance_id,
            "question": question,
            "db": db_id,
            # Signals (features)
            "cos_sim_top1": round(cos_sim_top1, 4),
            "cos_sim_top2": round(cos_sim_top2, 4),
            "cos_sim_gap": round(cos_sim_gap, 4),
            "cos_sim_variance": round(cos_sim_variance, 6),
            "llm_confidence": round(llm_confidence, 4),
            "sql_complexity_r1": sql_complexity,
            "question_length": question_length,
            "num_tables_in_db": num_tables,
            "result_rows_r1": result_rows_r1,
            # Intermediate results
            "r1_correct": r1_correct,
            "r2_correct": r2_correct,
            # LABEL
            "egsr_helped": egsr_helped,
            # SQL generated
            "sql_r1": sql_r1[:200] if sql_r1 else "",
            "sql_r2": sql_r2[:200] if sql_r2 else "",
        }

        signal_data.append(record)

        print(f"\n  {instance_id}: conf={llm_confidence:.2f}, "
              f"cos={cos_sim_top1:.3f}, complexity={sql_complexity}, "
              f"r1={'✓' if r1_correct else '✗'}, r2={'✓' if r2_correct else '✗'}, "
              f"egsr_helped={egsr_helped}")

    # Save
    os.makedirs("results", exist_ok=True)
    with open("results/signal_data.json", "w") as f:
        json.dump(signal_data, f, indent=2)

    print(f"\n\nData saved to results/signal_data.json")
    print(f"Total records: {len(signal_data)}")
    print(f"EGSR helped: {sum(1 for r in signal_data if r['egsr_helped'])}")
    print(f"EGSR did NOT help: {sum(1 for r in signal_data if not r['egsr_helped'])}")


if __name__ == "__main__":
    collect_signals()
