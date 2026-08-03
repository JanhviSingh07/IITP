import json, sqlite3, os, sys
sys.path.insert(0, '.')
import config

with open('results/spider1_results.json') as f:
    results = json.load(f)
with open(f'{config.SPIDER1_DATA_DIR}/dev.json') as f:
    full_data = json.load(f)
q_to_item = {item['question']: item for item in full_data}

recently_changed = [
    'List all singer names in concerts in year 2014.',
    'Find the model of the car whose weight is below the average weight.',
    'Find the major and age of students who do not have a cat pet.',
    'Find the first name and age of students who have a dog but do not have a cat as a pet.',
]

for r in results:
    if any(q in r['question'] for q in recently_changed) and r.get('generated_sql'):
        item = q_to_item.get(r['question'])
        if not item:
            continue
        db_id = item['db_id']
        db_path = os.path.join(config.SPIDER1_DATA_DIR, 'database', db_id, f'{db_id}.sqlite')
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute(item['query'])
        gold = cur.fetchall()
        cur.execute(r['generated_sql'])
        gen = cur.fetchall()
        conn.close()
        print(f"Q: {r['question'][:60]}")
        print(f"  Gold rows: {len(gold)}, Gen rows: {len(gen)}, Set match: {set(gold)==set(gen)}")
        print()