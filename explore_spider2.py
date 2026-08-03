import json, os

with open('data/spider2/Spider2/spider2-lite/spider2-lite.jsonl') as f:
    data = [json.loads(l) for l in f]

lo_items = {item['instance_id']: item for item in data if item['instance_id'].startswith('local')}

sql_dir = 'data/spider2/Spider2/spider2-lite/evaluation_suite/gold/sql'
lo_sqls = [f for f in os.listdir(sql_dir) if f.startswith('local')]

sqlite_dir = 'data/spider2/Spider2/spider2-lite/resource/databases/sqlite'
sqlite_dbs = [d.lower() for d in os.listdir(sqlite_dir)]

print('Local questions with gold SQL and their databases:')
matched = []
unmatched = []
for sql_file in lo_sqls:
    instance_id = sql_file.replace('.sql', '')
    item = lo_items.get(instance_id, {})
    db = item.get('db', 'unknown')
    if db.lower() in sqlite_dbs:
        matched.append((instance_id, db))
    else:
        unmatched.append((instance_id, db))

print(f'\nMatched (SQLite DB available): {len(matched)}')
for inst, db in matched:
    print(f'  {inst}: {db}')

print(f'\nUnmatched (SQLite DB NOT available): {len(unmatched)}')
for inst, db in unmatched:
    print(f'  {inst}: {db}')