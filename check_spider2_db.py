import sqlite3, json, os

db_path = 'data/spider2/Spider2/spider2-lite/resource/databases/sqlite/E_commerce/E_commerce.sqlite'
conn = sqlite3.connect(db_path)
cur = conn.cursor()

cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [r[0] for r in cur.fetchall()]
print('Tables:', tables)
print()

for table in tables[:3]:
    cur.execute(f'SELECT COUNT(*) FROM "{table}"')
    count = cur.fetchone()[0]
    print(f'{table}: {count} rows')

conn.close()

# JSON file ka structure dekho
json_path = 'data/spider2/Spider2/spider2-lite/resource/databases/sqlite/E_commerce/customers.json'
with open(json_path, encoding='utf-8') as f:
    data = json.load(f)
print()
print('JSON customers type:', type(data))
if isinstance(data, list) and data:
    print('First record:', list(data[0].items())[:3])
elif isinstance(data, dict):
    print('Dict keys:', list(data.keys())[:5])
    print('Sample:', str(data)[:200])
