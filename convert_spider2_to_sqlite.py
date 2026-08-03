"""
convert_spider2_to_sqlite.py
-----------------------------
Spider 2.0 ke JSON databases ko SQLite mein convert karta hai.
Har database folder mein DDL.csv (schema) aur JSON files (data) hain.
Yeh script in sab ko ek .sqlite file mein convert karta hai.

Run karo ek baar:
    python convert_spider2_to_sqlite.py
"""

import os
import json
import sqlite3
import csv

SQLITE_DIR = "data/spider2/Spider2/spider2-lite/resource/databases/sqlite"

def read_ddl(ddl_path):
    """DDL.csv se CREATE TABLE statements padhta hai.
    Format: table_name,DDL (CSV with two columns)
    """
    statements = []
    try:
        import csv
        with open(ddl_path, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.DictReader(f)
            for row in reader:
                ddl = row.get('DDL', '').strip()
                if ddl and ddl.upper().startswith('CREATE'):
                    if not ddl.endswith(';'):
                        ddl += ';'
                    statements.append(ddl)
    except Exception as e:
        print(f"  DDL read error: {e}")
    return statements

def json_to_sqlite(db_folder, db_name):
    """Ek database folder ko SQLite mein convert karta hai."""
    sqlite_path = os.path.join(db_folder, f"{db_name}.sqlite")

    # Already exists toh skip karo
    if os.path.exists(sqlite_path):
        print(f"  Already exists: {db_name}.sqlite")
        return True

    ddl_path = os.path.join(db_folder, "DDL.csv")
    if not os.path.exists(ddl_path):
        print(f"  No DDL.csv found for {db_name}")
        return False

    # DDL se table schemas padhte hain
    ddl_statements = read_ddl(ddl_path)
    if not ddl_statements:
        print(f"  No CREATE statements found in DDL for {db_name}")
        return False

    conn = sqlite3.connect(sqlite_path)
    cur = conn.cursor()

    # Tables create karo
    tables_created = []
    for stmt in ddl_statements:
        try:
            cur.execute(stmt)
            # Extract table name
            import re
            match = re.search(r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?["`]?(\w+)["`]?', stmt, re.IGNORECASE)
            if match:
                tables_created.append(match.group(1).lower())
        except sqlite3.Error as e:
            # Try without strict mode
            try:
                stmt_modified = stmt.replace('NOT NULL', '').replace('UNIQUE', '')
                cur.execute(stmt_modified)
            except:
                pass

    conn.commit()

    # JSON data load karo
    json_files = [f for f in os.listdir(db_folder) if f.endswith('.json')]
    rows_inserted = 0

    for json_file in json_files:
        table_name = json_file.replace('.json', '').lower()
        json_path = os.path.join(db_folder, json_file)

        try:
            with open(json_path, 'r', encoding='utf-8', errors='ignore') as f:
                data = json.load(f)

            # Spider 2.0 format: dict with 'sample_rows', 'column_names', etc.
            if isinstance(data, dict):
                rows = data.get('sample_rows', [])
                col_names = data.get('column_names', [])
            elif isinstance(data, list):
                rows = data
                col_names = list(rows[0].keys()) if rows and isinstance(rows[0], dict) else []
            else:
                continue

            if not rows:
                continue

            # Column names from first row if not provided
            if not col_names and isinstance(rows[0], dict):
                col_names = list(rows[0].keys())

            if not col_names:
                continue

            # Table exist karta hai?
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                       (table_name,))
            if not cur.fetchone():
                col_defs = ', '.join([f'"{c}" TEXT' for c in col_names])
                cur.execute(f'CREATE TABLE IF NOT EXISTS "{table_name}" ({col_defs})')

            col_str = ', '.join([f'"{c}"' for c in col_names])
            placeholders = ', '.join(['?' for _ in col_names])

            for row in rows:
                if isinstance(row, dict):
                    values = [str(row.get(c, '')) if row.get(c) is not None else None
                             for c in col_names]
                else:
                    values = [str(v) if v is not None else None for v in row]
                try:
                    cur.execute(f'INSERT OR IGNORE INTO "{table_name}" ({col_str}) VALUES ({placeholders})',
                               values)
                    rows_inserted += 1
                except:
                    pass

        except Exception as e:
            pass

    conn.commit()
    conn.close()
    print(f"  Created {db_name}.sqlite - {len(tables_created)} tables, {rows_inserted} rows")
    return True


# Main conversion
print("Converting Spider 2.0 databases to SQLite...\n")
success = 0
failed = 0

for db_folder_name in sorted(os.listdir(SQLITE_DIR)):
    db_folder = os.path.join(SQLITE_DIR, db_folder_name)
    if not os.path.isdir(db_folder):
        continue

    print(f"Processing: {db_folder_name}")
    if json_to_sqlite(db_folder, db_folder_name):
        success += 1
    else:
        failed += 1

print(f"\nDone! Success: {success}, Failed: {failed}")
print("\nVerifying created SQLite files:")
for db_folder_name in sorted(os.listdir(SQLITE_DIR)):
    db_folder = os.path.join(SQLITE_DIR, db_folder_name)
    sqlite_file = os.path.join(db_folder, f"{db_folder_name}.sqlite")
    if os.path.exists(sqlite_file):
        size = os.path.getsize(sqlite_file)
        print(f"  {db_folder_name}.sqlite ({size/1024:.1f} KB)")
