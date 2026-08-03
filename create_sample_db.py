"""
create_sample_db.py
--------------------


Database: "company_db" - employees, departments, projects (jaisa real-world schema)
"""

import sqlite3
import os

DB_PATH = "data/sample_dbs/company_db.sqlite"
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

# Agar pehle se exist karta hai toh delete karo, fresh banayein
if os.path.exists(DB_PATH):
    os.remove(DB_PATH)

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

# ---- Table 1: departments ----
cur.execute("""
CREATE TABLE departments (
    dept_id INTEGER PRIMARY KEY,
    dept_name TEXT NOT NULL,
    location TEXT
)
""")

# ---- Table 2: employees (FOREIGN KEY -> departments) ----
cur.execute("""
CREATE TABLE employees (
    emp_id INTEGER PRIMARY KEY,
    emp_name TEXT NOT NULL,
    dept_id INTEGER,
    salary REAL,
    hire_date TEXT,
    FOREIGN KEY (dept_id) REFERENCES departments(dept_id)
)
""")

# ---- Table 3: projects (FOREIGN KEY -> employees as project lead) ----
cur.execute("""
CREATE TABLE projects (
    project_id INTEGER PRIMARY KEY,
    project_name TEXT NOT NULL,
    lead_emp_id INTEGER,
    budget REAL,
    status TEXT,
    FOREIGN KEY (lead_emp_id) REFERENCES employees(emp_id)
)
""")

# ---- Sample Data ----
departments = [
    (1, "Engineering", "Bangalore"),
    (2, "Sales", "Mumbai"),
    (3, "HR", "Delhi"),
]
cur.executemany("INSERT INTO departments VALUES (?, ?, ?)", departments)

employees = [
    (1, "Rahul Sharma", 1, 95000.0, "2021-03-15"),
    (2, "Priya Singh", 1, 105000.0, "2020-07-01"),
    (3, "Amit Kumar", 2, 75000.0, "2022-01-10"),
    (4, "Sneha Patel", 3, 60000.0, "2019-11-20"),
    (5, "Vikram Rao", 1, 120000.0, "2018-05-05"),
]
cur.executemany("INSERT INTO employees VALUES (?, ?, ?, ?, ?)", employees)

projects = [
    (1, "AI Platform", 5, 500000.0, "active"),
    (2, "Sales CRM", 3, 200000.0, "completed"),
    (3, "Mobile App", 2, 350000.0, "active"),
]
cur.executemany("INSERT INTO projects VALUES (?, ?, ?, ?, ?)", projects)

conn.commit()
conn.close()

print(f"Sample database created at: {DB_PATH}")
print("Tables: departments, employees, projects")
