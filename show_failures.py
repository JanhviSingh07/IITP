"""
show_failures.py
-----------------
Results JSON se sirf FAILED cases nikal ke readable format mein dikhata hai -
taaki easily dekh sakein kya question tha, gold SQL kya tha, generated SQL
kya tha, aur kahan mismatch hua.
"""

import json

with open("results/spider1_results.json", "r") as f:
    results = json.load(f)

failed = [r for r in results if not r.get("result_matched")]

print(f"Total failed cases: {len(failed)}\n")
print("=" * 70)

for i, case in enumerate(failed, 1):
    print(f"\n--- Failed Case {i} ---")
    print(f"Question: {case['question']}")
    print(f"Hardness (estimated): {case['hardness']}")
    print(f"\nGold SQL (correct answer):\n  {case['gold_sql']}")
    print(f"\nGenerated SQL (what AskDB produced):\n  {case['generated_sql']}")
    print(f"\nExecution success: {case['execution_success']}")
    if case.get("error_message"):
        print(f"Error message: {case['error_message']}")
    print("=" * 70)    