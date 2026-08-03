"""
error_analyzer.py
------------------
Failure analysis tool - har failed question ko categorize karta hai
taaki pata chale exactly KAHAN aur KYUN fail ho raha hai.

Categories:
1. SCHEMA_LINKING_ERROR - galat table/column choose kiya
2. SYNTAX_ERROR - SQL dialect issue
3. SEMANTIC_LOGIC_ERROR - SQL chala but galat logic (JOIN/aggregation/filter)
4. EXTERNAL_KNOWLEDGE_GAP - domain knowledge missing
5. NESTED_QUERY_FAILURE - subquery/CTE issue
6. TIMEOUT_OR_MAX_ITERATIONS - agent stuck reh gaya
"""

from collections import Counter


def categorize_error(question, gold_sql, generated_sql, execution_success, error_message=None, result_matched=None):
    """
    Heuristic-based categorization. Production mein isko LLM se bhi
    classify karwa sakte ho for better accuracy ("LLM-as-judge" approach).
    """
    if not execution_success:
        error_msg_lower = (error_message or "").lower()

        if "no such table" in error_msg_lower or "no such column" in error_msg_lower:
            return "SCHEMA_LINKING_ERROR"
        elif "syntax error" in error_msg_lower:
            return "SYNTAX_ERROR"
        else:
            return "OTHER_EXECUTION_ERROR"

    # Execution successful hua but result match nahi hua
    if result_matched is False:
        gold_lower = (gold_sql or "").lower()
        gen_lower = (generated_sql or "").lower()

        # Check for nested query patterns
        if gold_lower.count("select") > 1 and gen_lower.count("select") <= 1:
            return "NESTED_QUERY_FAILURE"

        # Check JOIN count mismatch
        gold_joins = gold_lower.count("join")
        gen_joins = gen_lower.count("join")
        if gold_joins != gen_joins:
            return "SCHEMA_LINKING_ERROR"  # likely wrong tables/joins picked

        # Check aggregate function mismatch
        agg_functions = ["count", "sum", "avg", "max", "min"]
        gold_aggs = set(f for f in agg_functions if f in gold_lower)
        gen_aggs = set(f for f in agg_functions if f in gen_lower)
        if gold_aggs != gen_aggs:
            return "SEMANTIC_LOGIC_ERROR"

        return "SEMANTIC_LOGIC_ERROR"  # default bucket

    return "UNKNOWN"


def analyze_results_batch(results_list):
    """
    results_list: list of dicts, har dict mein hona chahiye:
        {
            "question": str,
            "gold_sql": str,
            "generated_sql": str,
            "execution_success": bool,
            "error_message": str or None,
            "result_matched": bool,
            "hardness": str  ("easy", "medium", "hard", "extra hard")
        }

    Return: summary report
    """
    categories = []
    failed_by_hardness = Counter()

    for r in results_list:
        is_correct = r.get("result_matched", False)
        if is_correct:
            continue  # sirf failures categorize karte hain

        category = categorize_error(
            r["question"], r.get("gold_sql"), r.get("generated_sql"),
            r["execution_success"], r.get("error_message"), r.get("result_matched")
        )
        categories.append(category)
        failed_by_hardness[r.get("hardness", "unknown")] += 1

    category_counts = Counter(categories)
    total_failed = len(categories)
    total = len(results_list)

    report = {
        "total_questions": total,
        "total_failed": total_failed,
        "overall_accuracy": round((total - total_failed) / total * 100, 2) if total else 0,
        "failure_breakdown_by_category": dict(category_counts),
        "failure_breakdown_by_hardness": dict(failed_by_hardness),
        "top_priority_fix": category_counts.most_common(1)[0] if category_counts else None,
    }
    return report


def print_report(report):
    print("=" * 60)
    print("FAILURE ANALYSIS REPORT")
    print("=" * 60)
    print(f"Total Questions: {report['total_questions']}")
    print(f"Total Failed: {report['total_failed']}")
    print(f"Overall Accuracy: {report['overall_accuracy']}%")
    print()
    print("Failure Breakdown by Category (sabse zyada problem kahan hai):")
    for category, count in sorted(report["failure_breakdown_by_category"].items(), key=lambda x: -x[1]):
        pct = round(count / report["total_failed"] * 100, 1) if report["total_failed"] else 0
        print(f"  {category}: {count} cases ({pct}%)")
    print()
    print("Failure Breakdown by Hardness Level:")
    for hardness, count in report["failure_breakdown_by_hardness"].items():
        print(f"  {hardness}: {count} cases")
    print()
    if report["top_priority_fix"]:
        print(f">>> SABSE ZYADA PRIORITY FIX KARO: {report['top_priority_fix'][0]} "
              f"({report['top_priority_fix'][1]} cases)")
    print("=" * 60)


# ---------------- TEST CODE ----------------
if __name__ == "__main__":
    # Dummy sample data - real evaluation se yeh aayega
    sample_results = [
        {"question": "Q1", "gold_sql": "SELECT a FROM t", "generated_sql": "SELECT a FROM t",
         "execution_success": True, "result_matched": True, "hardness": "easy"},
        {"question": "Q2", "gold_sql": "SELECT a FROM t WHERE x IN (SELECT b FROM t2)",
         "generated_sql": "SELECT a FROM t WHERE x = 5",
         "execution_success": True, "result_matched": False, "hardness": "hard"},
        {"question": "Q3", "gold_sql": "SELECT COUNT(*) FROM t JOIN t2", "generated_sql": "SELECT * FROM t",
         "execution_success": False, "error_message": "no such column: y", "result_matched": False,
         "hardness": "extra hard"},
    ]

    report = analyze_results_batch(sample_results)
    print_report(report)
