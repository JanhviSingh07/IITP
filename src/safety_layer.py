"""
safety_layer.py
----------------
Paper ka "Multi-Layered Safety Protocol" implement karta hai:
1. Risk classification - High-Risk vs Low-Risk
2. Automated guardrail playbooks:
   - PII Shield
   - SELECT * Interception
   - Destructive Operation Playbook (double confirmation)
"""

import re


# State-modifying SQL keywords jo High-Risk classify karte hain
HIGH_RISK_KEYWORDS = ["UPDATE", "DELETE", "DROP", "TRUNCATE", "ALTER", "INSERT", "CREATE"]

# Common PII-related column name patterns (semantic check ke saath combine karo)
PII_PATTERNS = [
    "ssn", "social_security", "password", "credit_card", "passport",
    "maiden_name", "bio", "date_of_birth", "dob", "phone_number",
    "salary", "email", "address",
]

DESTRUCTIVE_KEYWORDS = ["TRUNCATE", "DROP"]


def classify_risk(sql_or_question):
    """
    Risk classification - paper ke jaisa.
    Returns: "HIGH" or "LOW"
    """
    text_upper = sql_or_question.upper()

    for keyword in HIGH_RISK_KEYWORDS:
        if keyword in text_upper:
            return "HIGH"

    return "LOW"


def is_destructive(sql):
    """Double confirmation chahiye in operations ke liye."""
    sql_upper = sql.upper()
    return any(keyword in sql_upper for keyword in DESTRUCTIVE_KEYWORDS)


def check_pii_exposure(sql, schema_columns):
    """
    PII Shield - basic keyword check (production mein isko LLM-based
    semantic check se replace karo, paper jaisa "instructed LLM reasoning").
    """
    sql_lower = sql.lower()
    flagged_columns = []
    for col in schema_columns:
        col_lower = col.lower()
        if any(pattern in col_lower for pattern in PII_PATTERNS) and col_lower in sql_lower:
            flagged_columns.append(col)
    return flagged_columns


def check_select_star(sql):
    """SELECT * Interception playbook."""
    pattern = r"SELECT\s+\*\s+FROM"
    return bool(re.search(pattern, sql, re.IGNORECASE))


def evaluate_query_safety(sql, schema_columns=None):
    """
    Main safety check function - sab playbooks ek saath run karta hai.
    Returns a dict with risk level and any warnings.
    """
    schema_columns = schema_columns or []

    risk_level = classify_risk(sql)
    warnings = []

    if is_destructive(sql):
        warnings.append("DESTRUCTIVE_OPERATION: TRUNCATE/DROP detected - double confirmation required")

    pii_cols = check_pii_exposure(sql, schema_columns)
    if pii_cols:
        warnings.append(f"PII_SHIELD: Sensitive columns detected: {pii_cols}")
        risk_level = "HIGH"

    if check_select_star(sql):
        warnings.append("PERFORMANCE_WARNING: SELECT * detected - consider specifying columns")

    return {
        "risk_level": risk_level,
        "warnings": warnings,
        "requires_confirmation": risk_level == "HIGH",
    }


# ---------------- TEST CODE ----------------
if __name__ == "__main__":
    test_queries = [
        "SELECT * FROM employees",
        "DELETE FROM employees WHERE emp_id = 1",
        "SELECT salary, emp_name FROM employees",
        "TRUNCATE TABLE employees",
    ]

    for q in test_queries:
        result = evaluate_query_safety(q, schema_columns=["salary", "emp_name", "emp_id"])
        print(f"Query: {q}")
        print(f"  -> {result}\n")
