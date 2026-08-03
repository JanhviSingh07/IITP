"""
few_shot_examples.py
---------------------
IMPROVEMENT over paper: Yeh module Hard/Extra-Hard SQL patterns ke
curated examples store karta hai. Yeh prompt mein "in-context learning"
ke liye use hote hain - LLM ko pattern dikhana se woh complex queries
better generate karta hai.

Spider 1.0 mein hum dekh chuke hain (paper ke results se):
- Easy/Medium: 100% accuracy (already perfect)
- Hard: 80.7%, Extra Hard: 80.2% (yahan improvement chahiye)

In dono categories ke common patterns hote hain:
1. Nested subqueries (WHERE column IN (SELECT ...))
2. Multiple JOINs (3+ tables)
3. GROUP BY with HAVING
4. Set operations (INTERSECT, EXCEPT, UNION)
5. Correlated subqueries
6. Aggregate functions with nested conditions
"""

HARD_QUERY_EXAMPLES = [
    {
        "category": "nested_subquery",
        "question": "Find the names of employees who earn more than the average salary in their department.",
        "sql": """SELECT e.emp_name FROM employees e
WHERE e.salary > (
    SELECT AVG(e2.salary) FROM employees e2 WHERE e2.dept_id = e.dept_id
)"""
    },
    {
        "category": "multi_join",
        "question": "List project names along with the department name of their lead employee.",
        "sql": """SELECT p.project_name, d.dept_name
FROM projects p
JOIN employees e ON p.lead_emp_id = e.emp_id
JOIN departments d ON e.dept_id = d.dept_id"""
    },
    {
        "category": "group_by_having",
        "question": "Find departments that have more than 2 employees.",
        "sql": """SELECT d.dept_name, COUNT(e.emp_id) AS emp_count
FROM departments d
JOIN employees e ON d.dept_id = e.dept_id
GROUP BY d.dept_id, d.dept_name
HAVING COUNT(e.emp_id) > 2"""
    },
    {
        "category": "set_operation",
        "question": "Find employee names who are NOT leading any project.",
        "sql": """SELECT emp_name FROM employees
EXCEPT
SELECT e.emp_name FROM employees e JOIN projects p ON e.emp_id = p.lead_emp_id"""
    },
    {
        "category": "correlated_subquery",
        "question": "Find the department with the highest total salary expenditure.",
        "sql": """SELECT d.dept_name
FROM departments d
WHERE (
    SELECT SUM(e.salary) FROM employees e WHERE e.dept_id = d.dept_id
) = (
    SELECT MAX(dept_total) FROM (
        SELECT SUM(salary) AS dept_total FROM employees GROUP BY dept_id
    )
)"""
    },
    {
        "category": "nested_aggregate",
        "question": "Find the project with the largest budget among active projects.",
        "sql": """SELECT project_name FROM projects
WHERE status = 'active'
ORDER BY budget DESC
LIMIT 1"""
    },
]


def get_relevant_few_shot_examples(num_examples=3):
    """
    Production mein isko improve kar sakte ho - user query ke based pe
    sabse relevant examples select karna (semantic similarity se), abhi
    ke liye top-N fixed examples return kar rahe hain different categories se.
    """
    return HARD_QUERY_EXAMPLES[:num_examples]


def get_examples_by_category(category):
    """Specific pattern category ke examples chahiye (jaise sirf 'nested_subquery')."""
    return [ex for ex in HARD_QUERY_EXAMPLES if ex["category"] == category]
