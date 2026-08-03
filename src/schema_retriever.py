"""
schema_retriever.py
--------------------
Yeh module AskDB paper ka "Dynamic Schema-Aware Prompting" implement karta hai.

Kaam yeh karta hai:
1. Database schema (tables, columns, FK/PK) extract karta hai SQLite se
2. Table names ko embeddings mein convert karta hai (all-MiniLM-L6-v2 model se)
3. User ki query se semantic search karta hai - kaunse tables relevant hain
4. IMPROVEMENT (paper se aage): Column-level sample values bhi fetch karta hai
   taaki LLM ko pata chale actual data kaisa dikhta hai (yeh ReFoRCE ka idea hai)
"""

import sqlite3
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class SchemaRetriever:
    def __init__(self, db_path, embedding_model_name="all-MiniLM-L6-v2", offline_test_mode=False):
        """
        offline_test_mode=True: Hugging Face access ke bina TF-IDF mock embedder use karta hai
                                 (SIRF testing ke liye, jaise Claude ke sandbox mein).
        offline_test_mode=False: Asli sentence-transformers model use karta hai (PRODUCTION,
                                  tumhare local machine pe yeh use hoga - better accuracy).
        """
        self.db_path = db_path

        if offline_test_mode:
            from mock_embedder_for_testing import MockEmbedder
            self.embedder = MockEmbedder(embedding_model_name)
        else:
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer(embedding_model_name)

        self.full_schema = self._extract_full_schema()
        self.table_names = list(self.full_schema.keys())
        # Pre-compute embeddings for all table names (paper ke jaisa "pre-computed vector index")
        self.table_embeddings = self.embedder.encode(self.table_names)

    def _extract_full_schema(self):
        """
        SQLite database se poora schema nikalta hai:
        - Table names
        - Column names + data types
        - Primary keys
        - Foreign keys (constraints)
        """
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
        tables = [row[0] for row in cur.fetchall()]

        schema = {}
        for table in tables:
            # Column info: cid, name, type, notnull, default, is_pk
            cur.execute(f"PRAGMA table_info('{table}')")
            columns = cur.fetchall()

            # Foreign key info
            cur.execute(f"PRAGMA foreign_key_list('{table}')")
            fks = cur.fetchall()  # (id, seq, table, from, to, on_update, on_delete, match)

            col_list = []
            for col in columns:
                cid, name, dtype, notnull, default, is_pk = col
                col_list.append({
                    "name": name,
                    "type": dtype,
                    "is_pk": bool(is_pk),
                })

            fk_list = []
            for fk in fks:
                fk_list.append({
                    "from_column": fk[3],
                    "to_table": fk[2],
                    "to_column": fk[4],
                })

            schema[table] = {
                "columns": col_list,
                "foreign_keys": fk_list,
            }

        conn.close()
        return schema

    def get_sample_values(self, table, column, limit=3):
        """
        IMPROVEMENT over paper: column ke sample values fetch karta hai.
        Yeh LLM ko batata hai ki actual data kaisa format mein hai
        (jaise date format, enum values, naming patterns).
        Yeh ReFoRCE paper ka "Column Exploration" concept hai.
        """
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        try:
            cur.execute(f"SELECT DISTINCT \"{column}\" FROM \"{table}\" WHERE \"{column}\" IS NOT NULL LIMIT {limit}")
            values = [row[0] for row in cur.fetchall()]
        except sqlite3.Error:
            values = []
        conn.close()
        return values

    def search_relevant_tables(self, user_query, top_k=5):
        """
        Paper ka 'search_tables_by_name' tool - cosine similarity search
        se user query ke liye most relevant tables dhundta hai.
        """
        query_embedding = self.embedder.encode([user_query])
        similarities = cosine_similarity(query_embedding, self.table_embeddings)[0]

        # Top-k tables, similarity score ke order mein
        ranked_indices = np.argsort(similarities)[::-1][:top_k]
        ranked_tables = [(self.table_names[i], float(similarities[i])) for i in ranked_indices]
        return ranked_tables

    def format_schema_as_markdown(self, table_names, include_samples=True):
        """
        Scoped context injection: sirf selected tables ka schema
        ek structured markdown table mein format karta hai, jisme
        Constraint column FK/PK relationships clearly batata hai.
        """
        markdown_blocks = []

        for table in table_names:
            if table not in self.full_schema:
                continue

            info = self.full_schema[table]
            lines = [f"### Table: {table}", "", "| Column | Type | Constraint |", "|---|---|---|"]

            for col in info["columns"]:
                constraint_parts = []
                if col["is_pk"]:
                    constraint_parts.append("PRIMARY KEY")

                # Check if this column is a foreign key
                for fk in info["foreign_keys"]:
                    if fk["from_column"] == col["name"]:
                        constraint_parts.append(
                            f"FOREIGN KEY REFERENCES {fk['to_table']}({fk['to_column']})"
                        )

                constraint_str = ", ".join(constraint_parts) if constraint_parts else "-"

                col_line = f"| {col['name']} | {col['type']} | {constraint_str} |"

                if include_samples:
                    samples = self.get_sample_values(table, col["name"])
                    if samples:
                        col_line += f" <!-- sample values: {samples} -->"

                lines.append(col_line)

            markdown_blocks.append("\n".join(lines))

        return "\n\n".join(markdown_blocks)

    def get_scoped_context(self, user_query, top_k=5, include_samples=True):
        """
        Main function jo poora pipeline run karta hai:
        1. Relevant tables dhundo
        2. Unka schema markdown format mein banao
        3. Return karo prompt mein use karne ke liye
        """
        ranked_tables = self.search_relevant_tables(user_query, top_k=top_k)
        table_names = [t[0] for t in ranked_tables]
        schema_context = self.format_schema_as_markdown(table_names, include_samples=include_samples)
        return schema_context, ranked_tables


# ---------------- TEST CODE ----------------
if __name__ == "__main__":
    # offline_test_mode=True kyunki sandbox mein Hugging Face access nahi hai.
    # Apne local machine pe offline_test_mode=False (ya hata do) karna for production.
    retriever = SchemaRetriever("data/sample_dbs/company_db.sqlite", offline_test_mode=True)

    print("=== Full schema mein kitne tables hain ===")
    print(retriever.table_names)
    print()

    test_query = "Show me employee salaries by department"
    print(f"=== Query: '{test_query}' ke liye relevant tables ===")
    ranked = retriever.search_relevant_tables(test_query, top_k=3)
    for table, score in ranked:
        print(f"  {table}: similarity score = {score:.4f}")
    print()

    print("=== Scoped Schema Context (LLM ko yeh prompt mein jaayega) ===")
    context, _ = retriever.get_scoped_context(test_query, top_k=2)
    print(context)
