"""
sql_agent.py
------------
AskDB paper ka core ReAct framework + EGSR novel contribution.

IMPROVEMENTS over paper:
1. Few-shot examples in prompt
2. Self-consistency voting
3. Semantic result validation
4. EGSR: Execution-Guided Schema Refinement (novel contribution)
5. C-EGSR: Cosine-Similarity-Guided Schema Refinement
6. Groq/Gemini provider support
"""

import sqlite3
import re
import json
import numpy as np
from collections import Counter


class SQLAgent:

    def __init__(
        self,
        db_path,
        schema_retriever,
        api_key,
        model_name="gemini-2.0-flash",
        max_iterations=6,
        few_shot_examples=None,
        result_validator=None,
        provider="gemini",
    ):
        self.db_path = db_path
        self.schema_retriever = schema_retriever
        self.model_name = model_name
        self.max_iterations = max_iterations
        self.few_shot_examples = few_shot_examples or []
        self.provider = provider
        self.result_validator = result_validator

        if provider == "gemini":
            from google import genai

            self.client = genai.Client(api_key=api_key)

        elif provider == "groq":
            from openai import OpenAI

            self.client = OpenAI(
                api_key=api_key,
                base_url="https://api.groq.com/openai/v1",
            )

        else:
            raise ValueError(
                f"Unknown provider: {provider}. "
                f"Use 'gemini' or 'groq'."
            )

    # ---------------------------------------------------------
    # SQL EXECUTION
    # ---------------------------------------------------------

    def execute_query(self, sql):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        try:
            cur.execute(sql)

            rows = cur.fetchall()

            columns = (
                [desc[0] for desc in cur.description]
                if cur.description
                else []
            )

            conn.close()

            return True, {
                "columns": columns,
                "rows": rows,
            }

        except sqlite3.Error as e:
            conn.close()
            return False, str(e)

    # ---------------------------------------------------------
    # PROMPT BUILDING
    # ---------------------------------------------------------

    def _build_prompt(
        self,
        user_question,
        schema_context,
        history=None,
        error_feedback=None,
    ):
        few_shot_text = ""

        if self.few_shot_examples:
            few_shot_text = (
                "Here are some example questions and their correct SQL:\n\n"
            )

            for ex in self.few_shot_examples:
                few_shot_text += (
                    f"Question: {ex['question']}\n"
                    f"SQL: {ex['sql']}\n\n"
                )

        history_text = ""

        if history:
            history_text = (
                "Conversation history (previous attempts):\n"
                + "\n".join(history)
                + "\n\n"
            )

        error_text = ""

        if error_feedback:
            error_text = (
                "IMPORTANT: Your previous SQL attempt failed "
                "with this error:\n"
                f"{error_feedback}\n"
                "Please analyze the error and generate a "
                "corrected SQL query.\n\n"
            )

        prompt = f"""
You are an expert SQL agent. Given a database schema and a natural language
question, generate ONLY the SQL query that answers the question. Do not include
any explanation, markdown formatting, or code fences - output raw SQL only.

IMPORTANT RULES:
1. Select ONLY the columns explicitly asked for in the question - do not add extra columns.
2. Check column names carefully - if a column is named "average" or "total", it is a real
   stored column, NOT a signal to use AVG()/SUM(). Use aggregate functions only when the
   question explicitly asks to calculate/compute something.
3. When a question asks for a name or label, always return the name column, not an ID.
4. Always use DISTINCT when the question uses "different", "unique", or when a JOIN could
   produce duplicate rows for the same entity.
5. Use INTERSECT or EXCEPT directly rather than converting to JOINs.
6. Always return ALL columns the question asks for - never drop a requested column.
7. Use ORDER BY LIMIT 1 directly - do not replace with WHERE X = (SELECT MIN/MAX(X)).

{few_shot_text}
Database Schema (relevant tables only):
{schema_context}

{history_text}{error_text}
Question: {user_question}

SQL Query:
"""

        return prompt

    # ---------------------------------------------------------
    # CLEAN SQL OUTPUT
    # ---------------------------------------------------------

    def _clean_sql_output(self, raw_text):
        text = raw_text.strip()

        text = re.sub(
            r"<think>.*?</think>",
            "",
            text,
            flags=re.DOTALL,
        ).strip()

        text = re.sub(
            r"^```sql\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )

        text = re.sub(
            r"^```\s*",
            "",
            text,
        )

        text = re.sub(
            r"\s*```$",
            "",
            text,
        )

        return text.strip()

    # ---------------------------------------------------------
    # LLM CALL
    # ---------------------------------------------------------

    def _call_llm(self, prompt, temperature=0.0):
        import time

        if self.provider == "gemini":

            from google.genai import types

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature
                ),
            )

            return response.text

        else:
            # Groq
            max_retries = 3

            for attempt in range(max_retries):

                try:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {
                                "role": "user",
                                "content": prompt,
                            }
                        ],
                        temperature=temperature,
                    )

                    return response.choices[0].message.content

                except Exception as e:

                    error_str = str(e)

                    if (
                        "429" in error_str
                        and attempt < max_retries - 1
                    ):
                        wait_time = 60 * (attempt + 1)

                        print(
                            f"\n[Rate limit hit] "
                            f"Waiting {wait_time}s before retry "
                            f"{attempt + 2}/{max_retries}..."
                        )

                        time.sleep(wait_time)

                    else:
                        raise

    # ---------------------------------------------------------
    # BASELINE: REACT SQL AGENT
    # ---------------------------------------------------------

    def answer_question(
        self,
        user_question,
        verbose=True,
    ):
        schema_context, ranked_tables = (
            self.schema_retriever.get_scoped_context(
                user_question,
                top_k=5,
            )
        )

        if verbose:
            print(
                "[Reasoning] Relevant tables found: "
                f"{[t[0] for t in ranked_tables]}"
            )

        history = []
        error_feedback = None
        last_sql = None

        for iteration in range(
            1,
            self.max_iterations + 1,
        ):

            if verbose:
                print(
                    f"\n--- Iteration {iteration} ---"
                )

            prompt = self._build_prompt(
                user_question,
                schema_context,
                history,
                error_feedback,
            )

            raw_output = self._call_llm(prompt)

            sql = self._clean_sql_output(
                raw_output
            )

            last_sql = sql

            if verbose:
                print(
                    f"[Acting] Generated SQL: {sql}"
                )

            success, result = self.execute_query(sql)

            if success:

                if self.result_validator is not None:

                    is_valid, reason = (
                        self.result_validator.validate(
                            user_question,
                            sql,
                            result,
                        )
                    )

                    if verbose:
                        print(
                            "[Validation] "
                            f"{'PASS' if is_valid else 'FAIL'}: "
                            f"{reason}"
                        )

                    if not is_valid:

                        error_feedback = (
                            "The SQL executed successfully "
                            "but the result seems incorrect: "
                            f"{reason}. Please reconsider the "
                            "query logic (check JOINs, filters, "
                            "and aggregations)."
                        )

                        history.append(
                            f"Attempt {iteration}: "
                            f"SQL='{sql}' executed but FAILED "
                            f"semantic validation: {reason}"
                        )

                        continue

                if verbose:
                    print(
                        "[Observation] Success! "
                        f"Rows returned: "
                        f"{len(result['rows'])}"
                    )

                return {
                    "success": True,
                    "sql": sql,
                    "result": result,
                    "iterations_used": iteration,
                }

            else:

                if verbose:
                    print(
                        f"[Observation] Error: {result}"
                    )

                error_feedback = result

                history.append(
                    f"Attempt {iteration}: "
                    f"SQL='{sql}' FAILED with error: "
                    f"{result}"
                )

        return {
            "success": False,
            "sql": last_sql,
            "result": error_feedback,
            "iterations_used": self.max_iterations,
        }

    # ---------------------------------------------------------
    # EGSR: EXECUTION-GUIDED SCHEMA REFINEMENT
    # ---------------------------------------------------------

    def answer_with_egsr(
        self,
        user_question,
        verbose=True,
    ):
        """
        EGSR: Execution-Guided Schema Refinement.

        Round 1:
            Use top-3 tables.

        Round 2:
            If Round 1 fails, expand to top-6 tables.

        Execution results are used to decide whether
        schema refinement is required.
        """

        rounds = [
            {
                "top_k": 3,
                "description": "top-3 tables (standard)",
            },
            {
                "top_k": 6,
                "description": "top-6 tables (EGSR expansion)",
            },
        ]

        best_result = None
        best_sql = None
        best_rows = -1
        round_num = 1

        for round_num, round_config in enumerate(
            rounds,
            1,
        ):

            if verbose:
                print(
                    f"\n[EGSR] Round {round_num}: "
                    f"{round_config['description']}"
                )

            schema_context, ranked_tables = (
                self.schema_retriever.get_scoped_context(
                    user_question,
                    top_k=round_config["top_k"],
                )
            )

            history = []
            error_feedback = None
            round_success = False

            for iteration in range(
                1,
                self.max_iterations + 1,
            ):

                prompt = self._build_prompt(
                    user_question,
                    schema_context,
                    history,
                    error_feedback,
                )

                raw_output = self._call_llm(
                    prompt
                )

                sql = self._clean_sql_output(
                    raw_output
                )

                success, result = (
                    self.execute_query(sql)
                )

                if success:

                    row_count = len(
                        result.get("rows", [])
                    )

                    if verbose:
                        print(
                            f"[EGSR] Round {round_num} "
                            f"iter {iteration}: "
                            f"{row_count} rows"
                        )

                    if (
                        row_count > 0
                        and row_count > best_rows
                    ):
                        best_result = result
                        best_sql = sql
                        best_rows = row_count
                        round_success = True

                    elif (
                        row_count == 0
                        and best_rows <= 0
                    ):
                        best_result = result
                        best_sql = sql

                    break

                else:

                    if verbose:
                        print(
                            f"[EGSR] Round {round_num} "
                            f"iter {iteration}: "
                            f"Error - {result}"
                        )

                    error_feedback = result

                    history.append(
                        f"Attempt {iteration}: "
                        f"SQL='{sql}' FAILED: {result}"
                    )

            if (
                round_success
                and best_rows > 0
            ):

                if verbose:
                    print(
                        f"[EGSR] Good result in "
                        f"round {round_num}, "
                        f"stopping early"
                    )

                break

        if best_sql is None:

            return {
                "success": False,
                "sql": None,
                "result": (
                    "EGSR: All schema refinement "
                    "rounds exhausted without success"
                ),
                "iterations_used": (
                    len(rounds)
                    * self.max_iterations
                ),
                "egsr_rounds_used": round_num,
            }

        return {
            "success": True,
            "sql": best_sql,
            "result": best_result,
            "iterations_used": best_rows,
            "egsr_rounds_used": round_num,
        }

    # ---------------------------------------------------------
    # SELF-CONSISTENCY
    # ---------------------------------------------------------

    def answer_with_self_consistency(
        self,
        user_question,
        num_samples=3,
        verbose=True,
    ):
        """
        Generate multiple SQL candidates and use
        majority voting to choose the best answer.
        """

        schema_context, ranked_tables = (
            self.schema_retriever.get_scoped_context(
                user_question,
                top_k=5,
            )
        )

        candidates = []

        for i in range(num_samples):

            prompt = self._build_prompt(
                user_question,
                schema_context,
            )

            raw_output = self._call_llm(
                prompt,
                temperature=0.7,
            )

            sql = self._clean_sql_output(
                raw_output
            )

            success, result = (
                self.execute_query(sql)
            )

            if success:

                result_signature = json.dumps(
                    result["rows"],
                    default=str,
                    sort_keys=True,
                )

                candidates.append(
                    {
                        "sql": sql,
                        "result": result,
                        "signature": result_signature,
                    }
                )

            if verbose:

                status = (
                    "OK"
                    if success
                    else "FAILED"
                )

                print(
                    f"[Sample {i + 1}] "
                    f"{status}: {sql}"
                )

        if not candidates:

            if verbose:
                print(
                    "[Self-Consistency] "
                    "All failed, falling back "
                    "to ReAct loop..."
                )

            return self.answer_question(
                user_question,
                verbose=verbose,
            )

        signatures = [
            c["signature"]
            for c in candidates
        ]

        most_common_sig, count = (
            Counter(signatures)
            .most_common(1)[0]
        )

        winning_candidate = next(
            c
            for c in candidates
            if c["signature"] == most_common_sig
        )

        if verbose:
            print(
                "[Self-Consistency] Winner: "
                f"{count}/{len(candidates)} votes"
            )

        return {
            "success": True,
            "sql": winning_candidate["sql"],
            "result": winning_candidate["result"],
            "votes": f"{count}/{len(candidates)}",
        }

    # ---------------------------------------------------------
    # C-EGSR: COSINE-SIMILARITY-GUIDED SCHEMA REFINEMENT
    # ---------------------------------------------------------

    def answer_with_cegsr(
        self,
        user_question,
        verbose=True,
        gap_threshold=0.1,
        variance_threshold=0.02,
    ):
        """
        C-EGSR: Cosine-Similarity-Guided Schema Refinement.

        Instead of using the old cost model, this version
        uses retrieval uncertainty based on:

        1. Cosine similarity gap between top-1 and top-2
           retrieved tables.

        2. Variance of cosine similarities across tables.

        If retrieval is uncertain:
            top_k = 6

        Otherwise:
            top_k = 3
        """

        from sklearn.metrics.pairwise import (
            cosine_similarity as cos_sim
        )

        import numpy as np

        # -----------------------------------------------------
        # Compute query embedding
        # -----------------------------------------------------

        query_embedding = (
            self.schema_retriever.embedder.encode(
                [user_question]
            )
        )

        # -----------------------------------------------------
        # Compute cosine similarities
        # -----------------------------------------------------

        similarities = cos_sim(
            query_embedding,
            self.schema_retriever.table_embeddings,
        )[0]

        sorted_sims = sorted(
            similarities,
            reverse=True,
        )

        # -----------------------------------------------------
        # Top-1 similarity
        # -----------------------------------------------------

        cos_sim_top1 = (
            float(sorted_sims[0])
            if len(sorted_sims) > 0
            else 0.0
        )

        # -----------------------------------------------------
        # Top-2 similarity
        # -----------------------------------------------------

        cos_sim_top2 = (
            float(sorted_sims[1])
            if len(sorted_sims) > 1
            else 0.0
        )

        # -----------------------------------------------------
        # Gap between top-1 and top-2
        # -----------------------------------------------------

        gap = cos_sim_top1 - cos_sim_top2

        # -----------------------------------------------------
        # Variance of all similarities
        # -----------------------------------------------------

        variance = float(
            np.var(similarities)
        )

        # -----------------------------------------------------
        # Determine retrieval uncertainty
        # -----------------------------------------------------

        uncertain = (
            gap < gap_threshold
            or variance > variance_threshold
        )

        if verbose:
            print(
                f"[C-EGSR] gap={gap:.4f}, "
                f"variance={variance:.6f}, "
                f"uncertain={uncertain}"
            )

        # -----------------------------------------------------
        # Select number of tables
        # -----------------------------------------------------

        top_k = (
            6
            if uncertain
            else 3
        )

        if verbose:
            print(
                f"[C-EGSR] Using top-{top_k} tables"
            )

        # -----------------------------------------------------
        # Retrieve schema
        # -----------------------------------------------------

        schema_context, _ = (
            self.schema_retriever.get_scoped_context(
                user_question,
                top_k=top_k,
            )
        )

        history = []
        error_feedback = None
        last_sql = None

        # -----------------------------------------------------
        # ReAct SQL generation loop
        # -----------------------------------------------------

        for iteration in range(
            1,
            self.max_iterations + 1,
        ):

            if verbose:
                print(
                    f"\n--- Iteration {iteration} ---"
                )

            prompt = self._build_prompt(
                user_question,
                schema_context,
                history,
                error_feedback,
            )

            raw_output = self._call_llm(
                prompt
            )

            sql = self._clean_sql_output(
                raw_output
            )

            last_sql = sql

            if verbose:
                print(
                    f"[Acting] Generated SQL: {sql}"
                )

            # -------------------------------------------------
            # Execute generated SQL
            # -------------------------------------------------

            success, result = (
                self.execute_query(sql)
            )

            if success:

                if verbose:
                    print(
                        "[Observation] Success! "
                        f"Rows: {len(result['rows'])}"
                    )

                return {
                    "success": True,
                    "sql": sql,
                    "result": result,
                    "iterations_used": iteration,
                    "cegsr_used_expanded": uncertain,
                    "cegsr_top_k": top_k,
                    "cos_sim_gap": round(
                        gap,
                        4,
                    ),
                    "cos_sim_variance": round(
                        variance,
                        6,
                    ),
                }

            else:

                if verbose:
                    print(
                        f"[Observation] Error: "
                        f"{result}"
                    )

                error_feedback = result

                history.append(
                    f"Attempt {iteration}: "
                    f"SQL='{sql}' FAILED: {result}"
                )

        # -----------------------------------------------------
        # All iterations failed
        # -----------------------------------------------------

        return {
            "success": False,
            "sql": last_sql,
            "result": error_feedback,
            "iterations_used": self.max_iterations,
            "cegsr_used_expanded": uncertain,
            "cegsr_top_k": top_k,
        }