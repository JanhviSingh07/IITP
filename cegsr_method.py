"""
cegsr_method.py
----------------
C-EGSR method to add to SQLAgent class in sql_agent.py.

INSTRUCTIONS: Copy the method below and paste it inside the SQLAgent class
in src/sql_agent.py, after the answer_with_egsr method.
"""


def answer_with_cegsr(self, user_question, verbose=True,
                      gap_threshold=0.1, variance_threshold=0.02):
    """
    C-EGSR: Signal-Based Proactive Schema Refinement (Novel Contribution)
    ======================================================================
    Key finding from cost model training:
    - cos_sim_gap (top1 - top2 similarity) = 65.87% predictive power
    - cos_sim_variance = 34.13% predictive power
    - LLM confidence, SQL complexity, question length = 0% signal

    Schema retrieval UNCERTAINTY (not model confidence) is the primary
    predictor of whether refinement will help.

    C-EGSR PROACTIVELY decides schema size before executing SQL:
    - uncertain retrieval (low gap OR high variance) → top-6 tables
    - confident retrieval → top-3 tables (standard, cheaper)

    Unlike EGSR which reacts to execution failure, C-EGSR acts upfront.
    """
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim
    import numpy as np

    query_embedding = self.schema_retriever.embedder.encode([user_question])
    similarities = cos_sim(query_embedding, self.schema_retriever.table_embeddings)[0]
    sorted_sims = sorted(similarities, reverse=True)

    cos_sim_top1 = float(sorted_sims[0]) if len(sorted_sims) > 0 else 0.0
    cos_sim_top2 = float(sorted_sims[1]) if len(sorted_sims) > 1 else 0.0
    gap = cos_sim_top1 - cos_sim_top2
    variance = float(np.var(similarities))

    uncertain = (gap < gap_threshold) or (variance > variance_threshold)

    if verbose:
        print(f"[C-EGSR] gap={gap:.4f}, variance={variance:.6f}, uncertain={uncertain}")

    top_k = 6 if uncertain else 3

    if verbose:
        print(f"[C-EGSR] Using top-{top_k} tables "
              f"({'proactive expansion' if uncertain else 'standard'})")

    schema_context, _ = self.schema_retriever.get_scoped_context(
        user_question, top_k=top_k
    )

    history = []
    error_feedback = None
    last_sql = None

    for iteration in range(1, self.max_iterations + 1):
        if verbose:
            print(f"\n--- Iteration {iteration} ---")

        prompt = self._build_prompt(user_question, schema_context, history, error_feedback)
        raw_output = self._call_llm(prompt)
        sql = self._clean_sql_output(raw_output)
        last_sql = sql

        if verbose:
            print(f"[Acting] Generated SQL: {sql}")

        success, result = self.execute_query(sql)

        if success:
            if verbose:
                print(f"[Observation] Success! Rows: {len(result['rows'])}")
            return {
                "success": True,
                "sql": sql,
                "result": result,
                "iterations_used": iteration,
                "cegsr_used_expanded": uncertain,
                "cegsr_top_k": top_k,
                "cos_sim_gap": round(gap, 4),
                "cos_sim_variance": round(variance, 6),
            }
        else:
            if verbose:
                print(f"[Observation] Error: {result}")
            error_feedback = result
            history.append(f"Attempt {iteration}: SQL='{sql}' FAILED: {result}")

    return {
        "success": False,
        "sql": last_sql,
        "result": error_feedback,
        "iterations_used": self.max_iterations,
        "cegsr_used_expanded": uncertain,
        "cegsr_top_k": top_k,
    }

        """
        C-EGSR: Confidence-Aware / Signal-Based EGSR (Novel Contribution)
        ==================================================================
        Key finding from cost model training:
        - cos_sim_gap (top1 - top2 similarity) accounts for 65.87% of signal
        - cos_sim_variance accounts for 34.13% of signal
        - LLM confidence, SQL complexity, question length = 0% discriminative power

        This means schema retrieval UNCERTAINTY (not model confidence) is the
        primary predictor of whether refinement will help.

        C-EGSR uses these two signals to PROACTIVELY decide whether to use
        expanded schema (top-6) BEFORE executing SQL - unlike EGSR which
        reacts to execution failure.

        Trigger condition:
        - cos_sim_gap < gap_threshold (top tables have similar scores = ambiguous)
        - OR cos_sim_variance > variance_threshold (high spread = uncertain retrieval)
        """
        from sklearn.metrics.pairwise import cosine_similarity as cos_sim
        import numpy as np

        # Compute retrieval signals
        query_embedding = self.schema_retriever.embedder.encode([user_question])
        similarities = cos_sim(query_embedding, self.schema_retriever.table_embeddings)[0]
        sorted_sims = sorted(similarities, reverse=True)

        cos_sim_top1 = float(sorted_sims[0]) if len(sorted_sims) > 0 else 0.0
        cos_sim_top2 = float(sorted_sims[1]) if len(sorted_sims) > 1 else 0.0
        gap = cos_sim_top1 - cos_sim_top2
        variance = float(np.var(similarities))

        # Decision: use top-3 or top-6?
        uncertain = (gap < gap_threshold) or (variance > variance_threshold)

        if verbose:
            print(f"[C-EGSR] cos_sim_gap={gap:.4f}, variance={variance:.6f}")
            print(f"[C-EGSR] Retrieval uncertain: {uncertain} "
                  f"(gap<{gap_threshold}: {gap < gap_threshold}, "
                  f"var>{variance_threshold}: {variance > variance_threshold})")

        top_k = 6 if uncertain else 3

        if verbose:
            print(f"[C-EGSR] Using top-{top_k} tables "
                  f"({'expanded - proactive refinement' if uncertain else 'standard'})")

        schema_context, ranked_tables = self.schema_retriever.get_scoped_context(
            user_question, top_k=top_k
        )

        # Standard ReAct loop with chosen schema
        history = []
        error_feedback = None
        last_sql = None

        for iteration in range(1, self.max_iterations + 1):
            if verbose:
                print(f"\n--- Iteration {iteration} ---")

            prompt = self._build_prompt(
                user_question, schema_context, history, error_feedback
            )
            raw_output = self._call_llm(prompt)
            sql = self._clean_sql_output(raw_output)
            last_sql = sql

            if verbose:
                print(f"[Acting] Generated SQL: {sql}")

            success, result = self.execute_query(sql)

            if success:
                if verbose:
                    print(f"[Observation] Success! Rows: {len(result['rows'])}")
                return {
                    "success": True,
                    "sql": sql,
                    "result": result,
                    "iterations_used": iteration,
                    "cegsr_used_expanded": uncertain,
                    "cegsr_top_k": top_k,
                    "cos_sim_gap": round(gap, 4),
                    "cos_sim_variance": round(variance, 6),
                }
            else:
                if verbose:
                    print(f"[Observation] Error: {result}")
                error_feedback = result
                history.append(f"Attempt {iteration}: SQL='{sql}' FAILED: {result}")

        return {
            "success": False,
            "sql": last_sql,
            "result": error_feedback,
            "iterations_used": self.max_iterations,
            "cegsr_used_expanded": uncertain,
            "cegsr_top_k": top_k,
        }
