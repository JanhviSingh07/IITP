"""
result_validator.py
--------------------
GAP FIX: Paper ka auto-debug loop SIRF execution errors pe trigger hota hai.
Agar SQL successfully run ho jaaye lekin galat result de (wrong JOIN,
missing condition, wrong aggregation), agent ko pata nahi chalta -
kyunki koi "error" nahi aaya.

Yeh module ek EXTRA reasoning step add karta hai: SQL execute hone ke
baad, result ko LLM se validate karwate hain - "yeh result NL question
ka sahi jawab lagta hai kya?". Agar LLM khud "no" bole, toh retry trigger
hota hai - bina kisi database error ke bhi.

Yeh ReFoRCE paper ka "Self-Refinement" concept hai.
"""


class ResultValidator:
    def __init__(self, api_key, model_name="gemini-2.0-flash", provider="gemini"):
        self.model_name = model_name
        self.provider = provider

        if provider == "gemini":
            from google import genai
            self.client = genai.Client(api_key=api_key)
        elif provider == "groq":
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
        else:
            raise ValueError(f"Unknown provider: {provider}. Use 'gemini' or 'groq'.")

    def _build_validation_prompt(self, question, sql, result):
        """
        LLM se result ko critically review karwate hain.
        Important: hum poora result data nahi bhejte (token-heavy), sirf
        shape/summary bhejte hain - row count, sample rows, column names.
        """
        rows = result.get("rows", [])
        columns = result.get("columns", [])

        row_count = len(rows)
        sample_rows = rows[:5]  # sirf pehli 5 rows dikhate hain, poora data nahi

        prompt = f"""You are reviewing whether a SQL query's result correctly answers a natural
language question. Be critical - look for these common mistakes:
- Empty result when the question implies data should exist
- Single row when the question asks for multiple items (or vice versa)
- Wrong aggregation level (e.g., per-row when question asks "total")
- Missing a filter condition that the question implies
- Result that doesn't logically match the question's intent

Question: {question}

Generated SQL: {sql}

Result shape: {row_count} row(s), columns: {columns}
Sample rows: {sample_rows}

Respond in EXACTLY this format (no extra text):
VERDICT: <CORRECT or INCORRECT>
REASON: <one short sentence explaining why>"""
        return prompt

    def validate(self, question, sql, result):
        """
        Returns: (is_valid: bool, reason: str)
        """
        # Empty result ko hum direct heuristic se bhi catch kar sakte hain
        # (fast path, LLM call ki zaroorat nahi)
        rows = result.get("rows", [])
        if len(rows) == 0:
            return False, "Query returned zero rows - likely a missing/wrong filter condition"

        prompt = self._build_validation_prompt(question, sql, result)

        if self.provider == "gemini":
            from google.genai import types
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.0),
            )
            text = response.text.strip()
        else:  # groq
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            text = response.choices[0].message.content.strip()
        is_valid = "VERDICT: CORRECT" in text.upper()

        reason = "No reason provided"
        for line in text.split("\n"):
            if line.upper().startswith("REASON:"):
                reason = line.split(":", 1)[1].strip()
                break

        return is_valid, reason
