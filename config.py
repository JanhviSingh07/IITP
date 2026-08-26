"""
config.py
---------
Yahan saari settings aur API keys hoti hain.
IMPORTANT: Apni actual API key environment variable mein daalo,
kabhi bhi hardcode mat karo (security risk hai).

NOTE ON PROVIDER: Gemini API free tier India mein bina billing-account-link
(card verification) ke available nahi nikla (account/region eligibility
issue - dekho 429 RESOURCE_EXHAUSTED with limit:0 errors). Isliye is
project mein default provider GROQ rakha gaya hai - free, fast, no card
required, OpenAI-compatible API. Agar future mein Gemini access mil jaaye,
sirf PROVIDER = "gemini" karna hoga, baaki code waisa hi chalega.

Terminal mein yeh run karo (Windows PowerShell):
    $env:GROQ_API_KEY="your-key-here"
    $env:GEMINI_API_KEY="your-key-here"   (agar gemini use karna ho future mein)

Groq key yahan se milegi (free, no card): https://console.groq.com
Gemini key yahan se milegi: https://aistudio.google.com/apikey
"""

import os

# ---- Provider Selection ----
PROVIDER = "groq"   # "groq" ya "gemini"

# ---- API Keys ----
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# Active key, based on selected provider (use this in eval scripts)
ACTIVE_API_KEY = GROQ_API_KEY if PROVIDER == "groq" else GEMINI_API_KEY

# ---- Model Configuration ----
if PROVIDER == "groq":
    # Groq free models (as of testing) - fast inference, no card needed
    MODEL_SPIDER1 = "qwen/qwen3.6-27b"
    MODEL_SPIDER2 = "qwen/qwen3.6-27b"
    MODEL_FALLBACK_STRONG = "llama-3.1-8b-instant"
else:
    # Gemini model names (paper's original setup, kept for future use)
    MODEL_SPIDER1 = "qwen/qwen3.6-27b"
    MODEL_SPIDER2 = "qwen/qwen3.6-27b"
    MODEL_FALLBACK_STRONG = "gemini-2.5-pro"

# ---- Schema Retrieval Configuration ----
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # paper mein yahi use hua hai
TOP_K_TABLES = 5                        # kitne candidate tables retrieve karne hain

# ---- Agent Configuration ----
MAX_REACT_ITERATIONS = 6               # ReAct loop kitni baar retry kare (paper mein kam tha, hum badha rahe)
SELF_CONSISTENCY_SAMPLES = 3           # kitne SQL candidates generate karke vote karein

# ---- Paths ----
# NOTE: Actual extracted Spider 1.0 structure is data/spider1/dev.json,
# data/spider1/tables.json, data/spider1/database/<db_name>/<db_name>.sqlite
# (no extra "spider" subfolder, unlike some other distributions of this dataset)
SPIDER1_DATA_DIR = "data/spider1"
SPIDER2_DATA_DIR = "data/spider2"
RESULTS_DIR = "results"

