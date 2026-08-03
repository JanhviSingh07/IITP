# IMPORTANT SETUP NOTES

## 1. Embedding Model Download (First time only)
`sentence-transformers` library `all-MiniLM-L6-v2` model ko pehli baar use
karne par automatically Hugging Face se download karegi (~90 MB). Iske liye
internet connection chahiye. Yeh ek baar download hoke local cache mein
(`~/.cache/huggingface/`) store ho jaata hai, phir offline bhi kaam karega.

Agar download slow/fail ho raha hai, manually pehle download kar sakte ho:
    python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

## 2. Gemini API Key Setup
Apna API key environment variable mein set karo (HARDCODE MAT KARNA):

Linux/Mac:
    export GEMINI_API_KEY="your-key-here"

Windows PowerShell:
    $env:GEMINI_API_KEY="your-key-here"

Windows CMD:
    set GEMINI_API_KEY=your-key-here

Free API key yahan se: https://aistudio.google.com/apikey

## 3. Verify Setup
Yeh command chalao verify karne ke liye sab kuch ready hai:
    python3 -c "import os; print('API Key set:', bool(os.environ.get('GEMINI_API_KEY')))"
