# Dataset Download Instructions (APNE LOCAL MACHINE PE RUN KARNA HAI)
# ====================================================================
# Note: Yeh commands Claude ke sandbox mein nahi chal sakte kyunki
# Google Drive whitelist mein nahi hai. Apne laptop/PC pe yeh karo.

# ---------- SPIDER 1.0 ----------
# Step 1: Browser mein jao: https://yale-lily.github.io/spider
# Step 2: "Spider Dataset" link pe click karo (Google Drive khulega)
# Step 3: spider.zip download karo (~600 MB)
# Step 4: Unzip karo apne project ke data/spider1/ folder mein

# Terminal se bhi try kar sakte ho (gdown chahiye):
pip install gdown
# Spider dataset ka Google Drive file ID (README se confirm karo, change ho sakta hai):
gdown --id 1TqleXec_OykOYFREKKtschzY29dUcVAQ -O spider.zip
unzip spider.zip -d data/spider1/

# Verify karo yeh files hone chahiye:
# data/spider1/spider/train_spider.json
# data/spider1/spider/dev.json
# data/spider1/spider/tables.json
# data/spider1/spider/database/   <- yeh folder mein saare .sqlite files honge


# ---------- SPIDER 2.0 ----------
# Yeh GitHub pe hi available hai (Google Drive ki zaroorat nahi)
git clone https://github.com/xlang-ai/Spider2.git
cd Spider2/spider2-lite

# Spider 2.0-lite ke SQLite databases aur questions yahan honge:
# Spider2/spider2-lite/resource/databases/   <- sqlite dbs
# Spider2/spider2-lite/spider2-lite.jsonl     <- questions + ground truth

# README follow karo unke repo ke andar, kyunki kabhi kabhi extra setup
# scripts chalane padte hain (database download script).


# ---------- VERIFY KARNE KE LIYE (yeh command chalao dataset milne ke baad) ----------
# python3 -c "import json; data = json.load(open('data/spider1/spider/dev.json')); print(f'Total dev questions: {len(data)}'); print(data[0])"
