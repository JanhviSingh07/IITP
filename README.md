# AskDB: LLM-Powered Text-to-SQL System

A **Natural Language to SQL (Text-to-SQL)** system developed during my internship at **IIT Patna**. The project converts user questions into executable SQL queries using **Large Language Models (LLMs)**, schema retrieval, and multiple validation layers.

The system is evaluated on the **Spider Benchmark** and includes utilities for schema retrieval, SQL generation, safety validation, and automated evaluation.

---

## ✨ Features

- Natural Language → SQL generation
- Schema-aware prompt construction
- Few-shot prompting
- SQL safety validation
- Result validation
- Spider 1 evaluation
- Spider 2 evaluation
- Error analysis utilities
- Automatic rescoring & retry pipeline

---

## 📁 Project Structure

```text
askdb_project/
│
├── src/
│   ├── sql_agent.py
│   ├── schema_retriever.py
│   ├── safety_layer.py
│   ├── result_validator.py
│   ├── few_shot_examples.py
│   └── mock_embedder_for_testing.py
│
├── eval/
│   ├── run_spider1.py
│   ├── run_spider2.py
│   └── error_analyzer.py
│
├── test_agent.py
├── config.py
├── requirements.txt
├── DOWNLOAD_DATASET.md
├── SETUP_NOTES.md
└── README.md
```

---

## ⚙️ System Pipeline

```text
Natural Language Question
          │
          ▼
  Schema Retrieval
          │
          ▼
 Prompt Construction
          │
          ▼
    LLM SQL Agent
          │
          ▼
    Safety Layer
          │
          ▼
   SQL Validation
          │
          ▼
 Execute SQL Query
          │
          ▼
    Return Result
```

---

## 🛠️ Technologies Used

- Python
- SQLite
- SQL
- JSON
- Large Language Models (LLMs)
- Prompt Engineering
- Spider Benchmark

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/JanhviSingh07/IITP.git
```

Move into the project directory:

```bash
cd IITP
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

---

## 📂 Dataset Setup

The Spider benchmark datasets are **not included** due to GitHub storage limitations.

Follow the instructions in **DOWNLOAD_DATASET.md** and place the datasets in:

```text
data/
├── spider1/
└── spider2/
```

---

## ▶️ Running the Project

Run the main agent:

```bash
python test_agent.py
```

Evaluate on Spider 1:

```bash
python eval/run_spider1.py
```

Evaluate on Spider 2:

```bash
python eval/run_spider2.py
```

Generate error analysis:

```bash
python eval/error_analyzer.py
```

---

## 📊 Evaluation

The project supports evaluation using the Spider benchmark with:

- SQL execution accuracy
- Error analysis
- Result rescoring
- Retry pipeline
- Confidence analysis

---

## 📦 Repository Contents

| Folder/File | Description |
|--------------|-------------|
| `src/` | Core implementation |
| `eval/` | Evaluation scripts |
| `requirements.txt` | Project dependencies |
| `DOWNLOAD_DATASET.md` | Dataset setup guide |
| `SETUP_NOTES.md` | Environment configuration |

---

## 🔮 Future Improvements

- RAG-based schema retrieval
- Multi-database support
- Fine-tuned LLM integration
- Interactive web interface
- Query explanation module
- Cost-aware SQL optimization

---

## 👩‍💻 Author

Janhvi Singh
B.Tech, Data Science & Engineering  
Manipal Institute of Technology  
Internship Project – IIT Patna