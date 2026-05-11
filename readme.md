# 💰 Personal Finance Analysis Tool

A web app for analyzing personal bank transactions — upload your spreadsheets, classify expenses, and get a clear picture of where your money goes.

🔗 **[Live Demo → personal-finance-analysis-ly.streamlit.app](https://personal-finance-analysis-ly.streamlit.app/)**

---

## 🎯 Why I built this

Most budgeting apps require you to connect your bank account or use a specific bank. I wanted something that works with any bank's CSV/XLSX export, gives full control over categories, and doesn't store any personal data in the cloud.

The tool grew through 3 major versions as the data pipeline became more sophisticated — each version introduced significant changes to how transactions are normalized, classified, and persisted across sessions.

---

## ✨ Features

**Data ingestion**
- Upload one or multiple CSV/XLSX transaction files simultaneously
- Automatic normalization and merging across different bank formats
- Removes internal transfers automatically

**Classification engine**
- Rule-based classification using two layers:
  - `pos_rules` — shared pattern matching (POS terminals, payroll, bill payments)
  - `specific_rules` — your personal rules, applied with priority
- Manual classification UI with Save and Undo support
- Unique classification for repeated patterns, specific classification by transaction ID

**Custom categories**
- Build your own category and sub-category tree
- Scope categories as Expense or Income

**Analysis views**
- Home dashboard with full transaction overview
- Dedicated Income and Expenses tabs
- Date range filters and search across all views

**Rules persistence**
- Export your personal rules as `specific_rules.json`
- Re-upload in future sessions to restore your classification logic
- Optional PostgreSQL backend for shared `pos_rules` across multiple users

---

## 🛠 Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-336791?style=flat-square&logo=postgresql&logoColor=white)

---

## 🚀 How to Use (Online)

1. Open the [live app](https://personal-finance-analysis-ly.streamlit.app/)
2. Upload your transaction files (`CSV` or `XLSX`)
3. Optionally upload a previous `specific_rules.json` to restore your categories
4. Click **Process** — the app normalizes, merges, and auto-classifies transactions
5. Review data in **Home**, **Income**, and **Expenses** tabs
6. Classify remaining items in **Uncategorized**
7. Build custom categories in **Custom Categories**
8. Export processed data and rules in **Save/Download**

---

## 🖥 How to Run Locally

```bash
git clone https://github.com/Lucasyuki01/Personal-Finance-Analysis.git
cd Personal-Finance-Analysis
pip install -r requirements.txt
streamlit run v3/streamlit_app.py
```

**Optional — enable shared PostgreSQL rules:**

```bash
# .env or Streamlit secrets
POS_RULES_DATABASE_URL=postgresql://USER:PASSWORD@HOST:5432/DBNAME
```

Without this, the app falls back to a local `pos_rules.json` file.

---

## 🗂 Project Structure

```
Personal-Finance-Analysis/
├── v1/                      # Initial version
├── v2/                      # Second iteration
├── v3/                      # Current version
│   ├── streamlit_app.py     # Main app entry point
│   ├── src/
│   │   ├── processing.py    # Data pipeline (normalize, merge, classify)
│   │   ├── rules.py         # Rule engine (pos_rules + specific_rules)
│   │   ├── pages.py         # Page rendering
│   │   ├── charts.py        # Visualizations
│   │   └── constants.py     # Shared constants
│   └── data/
│       ├── pos_rules.json   # Shared pattern rules (gitignored)
│       └── specific_rules.json  # Personal rules (gitignored)
├── requirements.txt
└── README.md
```

---

## 🔒 Privacy

No transaction data is stored on any server. All processing happens in-session — when you close the browser, your data is gone. Only classification rules (no amounts or descriptions) can optionally be persisted via PostgreSQL.

---

## 👨‍💻 Author

**Lucas Yuki Nishimoto**
[github.com/Lucasyuki01](https://github.com/Lucasyuki01) · [lucasnishimoto.dev](https://lucasnishimoto.dev)
