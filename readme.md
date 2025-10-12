# Personal Finance Analysis Dashboard

Interactive Streamlit dashboard for exploring Scotiabank transaction exports. The app cleans raw statements, classifies entries with POS rules/manual overrides, and visualises spending, income, and balance trends.

## Features
- **Automated classification** based on `config/pos_rules.csv`, with optional manual overrides in `config/manual_overrides.csv`.
- **Date, class, category, and sub-category filters** to focus on the period or segments you care about.
- **Key metrics** (total spent, total earned, net delta, median & average spending) updated in real time.
- **Visuals**:
  - Monthly expenses and earnings (bar charts).
  - Category distributions (pie charts with amounts and percentages).
  - Daily account balance trend (line chart).
  - Top 10 largest expenses and sub-category drilldown.
  - “Others” table to review uncategorised transactions.
- **One-click download** of the classified dataset.

## Project structure
```
config/
  pos_rules.csv            # POS classification rules
  manual_overrides.csv     # Specific transaction overrides (optional)
data/
  acc1.csv, acc2.csv       # Raw Scotiabank exports (excluded from Git)
src/
  nagui.py                 # Streamlit app entrypoint
  classify_transactions.py # Auxiliary classification helpers
notebooks/                 # Exploration notebooks (optional)
requirements.txt           # Python dependencies
```

## Setup
1. Install Python 3.10+.
2. Clone the repository and create a virtual environment:
   ```bash
   git clone <repo-url>
   cd personal-finance-analysis
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   source .venv/bin/activate  # macOS/Linux
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Place your Scotiabank CSV exports in the `data/` folder as `acc1.csv`, `acc2.csv` (or update the loader in `src/nagui.py`).
5. Optionally adjust classification rules in `config/pos_rules.csv` or seed manual overrides in `config/manual_overrides.csv`.

## Run the dashboard
```bash
streamlit run src/nagui.py
```
The app automatically loads datasets from `data/`, performs cleaning/classification, and renders the dashboard in your browser.

## Customising classifications
- Edit **`config/pos_rules.csv`** to add/update POS patterns (pattern, category, sub-category, priority, match type).
- Edit **`config/manual_overrides.csv`** for per-transaction adjustments (match by Date / Description / Sub-description / Amount).
- Restart the app after changes; Streamlit will re-cache the classified dataset.

## Data handling notes
- Raw statements under `data/` are ignored by Git (.gitignore) to keep private information out of the repository.
- A template `.env.example` can be provided if needed, but `.env` is ignored by default.

Enjoy tracking your finances!
