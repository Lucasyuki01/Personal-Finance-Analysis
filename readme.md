# Personal Finance Analysis Dashboard

Interactive Streamlit dashboard for exploring Scotiabank transaction exports. The app cleans raw statements, classifies entries with POS rules/manual overrides, and visualises spending, income, and balance trends.

## Features
- **Automated classification** based on `config/pos_rules.csv`, with optional manual overrides in `config/manual_overrides.csv`.
- **Flexible filters** for date range, class, category, and sub-category.
- **Key metrics** (total spent, total earned, net delta, median & average spending) updated live.
- **Visuals**: monthly expenses and earnings, category breakdowns, daily balance trend, Top 10 expenses, and an "Others" review table.
- **One-click download** of the classified dataset.

## Project structure
```
config/
  pos_rules.csv            # POS classification rules
  manual_overrides.csv     # Transaction-specific overrides (optional)
data/                      # Raw Scotiabank exports (ignored by Git)
reserve/                   # Local backups / prototypes (ignored by Git)
src/
  app.py                   # Streamlit app entrypoint
requirements.txt           # Python dependencies
.gitignore                 # Keeps sensitive/local files out of Git
readme.md                  # Project documentation
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
4. Place your Scotiabank CSV exports in the `data/` folder as `acc1.csv`, `acc2.csv` (or adjust the loader in `src/app.py`).
5. Optionally tweak `config/pos_rules.csv` or populate `config/manual_overrides.csv` with transaction overrides.

## Run the dashboard
```bash
streamlit run src/app.py
```
The app loads the datasets, applies cleaning/classification, and serves the dashboard in your browser.

## Customising classifications
- Edit **`config/pos_rules.csv`** to refine POS matching (pattern, category, sub-category, priority, match type).
- Edit **`config/manual_overrides.csv`** for per-transaction adjustments (matched by Date / Description / Sub-description / Amount).
- Restart Streamlit after changes so the data cache is refreshed.

## Data handling notes
- Raw statements in `data/` and any personal backups in `reserve/` are ignored by Git to keep private information out of the repository.
- `.env` files are ignored; provide a `.env.example` if you need to document environment variables.

Enjoy tracking your finances!
