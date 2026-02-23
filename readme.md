# Personal Finance Analysis

## Online User Tutorial (Step by Step)
1. Open the online app link in your browser.
2. On the landing page, upload your transaction spreadsheets (`CSV` or `XLSX`):
   - You can upload one or multiple files at the same time.
3. (Optional) Upload your previous `specific_rules.json`:
   - Use this when you want to reuse your own categories/sub-categories from earlier sessions.
4. Click `Process`:
   - The app will normalize, merge, and classify transactions automatically.
5. Review data in `Home`, `Income`, and `Expenses`:
   - Use date filters to focus on a period.
   - Use search boxes to find transactions quickly.
6. Classify uncategorized items in `Uncategorized`:
   - `Unique classification` for repeated patterns (POS/Payroll/Bill Payment/Service Charge groups).
   - `Specific classification (by ID)` for single transactions (for example withdrawals).
   - Use `Save` to apply labels, or `Undo` to revert the last change.
7. Create or extend personal category trees in `Custom Categories`:
   - Choose scope (`Expense` or `Income`).
   - Add a new category with sub-categories, or append sub-categories to an existing category.
8. Save outputs in `Save/Download files`:
   - Download processed transactions.
   - Download `specific_rules.json` (important for your personal rules backup and reuse).
   - Download `pos_rules.json` (backup/export of shared pattern rules).
9. Reuse your rules later:
   - In a future session, upload your saved `specific_rules.json` before processing.
   - Your personal category logic will be restored for that session.

## What the Tool Does
- Normalizes and merges multiple transaction spreadsheets.
- Creates/updates columns (`Source`, `Category`, `Sub-Category`, `Profit`, `ID`).
- Removes internal transfers.
- Classifies transactions using:
  - `pos_rules` (fallback)
  - `specific_rules` (priority)
- Supports manual editing with `Save` and `Undo`.

## Project Structure
```text
v1/                          # previous versions
v2/                          # previous versions
v3/                          # current version (main app)
  streamlit_app.py
  requirements.txt
  README.md
  src/
    processing.py
    rules.py
    pages.py
    charts.py
    constants.py
  data/
    pos_rules.json
    specific_rules.json
```

## Rules Persistence
- `specific_rules`: local/session flow + download/re-upload.
- `pos_rules`: can be centralized to collect labels from multiple users.

### Recommended for Deploy (crowdsourced labels)
Set this environment variable:
```bash
POS_RULES_DATABASE_URL=postgresql://USER:PASSWORD@HOST:5432/DBNAME
```

With this setup:
- the app uses Postgres for `pos_rules` (shared across users);
- without it, fallback is local `v3/data/pos_rules.json`.

## GitHub Preparation
- Sensitive/runtime files are already ignored in `.gitignore`:
  - `.env`, `.streamlit/secrets.toml`, caches, virtual environments
  - intermediate outputs
  - `v3/data/pos_rules.json` and `v3/data/specific_rules.json`
- Before pushing:
  1. review `git status`
  2. confirm no personal data is included in local spreadsheets
  3. commit only code and documentation

## Deploy
For Streamlit Cloud:
1. Set `Main file path` to `v3/streamlit_app.py`.
2. Ensure deployment installs `v3/requirements.txt`.
3. (Optional, recommended) configure `POS_RULES_DATABASE_URL`.
