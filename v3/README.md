# Personal Finance Analysis (v3)

## Overview
A Streamlit app for personal finance analysis. Upload one or more transaction spreadsheets (CSV/XLSX) plus an optional `specific_rules.json`. The app normalizes and merges data, enriches columns, applies rules, and provides dashboards plus editing with Save/Undo.

## How to Run
```bash
streamlit run streamlit_app.py
```

## Expected Input Columns
Required columns in each spreadsheet:
- `Date`
- `Amount`
- `Description`

Optional columns:
- `Sub-description`
- `filter`
- `Source`
- `Category`
- `Sub-Category`
- `ID`

If `Sub-description` is missing, it will be created and filled with an empty string.
If `Category` or `Sub-Category` is missing, it will be created and empty cells filled with `none`.

## Pipeline Summary (Order)
1. Per-file normalization (drop `filter`, ensure `Source`, `Category`, `Sub-Category`, `Sub-description`, normalize types)
2. Merge all files
3. `Profit` column: `1` if `Amount > 0`, else `0`
4. Drop internal transfers where `Description` is `customer transfer cr.` or `customer transfer dr.`
5. Ensure unique `ID` (generate random 6-char hex for missing/duplicates)
6. Apply `pos_rules` (fallback only, does not overwrite existing values)
7. Apply `specific_rules` (priority, can overwrite)

## Rules and Priority
- `pos_rules.json` applies only to rows where `Description == "pos purchase"` (normalized).
- `specific_rules.json` has priority over `pos_rules`:
  - `by_id` rules apply first and overwrite Category/Sub-Category for that ID.
  - `by_pattern` rules apply next and overwrite Category/Sub-Category by key.

### pos_rules.json format
```json
{
  "0|pos purchase|dollorama": {
    "category": "Shopping",
    "sub_category": "Clothing",
    "updated_at": "2026-02-11T00:00:00Z"
  }
}
```

### specific_rules.json format
```json
{
  "by_id": {
    "abc123": {
      "category": "Fees",
      "sub_category": "Bank Fees",
      "updated_at": "2026-02-11T00:00:00Z"
    }
  },
  "by_pattern": {
    "0|withdrawal|atm": {
      "category": "Fees",
      "sub_category": "Service Fees",
      "updated_at": "2026-02-11T00:00:00Z"
    }
  }
}
```

## Undo Behavior
Each Save action (POS batch or Specific by ID) pushes an undo record containing:
- affected row IDs
- previous Category/Sub-Category values
- prior rule state (or absence)

Undo reverts the dataframe edits and restores/removes the corresponding rule entry.

## Categories and Subcategories
Defaults are defined in `src/constants.py`. If you need a custom taxonomy, update:
- `INCOME_CATEGORIES`
- `EXPENSE_CATEGORIES`
- `CATEGORY_TO_SUBCATEGORIES`

## Deployment Notes
Rules are stored in `data/pos_rules.json` and `data/specific_rules.json`. On Streamlit Cloud, filesystem persistence may be limited, so the app always allows downloading rules for re-use.
