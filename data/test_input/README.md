# Test input folder

Use these files in the app to verify upload, cleaning, harmonization, and **Multi-Source Harmonization**.

---

## Input data (main upload or multi-source)

| File | Description |
|------|-------------|
| **missing_input.csv** | 20 rows × 9 columns with missing values. Use to test data quality / imputation. |
| **correct_input.csv** | Same schema, 20 rows × 9 columns, no missing values. Use to confirm app works end-to-end. |
| **test_historical.csv** | 10 rows, same schema. Use as **historical** in Multi-Source Harmonization. |
| **test_current.csv** | 10 rows, same schema. Use as **current** in Multi-Source Harmonization. |

---

## Supporting files (mapping, validation, master, stats)

| File | Description |
|------|-------------|
| **test_master_metadata.yaml** | Master schema (column names, types, aliases). Use as **Master metadata** in Multi-Source. |
| **test_mapping_table.yaml** | Value mappings (channel, purchase_intent, gender). Use as **Mapping table**. |
| **test_validation_rules.yaml** | Validation rules (min/max scores, allowed age/country). Use as **Validation rules**. |
| **test_descriptive_stats.csv** | Baseline descriptive stats. Use as **Descriptive stats** (optional). |

---

## How to test

### 1. Single-file flow (Upload & Process)

1. Run: `streamlit run app.py`
2. **Upload & Process** tab → upload `missing_input.csv` or `correct_input.csv`
3. Optionally run **Build supporting files** (or use metadata from disk)
4. Optionally run agentic cleaning, then **Start Harmonization**
5. Check **Results Dashboard** for harmonized output

### 2. Multi-Source Harmonization (with test_input supporting files)

1. Run: `streamlit run app.py`
2. Open **Multi-Source Harmonization** (expandable section)
3. **Uncheck** “Auto-detect files from data/input and metadata” so you can pick test_input files
4. Upload (or point paths to `data/test_input/`):
   - **Master metadata file** → `test_master_metadata.yaml`
   - **Historical data** → `test_historical.csv`
   - **Current data** → `test_current.csv`
   - **Validation rules file** → `test_validation_rules.yaml`
   - **Mapping table file** → `test_mapping_table.yaml`
   - **Descriptive stats file** (optional) → `test_descriptive_stats.csv`
5. Click **Run Multi-Source Harmonization**
6. Check results and reports (e.g. `data/output/`, `reports/`)

**Tip:** If the app expects paths under `data/input/`, copy the test_input files into `data/input/` and use Auto-detect, or upload each file via the file pickers so they are saved to input and paths are set automatically.
