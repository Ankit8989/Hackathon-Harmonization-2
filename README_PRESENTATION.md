# 🤖 Agentic AI Data Harmonization System

**Autonomous data pipeline that ingests, validates, and harmonizes datasets using Azure OpenAI and multi-agent orchestration.**

Built for **Hackathon 2024**.

---

## 🎯 The Problem

- Data comes from **many sources** (CSV, Excel, SPSS, JSON) with **inconsistent schemas**
- Manual validation and harmonization is **slow, error-prone, and doesn’t scale**
- Teams need **auditability** and **quality scores**, not just cleaned files

---

## 💡 Our Solution

An **agent-based pipeline** that:

1. **Validates** structure against a master schema and detects drift  
2. **Assesses** quality (anomalies, outliers, business rules)  
3. **Harmonizes** data to a canonical schema with LLM-assisted mapping  
4. **Reports** everything: audit trail, quality scores, and downloadable outputs  

All orchestrated by a **Supervisor Agent** with go/no-go decisions and a **Streamlit UI** for live demos.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SUPERVISOR ORCHESTRATOR AGENT                     │
│                  (Controls Flow, Go/No-Go Decisions)                 │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
│   STRUCTURAL    │   │   DATA QUALITY  │   │ HARMONIZATION   │
│   VALIDATION    │   │     AGENT       │   │     AGENT       │
└────────┬────────┘   └────────┬────────┘   └────────┬────────┘
         │                     │                     │
         └─────────────────────┼─────────────────────┘
                               ▼
                    ┌─────────────────────┐
                    │  LLM REASONING       │
                    │  (Azure OpenAI)      │
                    └─────────────────────┘
```

| Agent | Role |
|-------|------|
| **Supervisor** | Orchestrates pipeline, aggregates results, audit trail |
| **Structural Validation** | Schema check, column mapping, drift detection |
| **Data Quality** | Anomalies, outliers, business rules, quality score |
| **Harmonization** | Transformations, scale normalization, value standardization |
| **LLM Reasoning** | Semantic matching, ambiguity resolution |

---

## ✨ Key Features

| Capability | Description |
|------------|-------------|
| **Multi-format** | CSV, Excel (.xlsx, .xls), SPSS (.sav), JSON |
| **LLM-powered mapping** | Semantic column matching when schema differs |
| **Schema drift** | Missing/extra columns, type mismatches, confidence scores |
| **Quality scoring** | 0–100 score, issues by severity, recommendations |
| **Audit trail** | Full log of agent actions and decisions |
| **Streamlit UI** | Upload → Process → Results Dashboard → Reports |

---

## 🚀 Live Demo (How to Run)

### 1. Setup (one time)

```bash
# Clone/navigate to project
cd Hackathon

# Virtual environment (recommended)
python -m venv venv
venv\Scripts\activate

# Dependencies
pip install -r requirements.txt
```

Add a `.env` file with your Azure OpenAI key:

```env
AZURE_OPENAI_API_KEY=your-key-here
```

### 2. Run the app

```bash
streamlit run app.py
```

Then in the UI:

- **Tab 1 – Upload & Process:** Drop a CSV/Excel file (raw data).  
- **Tab 2 – Results Dashboard:** See the fully agentic pipeline in action (cleaning + harmonization).  

### 3. CLI (optional)

```bash
# Process a specific file
python main.py --input data/input/your_file.csv

# With custom rules
python main.py --input data.csv --rules "min 100 records; satisfaction 0-100"
```

---

## 📊 Outputs You Can Show

- **`data/output/harmonized.csv`** – Canonical, cleaned dataset  
- **`reports/final_audit.html`** – Interactive audit (quality, issues, agent summary)  
- **`reports/validation_report.json`** – Schema validation and column mappings  
- **`reports/dq_report.json`** – Data quality assessment  
- **`reports/harmonization_report.json`** – Transformations applied  

---

## 🛠️ Tech Stack

- **Python 3.10+**  
- **Azure OpenAI** (GPT-5.2) for reasoning and semantic mapping  
- **Streamlit** for the web UI  
- **Pandas** for data; **openpyxl** for Excel; **pyreadstat** for SPSS  

---

## 📌 Talking Points for Judges

1. **Agentic design** – Multiple specialized agents + supervisor; not a single monolith.  
2. **Production-ready** – Config-driven (master schema, mapping tables, business rules), CLI + UI.  
3. **Transparency** – Every step logged; quality scores and severity; downloadable reports.  
4. **Extensible** – New datasets = add file + optional schema/mappings; pipeline handles the rest.  
5. **LLM where it helps** – Column mapping and ambiguity resolution; validation and rules stay deterministic where possible.  

---

## 🧠 Agentic Enhancements Implemented

This hackathon version turns the app into a **closed-loop, autonomous data engineer**:

### 1. Agentic Data Cleaning Loop

- **Stats-only to LLM**: The LLM never sees full tables; it only receives:
  - Per-column stats: `dtype`, non-null/null counts, null %, min/max/mean (for numeric), top 5 values.
  - Data-quality summary: overall quality score, counts of blocking/fixable/ignorable issues.
  - Last error message (if previous code failed).
- **Code generation**: `LLMReasoningAgent.generate_agentic_code(...)` produces Python code that:
  - Assumes `df`, `pd`, `np` already exist.
  - Must actually modify `df` (no-op code is disallowed in the prompt).
  - Focuses on missing values and obvious range issues.
- **Local execution & validation** (`utils/agentic_loop.py`):
  - Code is executed locally on the DataFrame in a **restricted namespace**.
  - After each iteration, `DataQualityAgent` recomputes quality metrics.
  - A history log is recorded for each step: code, before/after scores, improvement, errors.
- **Self-correction**:
  - If code throws an error (e.g. bad column name), the full traceback is captured as `last_error`.
  - The next iteration’s prompt includes that traceback so the LLM can fix its own code.
- **Scalability**:
  - For large datasets (e.g. 80k+ rows), only **aggregated statistics** are sent to the LLM.
  - All heavy operations (imputation, transformations) happen locally on the full DataFrame.

### 2. Fallback Rule-Based Cleaning (Guaranteed Change)

- If, after the agentic iterations:
  - The number of missing values is **not lower than the original**, or
  - The LLM calls fail due to rate limits/capacity,
- Then a final **rule-based fallback** runs automatically:
  - Numeric columns: `NaN` → column **mean**.
  - Non-numeric columns: `NaN` → `"Unknown"`.
- This guarantees:
  - A visibly different, safer **cleaned dataset** even when the API is overloaded.
  - A final data-quality run so judges can see quality improvement metrics.

### 3. Fully Agentic Harmonization Flow

- **No manual sliders/toggles** in the main path:
  - Old per-column missing-value strategy UIs and extra harmonization workflows have been removed.
  - The only user actions are:
    1. Upload file.
    2. Click **“Let AI Clean the Data”** (agentic loop).
    3. Click **“Start Harmonization”**.
- **Input to harmonization**:
  - If `agentic_clean_df` exists → harmonization uses that as the source.
  - Otherwise falls back to the mapped/original DataFrame.
- **Harmonization itself**:
  - Still uses **deterministic, local transformations** (scale normalization, code standardization, etc.) to keep outputs stable.
  - The LLM’s job is to **prepare** a high-quality input (via the agentic loop), not to blindly rewrite all business rules.

### 4. Human-in-the-Loop Only When Needed

- The system surfaces human decisions **only in failure or ambiguity cases**:
  - If LLM or pipeline errors occur, tracebacks and iteration logs are shown in the UI.
  - Schema validation and data-quality agents still record issues, but the normal flow does not force manual choices.
- For demo:
  - You can show the **Agentic Loop Log**:
    - Iteration-by-iteration code,
    - Before/after quality scores,
    - Any errors and how the next iteration (or fallback) fixed them.

---

## 📁 Project Layout (High Level)

```
Hackathon/
├── app.py              # Streamlit UI
├── main.py             # CLI entry point
├── config.py           # Paths, Azure OpenAI config
├── agents/             # Supervisor, Validation, DQ, Harmonization, LLM
├── metadata/           # master_schema, mapping_tables
├── data/input/         # Drop files here (or upload in UI)
├── data/output/        # harmonized.csv, etc.
├── reports/            # JSON + final_audit.html
└── Input files/        # (Legacy) pre-saved harmonization outputs (no longer shown in UI)
```

---

For full API details, configuration, and troubleshooting, see **README.md**.
