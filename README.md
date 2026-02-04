# 🤖 Agentic AI Data Harmonization System

> **Autonomous Data Pipeline powered by Azure OpenAI GPT-5.2**

An intelligent, agent-based system for automated data ingestion, validation, quality assessment, and harmonization. Built for the Hackathon 2024.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Azure OpenAI](https://img.shields.io/badge/Azure-OpenAI-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Agent Documentation](#agent-documentation)
- [Configuration](#configuration)
- [Plugging New Datasets](#plugging-new-datasets)
- [Reports](#reports)
- [Stretch Goals](#stretch-goals)
- [Troubleshooting](#troubleshooting)

---

## 🎯 Overview

The **Agentic AI Data Harmonization System** is an end-to-end intelligent pipeline that:

1. **Ingests** datasets from multiple formats (CSV, Excel, SPSS, JSON)
2. **Validates** structure against master metadata
3. **Assesses** data quality and detects anomalies
4. **Harmonizes** data to a canonical schema
5. **Uses LLM reasoning** for ambiguity resolution
6. **Orchestrates** via a Supervisor Agent
7. **Produces** comprehensive reports and audit trails

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
│     AGENT       │   │                 │   │                 │
└────────┬────────┘   └────────┬────────┘   └────────┬────────┘
         │                     │                     │
         └─────────────────────┼─────────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  LLM REASONING      │
                    │     AGENT           │
                    │  (Shared Utility)   │
                    └─────────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Azure OpenAI      │
                    │    GPT-5.2-chat     │
                    └─────────────────────┘
```

### Agents

| Agent | Responsibility |
|-------|---------------|
| **SupervisorOrchestratorAgent** | Controls pipeline flow, makes go/no-go decisions, aggregates results |
| **StructuralValidationAgent** | Validates columns against master schema, detects drift |
| **DataQualityAgent** | Performs quality checks, outlier detection, business rule validation |
| **HarmonizationAgent** | Applies transformations, normalizes scales, standardizes values |
| **LLMReasoningAgent** | Shared utility for intelligent inference and ambiguity resolution |

---

## ✨ Features

### Core Capabilities

- ✅ **Multi-format Support**: CSV, Excel (.xlsx, .xls), SPSS (.sav), JSON
- ✅ **Intelligent Column Mapping**: LLM-powered semantic matching
- ✅ **Schema Drift Detection**: Identifies missing, extra, and mismatched fields
- ✅ **Data Quality Analysis**: Missing values, outliers, duplicates, anomalies
- ✅ **Business Rule Validation**: Configurable rules with severity classification
- ✅ **Scale Normalization**: Automatic conversion (e.g., 1-10 → 0-100)
- ✅ **Value Standardization**: Mapping tables for categorical values
- ✅ **Audit Trail**: Complete logging of all agent actions and decisions

### Reports Generated

- 📄 `harmonized.csv` - Harmonized output data
- 📋 `validation_report.json` - Structural validation results
- 📊 `dq_report.json` - Data quality assessment
- 🔄 `harmonization_report.json` - Transformation details
- 📑 `final_audit.html` - Interactive HTML audit report

### Stretch Goals (Implemented)

- 🧠 **Vector Memory (FAISS)**: Schema history for auto-learning
- 📱 **Streamlit UI**: Modern web interface for demo

---

## 💻 Installation

### Prerequisites

- Python 3.10 or higher
- Azure OpenAI API access
- pip package manager

### Setup

1. **Clone/Download the project**

```bash
cd "C:\Users\ShindeAnk\Downloads\Agentic AI Traning\Hackathon"
```

2. **Create virtual environment (recommended)**

```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Configure Azure OpenAI API key**

Create a `.env` file in the project root:

```env
AZURE_OPENAI_API_KEY=your-azure-openai-api-key-here
```

Or set as environment variable:

```bash
# Windows CMD
set AZURE_OPENAI_API_KEY=your-api-key

# Windows PowerShell
$env:AZURE_OPENAI_API_KEY="your-api-key"

# Linux/Mac
export AZURE_OPENAI_API_KEY="your-api-key"
```

---

## 🚀 Quick Start

### One-Command Run

```bash
python main.py
```

This will:
1. Auto-detect input files in `data/input/`
2. Create sample data if none exists
3. Run the full harmonization pipeline
4. Generate all reports in `reports/`

### With Specific Input

```bash
python main.py --input "PMI CIV_NT_AQ_CASEDATA_202209.xlsx"
```

### Create Sample Data First

```bash
python main.py --create-sample
```

### Run Streamlit UI

```bash
streamlit run app.py
```

---

## 📖 Usage

### Command Line Interface

```bash
python main.py [OPTIONS]
```

**Options:**

| Option | Description |
|--------|-------------|
| `--input, -i` | Path to input data file |
| `--output, -o` | Path for harmonized output file |
| `--schema, -s` | Master schema name (default: `master_schema`) |
| `--mappings, -m` | Mapping tables name (default: `mapping_tables`) |
| `--rules, -r` | Business rules (semicolon-separated) |
| `--create-sample` | Create sample data and metadata |
| `--verbose, -v` | Enable verbose output |

**Examples:**

```bash
# Process specific file
python main.py --input data/survey.csv

# Specify output location
python main.py --input data.xlsx --output results/harmonized.csv

# Custom schema and rules
python main.py --input data.csv --schema custom_schema --rules "min 100 records;satisfaction range 1-10"
```

### Programmatic Usage

```python
from agents.supervisor_agent import SupervisorOrchestratorAgent

# Initialize supervisor
supervisor = SupervisorOrchestratorAgent()

# Run pipeline
result = supervisor.execute(
    input_file="data/input/raw.csv",
    master_schema_name="master_schema",
    mapping_tables_name="mapping_tables",
    business_rules=["Minimum 30 records required"],
    output_file="data/output/harmonized.csv"
)

# Check result
if result.success:
    print(f"Quality Score: {result.result['final_quality_score']}")
    print(f"Output: {result.result['output_file']}")
```

---

## 🤖 Agent Documentation

### StructuralValidationAgent

**Purpose**: Validates dataset structure against master metadata.

**Key Methods:**
- `execute(df, master_schema, metadata)` - Run validation
- `validate_single_column(source_column, master_schema)` - Validate one column

**Output**: `SchemaValidationResult` with:
- Column mappings with confidence scores
- Schema drift analysis
- Missing/extra columns
- Type mismatches

### DataQualityAgent

**Purpose**: Comprehensive data quality assessment.

**Key Methods:**
- `execute(df, master_schema, business_rules, column_mappings)` - Full analysis
- `quick_assessment(df)` - Fast quality check without LLM

**Output**: `DataQualityReport` with:
- Quality score (0-100)
- Column statistics
- Issues by severity (blocking/fixable/ignorable)
- LLM recommendations

### HarmonizationAgent

**Purpose**: Transforms data to canonical schema.

**Key Methods:**
- `execute(df, column_mappings, master_schema, mapping_tables, output_path)` - Full harmonization
- `harmonize_single_column(series, source_name, target_name, target_type, mapping_table)` - Single column

**Output**: `HarmonizationResult` with:
- Transformation summaries
- Before/after comparison
- Applied mappings
- Output file path

### SupervisorOrchestratorAgent

**Purpose**: Orchestrates entire pipeline.

**Key Methods:**
- `execute(input_file, master_schema_name, mapping_tables_name, business_rules, output_file)` - Run pipeline
- `run_quick_validation(input_file)` - Quick validation only

**Output**: `PipelineResult` with:
- All agent results
- Supervisor decisions
- Audit trail
- Final metrics

---

## ⚙️ Configuration

### Master Schema (`metadata/master_schema.yaml`)

Define your canonical schema:

```yaml
schema_name: my_schema
version: "1.0.0"
columns:
  - name: customer_id
    canonical_name: customer_id
    data_type: integer
    is_required: true
    is_key: true
    aliases:
      - cust_id
      - id
      - customer_number
    validation_rules:
      min_value: 1
```

### Mapping Tables (`metadata/mapping_tables.yaml`)

Define value mappings:

```yaml
region:
  "N": "North"
  "S": "South"
  "NORTH": "North"
  "north": "North"

satisfaction_scale:
  "1": "Very Dissatisfied"
  "2": "Dissatisfied"
  "3": "Neutral"
  "4": "Satisfied"
  "5": "Very Satisfied"
```

### Business Rules

Pass rules via CLI or programmatically:

```bash
python main.py --rules "min 100 records; satisfaction_score range 0-100; respondent_id required"
```

---

## 🔌 Plugging New Datasets

### Step 1: Place Your Data

Put your data file in `data/input/` directory. Supported formats:
- `.csv` - Comma-separated values
- `.xlsx`, `.xls` - Excel files
- `.sav` - SPSS files
- `.json` - JSON files

### Step 2: Update Master Schema (Optional)

If your data has different columns, update `metadata/master_schema.yaml`:

```yaml
columns:
  - name: your_new_column
    canonical_name: canonical_name
    data_type: string  # string, integer, float, categorical, datetime
    is_required: false
    aliases:
      - alternative_name
      - another_alias
```

### Step 3: Add Mapping Tables (Optional)

If you have categorical values that need standardization:

```yaml
your_column:
  "source_value_1": "Standard Value 1"
  "source_value_2": "Standard Value 2"
```

### Step 4: Run Pipeline

```bash
python main.py --input data/input/your_file.csv
```

The system will:
1. Auto-detect column types
2. Use LLM to infer mappings for unknown columns
3. Apply transformations based on target schema
4. Generate comprehensive reports

---

## 📊 Reports

### Final Audit Report (`reports/final_audit.html`)

Interactive HTML report containing:
- Pipeline summary with quality scores
- Agent execution results
- Data quality issues by severity
- LLM analysis and recommendations
- Complete audit trail
- Generated file list

### Validation Report (`reports/validation_report.json`)

```json
{
  "is_valid": true,
  "confidence_score": 0.92,
  "column_mappings": [...],
  "schema_drift": {
    "missing_columns": [],
    "extra_columns": ["unknown_col"],
    "drift_score": 0.05
  }
}
```

### Data Quality Report (`reports/dq_report.json`)

```json
{
  "overall_quality_score": 87.5,
  "is_acceptable": true,
  "total_issues": 5,
  "blocking_issues": [],
  "fixable_issues": [...],
  "recommendations": [...]
}
```

---

## 🎯 Stretch Goals

### Vector Memory (FAISS)

Auto-learning from previous harmonization runs:

```python
from utils.vector_memory import get_auto_learning_mapper

mapper = get_auto_learning_mapper()

# Learn from successful mappings
mapper.learn_from_mappings(successful_mappings)

# Get suggestions for new columns
suggestions = mapper.suggest_mappings(source_columns)
```

### Streamlit UI

Run the interactive web interface:

```bash
streamlit run app.py
```

Features:
- Drag-and-drop file upload
- Real-time processing with progress
- Interactive results dashboard
- Report preview and download
- Audit trail visualization

---

## 🔧 Troubleshooting

### Common Issues

**1. API Key Not Set**
```
ERROR: AZURE_OPENAI_API_KEY environment variable is not set!
```
Solution: Set the environment variable or create `.env` file.

**2. No Input Files Found**
```
No input files found. Please place a data file in data/input/
```
Solution: Place your data file in the `data/input/` directory or use `--input` flag.

**3. SPSS Files Not Loading**
```
pyreadstat not installed. SPSS file support disabled.
```
Solution: `pip install pyreadstat`

**4. Vector Memory Disabled**
```
Vector memory dependencies not available
```
Solution: `pip install faiss-cpu sentence-transformers`

### Debug Mode

For detailed logging:

```bash
python main.py --verbose
```

Check logs in `logs/harmonization.log`

---

## 📁 Project Structure

```
Hackathon/
├── main.py                    # CLI entry point
├── app.py                     # Streamlit UI
├── config.py                  # Configuration settings
├── requirements.txt           # Dependencies
├── README.md                  # This file
│
├── agents/                    # Agent implementations
│   ├── __init__.py
│   ├── base_agent.py          # Abstract base class
│   ├── llm_reasoning_agent.py # Shared LLM utility
│   ├── structural_validation_agent.py
│   ├── data_quality_agent.py
│   ├── harmonization_agent.py
│   └── supervisor_agent.py
│
├── models/                    # Pydantic schemas
│   ├── __init__.py
│   └── schemas.py
│
├── utils/                     # Utility modules
│   ├── __init__.py
│   ├── logger.py
│   ├── file_handlers.py
│   ├── report_generator.py
│   └── vector_memory.py       # FAISS integration
│
├── metadata/                  # Schema and mappings
│   ├── master_schema.yaml
│   └── mapping_tables.yaml
│
├── data/
│   ├── input/                 # Input data files
│   └── output/                # Harmonized outputs
│
├── reports/                   # Generated reports
│
└── logs/                      # Log files
```

---

## 📜 License

MIT License - Built for Hackathon 2024

---

## 🙏 Acknowledgments

- Azure OpenAI for GPT-5.2 capabilities
- FAISS for efficient vector similarity search
- Streamlit for the interactive UI framework

---

**Built with ❤️ using Agentic AI principles**


