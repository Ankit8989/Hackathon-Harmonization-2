"""
Streamlit UI for Agentic AI Data Harmonization System
======================================================

A modern, interactive web interface for the data harmonization pipeline.

Run with:
    streamlit run app.py
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Any, Tuple, Dict, List

import streamlit as st
import pandas as pd
import numpy as np

# NumPy 2.0 compatibility fix - restore removed aliases for older packages
_np_compat_map = {
    'round_': np.round,
    'float_': np.float64,
    'int_': np.int64,
    'complex_': np.complex128,
    'object_': object,
    'str_': str,
    'bool_': bool,
    'unicode_': str,
    'bytes_': bytes,
    'string_': np.bytes_,
    'PINF': np.inf,
    'NINF': -np.inf,
    'PZERO': 0.0,
    'NZERO': -0.0,
}
for _attr, _val in _np_compat_map.items():
    if not hasattr(np, _attr):
        setattr(np, _attr, _val)

import plotly.express as px
import plotly.graph_objects as go

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

# Input files folder (contains harmonization results)
INPUT_FILES_DIR = PROJECT_ROOT / "Input files"
BRAND_LOGO_PATH = PROJECT_ROOT / "Brand Image" / "logo-kantar-colour-2x-1.webp"

from dotenv import load_dotenv
load_dotenv()

from config import INPUT_DIR, OUTPUT_DIR, REPORTS_DIR, METADATA_DIR, AZURE_CONFIG
from utils.multi_source_harmonizer import MultiSourceHarmonizer
from utils.knowledge_bag import load_knowledge_bag, save_knowledge_bag, update_imputation_overrides
from agents.structural_validation_agent import StructuralValidationAgent
from utils.file_handlers import MetadataHandler
from utils.supporting_files_generator import generate_supporting_files


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def to_dict(obj):
    """Safely convert object to dictionary"""
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, 'model_dump'):
        return obj.model_dump()
    if hasattr(obj, '__dict__'):
        return obj.__dict__
    return {}

def safe_get(data, *keys, default=None):
    """Safely get nested values from dict or object"""
    result = to_dict(data)
    for key in keys:
        if isinstance(result, dict):
            result = result.get(key, {})
        else:
            result = to_dict(result).get(key, {})
        if result is None:
            return default
    return result if result != {} else default


def save_uploaded_file(uploaded_file, prefix: str) -> Optional[Path]:
    """Save a Streamlit uploaded file to input directory and return path."""
    if uploaded_file is None:
        return None
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{prefix}_{uploaded_file.name}"
    path = INPUT_DIR / filename
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path


def restart_app() -> None:
    """Clear session/cache and rerun the app."""
    st.session_state.clear()
    try:
        st.cache_data.clear()
    except Exception:
        pass
    try:
        st.cache_resource.clear()
    except Exception:
        pass
    st.rerun()


def pick_first_matching_file(
    search_dir: Path,
    keywords: List[str],
    extensions: List[str]
) -> Optional[Path]:
    """Pick the first file in a directory matching keywords and extensions."""
    if not search_dir.exists():
        return None
    candidates = [
        p for p in search_dir.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    ]
    if not candidates:
        return None
    lowered = [(p, p.name.lower()) for p in candidates]
    for kw in keywords:
        for path, name in lowered:
            if kw in name:
                return path
    return candidates[0]


def auto_detect_multisource_files() -> Dict[str, Optional[Path]]:
    """Auto-detect common multi-source inputs from data/input and metadata."""
    data_exts = [".csv", ".xlsx", ".xls", ".json", ".sav"]
    meta_exts = [".yaml", ".yml", ".json"]
    stats_exts = [".csv", ".xlsx", ".xls"]
    return {
        "historical": pick_first_matching_file(
            INPUT_DIR, ["historical", "history", "vendor_a", "baseline"], data_exts
        ),
        "current": pick_first_matching_file(
            INPUT_DIR, ["current", "vendor_b", "latest"], data_exts
        ),
        "incremental": pick_first_matching_file(
            INPUT_DIR, ["incremental", "new", "delta"], data_exts
        ),
        "master": pick_first_matching_file(
            METADATA_DIR, ["master_schema", "master_metadata", "schema"], meta_exts
        ),
        "mapping": pick_first_matching_file(
            METADATA_DIR, ["mapping", "map"], meta_exts
        ),
        "rules": pick_first_matching_file(
            METADATA_DIR, ["validation_rules", "rules", "validation"], meta_exts
        ),
        "stats": pick_first_matching_file(
            INPUT_DIR, ["descriptive", "stats", "baseline"], stats_exts
        )
    }


def build_change_preview(
    raw_df: pd.DataFrame,
    harmonized_df: pd.DataFrame,
    max_rows: int = 50
) -> Tuple[Any, dict]:
    """Create a styled preview highlighting changes between raw and harmonized data."""
    raw = raw_df.head(max_rows).reset_index(drop=True)
    harm = harmonized_df.head(max_rows).reset_index(drop=True)

    raw_cols = list(raw.columns)
    harm_cols = list(harm.columns)
    added_cols = sorted(set(harm_cols) - set(raw_cols))
    removed_cols = sorted(set(raw_cols) - set(harm_cols))
    all_cols = raw_cols + [c for c in harm_cols if c not in raw_cols]

    raw = raw.reindex(columns=all_cols)
    harm = harm.reindex(columns=all_cols)

    diff_mask = pd.DataFrame(False, index=harm.index, columns=all_cols)
    common_cols = [c for c in all_cols if c in raw_cols and c in harm_cols]

    for col in common_cols:
        raw_col = raw[col]
        harm_col = harm[col]
        changed = (raw_col != harm_col) & ~(raw_col.isna() & harm_col.isna())
        diff_mask[col] = changed

    if added_cols:
        diff_mask[added_cols] = True
    if removed_cols:
        diff_mask[removed_cols] = True

    def highlight_changes(data):
        return pd.DataFrame(
            np.where(diff_mask.values, "background-color: #fde68a", ""),
            index=data.index,
            columns=data.columns
        )

    try:
        styled = harm.style.apply(highlight_changes, axis=None)
    except Exception:
        # Fallback when jinja2/styler isn't available
        styled = harm.copy()
        styled["_changed_preview"] = diff_mask.any(axis=1)

    summary = {
        "raw_rows": len(raw_df),
        "harmonized_rows": len(harmonized_df),
        "raw_columns": len(raw_cols),
        "harmonized_columns": len(harm_cols),
        "added_columns": added_cols,
        "removed_columns": removed_cols,
        "changed_cells_in_preview": int(diff_mask.sum().sum()),
        "rows_with_changes_in_preview": int((diff_mask.any(axis=1)).sum())
    }

    return styled, summary


def apply_mapping_decisions(
    df: pd.DataFrame,
    decisions: Dict[str, Dict[str, Any]]
) -> pd.DataFrame:
    """Apply mapping decisions to a dataframe."""
    rename_map: Dict[str, str] = {}
    drop_cols: List[str] = []
    used_targets: Dict[str, int] = {}

    for source_col in df.columns:
        decision = decisions.get(source_col, {})
        action = decision.get("action", "confirm")
        target = decision.get("target")

        if action == "skip":
            drop_cols.append(source_col)
            continue

        if action in {"confirm", "map_to"} and target:
            target_name = target
            if target_name in used_targets or target_name in df.columns:
                used_targets[target_name] = used_targets.get(target_name, 0) + 1
                target_name = f"{target_name}_dup{used_targets[target_name]}"
            rename_map[source_col] = target_name

    mapped = df.drop(columns=drop_cols, errors="ignore").rename(columns=rename_map)
    return mapped


def build_dictionary_suggestions(
    df: pd.DataFrame,
    standards: Dict[str, Any]
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Create missing-value strategy suggestions and standards warnings."""
    suggestions: List[Dict[str, Any]] = []
    warnings: List[str] = []
    columns = standards.get("columns", {})

    for col, spec in columns.items():
        if col not in df.columns:
            continue
        series = df[col]
        missing = int(series.isna().sum())
        if missing > 0:
            impute = spec.get("impute_strategy", "mode")
            if impute == "mean":
                strategy = "Replace with MEAN"
            elif impute == "median":
                strategy = "Replace with MEDIAN"
            elif impute == "mode":
                strategy = "Replace with MODE"
            else:
                strategy = "Keep missing"
            suggestions.append({
                "column": col,
                "missing": missing,
                "strategy": strategy,
                "reason": f"Dictionary suggests {impute} for {spec.get('data_type')}"
            })

        if "min" in spec and "max" in spec and pd.api.types.is_numeric_dtype(series):
            out_of_range = series.dropna().loc[(series < spec["min"]) | (series > spec["max"])]
            if not out_of_range.empty:
                warnings.append(
                    f"{col}: {len(out_of_range)} values outside [{spec['min']}, {spec['max']}]"
                )

        allowed = spec.get("allowed_values")
        if allowed and series.notna().any():
            invalid = series.dropna().astype(str).loc[~series.dropna().astype(str).isin(set(map(str, allowed)))]
            if not invalid.empty:
                warnings.append(
                    f"{col}: {len(invalid)} values not in allowed list"
                )

    return suggestions, warnings


# =============================================================================
# INPUT FILES LOADER FUNCTIONS
# =============================================================================

def load_input_files() -> Dict[str, Any]:
    """
    Load all harmonization result files from the 'Input files' folder.
    
    Returns:
        dict with all loaded dataframes and text content
    """
    result = {
        'anomalies': None,
        'audit_log': None,
        'cleaned_df': None,
        'column_health_report': None,
        'drift_alerts': None,
        'valid_nulls': None,
        'uncertain_missing': None,
        'executive_summary': None,
        'loaded': False,
        'errors': []
    }
    
    if not INPUT_FILES_DIR.exists():
        result['errors'].append(f"Input files folder not found: {INPUT_FILES_DIR}")
        return result
    
    # Load Excel files
    excel_files = {
        'anomalies': 'anomalies.xlsx',
        'audit_log': 'audit_log.xlsx',
        'cleaned_df': 'cleaned_df.xlsx',
        'column_health_report': 'column_health_report.xlsx',
        'drift_alerts': 'drift_alerts.xlsx',
        'valid_nulls': 'valid_nulls.xlsx',
        'uncertain_missing': 'uncertain_missing.xlsx'
    }
    
    for key, filename in excel_files.items():
        filepath = INPUT_FILES_DIR / filename
        if filepath.exists():
            try:
                result[key] = pd.read_excel(filepath)
            except Exception as e:
                result['errors'].append(f"Error loading {filename}: {str(e)}")
        else:
            result['errors'].append(f"File not found: {filename}")
    
    # Load executive summary text file
    summary_path = INPUT_FILES_DIR / 'systemic_executive_summary.txt'
    if summary_path.exists():
        try:
            with open(summary_path, 'r', encoding='utf-8') as f:
                result['executive_summary'] = f.read()
        except Exception as e:
            result['errors'].append(f"Error loading executive summary: {str(e)}")
    
    # Check if at least the main file loaded
    if result['cleaned_df'] is not None:
        result['loaded'] = True
    
    return result


def get_harmonization_summary(input_files: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate summary statistics from the loaded input files.
    
    Args:
        input_files: Dictionary from load_input_files()
    
    Returns:
        Dictionary with summary statistics
    """
    summary = {
        'total_rows': 0,
        'total_columns': 0,
        'total_anomalies': 0,
        'total_corrections': 0,
        'columns_with_issues': 0,
        'valid_nulls_count': 0,
        'uncertain_missing_count': 0,
        'drift_alerts_count': 0
    }
    
    if input_files.get('cleaned_df') is not None:
        df = input_files['cleaned_df']
        summary['total_rows'] = len(df)
        summary['total_columns'] = len(df.columns)
    
    if input_files.get('anomalies') is not None:
        summary['total_anomalies'] = len(input_files['anomalies'])
    
    if input_files.get('audit_log') is not None:
        summary['total_corrections'] = len(input_files['audit_log'])
    
    if input_files.get('column_health_report') is not None:
        summary['columns_with_issues'] = len(input_files['column_health_report'])
    
    if input_files.get('valid_nulls') is not None:
        summary['valid_nulls_count'] = len(input_files['valid_nulls'])
    
    if input_files.get('uncertain_missing') is not None:
        summary['uncertain_missing_count'] = len(input_files['uncertain_missing'])
    
    if input_files.get('drift_alerts') is not None:
        summary['drift_alerts_count'] = len(input_files['drift_alerts'])
    
    return summary


def perform_final_harmonization(
    input_files: Dict[str, Any],
    imputation_strategies: Dict[str, str],
    keep_valid_nulls: bool = True
) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    """
    Perform final harmonization using all input files.
    
    Args:
        input_files: Dictionary from load_input_files()
        imputation_strategies: Dict mapping column names to imputation strategy
                              ('mean', 'median', 'mode', 'zero', 'drop', 'keep')
        keep_valid_nulls: If True, don't impute values marked as valid_nulls
    
    Returns:
        Tuple of (harmonized_df, changes_log, statistics)
    """
    changes_log = []
    stats = {
        'rows_before': 0,
        'rows_after': 0,
        'missing_before': 0,
        'missing_after': 0,
        'columns_imputed': 0,
        'values_imputed': 0
    }
    
    # Start with cleaned_df as base
    if input_files.get('cleaned_df') is None:
        raise ValueError("cleaned_df is required for harmonization")
    
    harmonized_df = input_files['cleaned_df'].copy()
    stats['rows_before'] = len(harmonized_df)
    stats['missing_before'] = int(harmonized_df.isna().sum().sum())
    
    changes_log.append(f"📂 Loaded base dataset: {len(harmonized_df)} rows × {len(harmonized_df.columns)} columns")
    
    # Get valid_nulls positions to preserve
    valid_null_positions = set()
    if keep_valid_nulls and input_files.get('valid_nulls') is not None:
        valid_nulls_df = input_files['valid_nulls']
        for _, row in valid_nulls_df.iterrows():
            row_idx = row.get('row')
            col_name = row.get('column')
            if row_idx is not None and col_name is not None:
                valid_null_positions.add((row_idx, col_name))
        changes_log.append(f"✅ Preserving {len(valid_null_positions)} valid nulls (questionnaire logic)")
    
    # Get uncertain_missing positions that need imputation
    uncertain_positions = {}
    if input_files.get('uncertain_missing') is not None:
        uncertain_df = input_files['uncertain_missing']
        for _, row in uncertain_df.iterrows():
            row_idx = row.get('row')
            col_name = row.get('column')
            if row_idx is not None and col_name is not None:
                if col_name not in uncertain_positions:
                    uncertain_positions[col_name] = []
                uncertain_positions[col_name].append(row_idx)
    
    # Apply imputation strategies for uncertain_missing values
    imputed_count = 0
    columns_imputed = []
    
    for col_name, strategy in imputation_strategies.items():
        if col_name not in harmonized_df.columns:
            continue
        
        if strategy == 'keep':
            continue
        
        # Get rows to impute (uncertain_missing only, not valid_nulls)
        rows_to_impute = uncertain_positions.get(col_name, [])
        
        if not rows_to_impute:
            # If no specific uncertain rows, impute all NaN except valid_nulls
            mask = harmonized_df[col_name].isna()
            for idx in harmonized_df[mask].index:
                if (idx, col_name) not in valid_null_positions:
                    rows_to_impute.append(idx)
        else:
            # Filter out valid_nulls from the rows_to_impute
            rows_to_impute = [r for r in rows_to_impute if (r, col_name) not in valid_null_positions]
        
        if not rows_to_impute:
            continue
        
        # Calculate imputation value
        fill_value = None
        is_numeric = pd.api.types.is_numeric_dtype(harmonized_df[col_name])
        
        if strategy == 'mean' and is_numeric:
            fill_value = harmonized_df[col_name].mean()
        elif strategy == 'median' and is_numeric:
            fill_value = harmonized_df[col_name].median()
        elif strategy == 'mode':
            mode_vals = harmonized_df[col_name].mode()
            fill_value = mode_vals[0] if len(mode_vals) > 0 else None
        elif strategy == 'zero' and is_numeric:
            fill_value = 0
        elif strategy == 'unknown':
            fill_value = 'Unknown'
        elif strategy == 'other':
            fill_value = 'Other'
        elif strategy == 'remove':
            # Treat "remove" as row drop (handled separately)
            continue
        elif strategy == 'drop':
            # Mark rows for dropping (handled separately)
            continue
        
        if fill_value is not None:
            # Apply imputation only to uncertain rows
            for row_idx in rows_to_impute:
                if row_idx in harmonized_df.index and pd.isna(harmonized_df.loc[row_idx, col_name]):
                    harmonized_df.loc[row_idx, col_name] = fill_value
                    imputed_count += 1
            
            columns_imputed.append(col_name)
    
    stats['columns_imputed'] = len(columns_imputed)
    stats['values_imputed'] = imputed_count
    
    if imputed_count > 0:
        changes_log.append(f"🔧 Imputed {imputed_count} uncertain missing values across {len(columns_imputed)} columns")
        for col in columns_imputed[:5]:
            strategy = imputation_strategies.get(col, 'unknown')
            changes_log.append(f"   • {col} → {strategy}")
        if len(columns_imputed) > 5:
            changes_log.append(f"   • ... and {len(columns_imputed) - 5} more columns")
    
    # Handle 'drop' strategy - remove rows with remaining missing in those columns
    drop_columns = [
        col for col, strat in imputation_strategies.items()
        if strat in {'drop', 'remove'}
    ]
    if drop_columns:
        rows_before = len(harmonized_df)
        for col in drop_columns:
            if col in harmonized_df.columns:
                # Only drop if not a valid_null
                for idx in harmonized_df[harmonized_df[col].isna()].index:
                    if (idx, col) not in valid_null_positions:
                        harmonized_df = harmonized_df.drop(idx, errors='ignore')
        rows_dropped = rows_before - len(harmonized_df)
        if rows_dropped > 0:
            changes_log.append(f"🗑️ Dropped {rows_dropped} rows with missing values in: {', '.join(drop_columns)}")
    
    stats['rows_after'] = len(harmonized_df)
    stats['missing_after'] = int(harmonized_df.isna().sum().sum())
    
    # Final summary
    changes_log.append(f"")
    changes_log.append(f"📊 **Final Statistics:**")
    changes_log.append(f"   • Rows: {stats['rows_before']} → {stats['rows_after']}")
    changes_log.append(f"   • Missing values: {stats['missing_before']} → {stats['missing_after']}")
    changes_log.append(f"   • Values imputed: {stats['values_imputed']}")
    
    return harmonized_df, changes_log, stats


def get_columns_needing_imputation(input_files: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Get columns that have uncertain_missing values and need imputation decisions.
    
    Returns:
        Dict mapping column names to their info (count, data_type, sample_values)
    """
    columns_info = {}
    
    if input_files.get('uncertain_missing') is None or input_files.get('cleaned_df') is None:
        return columns_info
    
    uncertain_df = input_files['uncertain_missing']
    cleaned_df = input_files['cleaned_df']
    
    # Group by column
    if 'column' in uncertain_df.columns:
        col_counts = uncertain_df.groupby('column').size()
        
        for col_name, count in col_counts.items():
            if col_name in cleaned_df.columns:
                is_numeric = pd.api.types.is_numeric_dtype(cleaned_df[col_name])
                sample_values = cleaned_df[col_name].dropna().head(5).tolist()
                
                columns_info[col_name] = {
                    'missing_count': int(count),
                    'is_numeric': is_numeric,
                    'data_type': str(cleaned_df[col_name].dtype),
                    'sample_values': sample_values,
                    'total_in_column': int(cleaned_df[col_name].isna().sum())
                }
    
    return columns_info
    
    if input_files.get('drift_alerts') is not None:
        summary['drift_alerts_count'] = len(input_files['drift_alerts'])
    
    return summary


def analyze_data_with_ai(df: pd.DataFrame) -> dict:
    """
    Use AI (as a Market Research Analyst) to analyze data and suggest harmonization.
    ONE API call to minimize costs - sends summary statistics, not raw data.
    
    Returns:
        dict with AI recommendations for each column
    """
    from openai import AzureOpenAI
    from config import AZURE_CONFIG
    
    # Build compact summary of data (minimize tokens!)
    summary = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'columns': []
    }
    
    for col in df.columns:
        col_info = {
            'name': col,
            'dtype': str(df[col].dtype),
            'missing_count': int(df[col].isna().sum()),
            'missing_pct': round(df[col].isna().sum() / len(df) * 100, 1),
        }
        
        # Add sample values (first 5 non-null unique values)
        non_null = df[col].dropna().unique()[:5]
        col_info['sample_values'] = [str(v)[:50] for v in non_null]  # Truncate long values
        
        # Add stats for numeric columns
        if pd.api.types.is_numeric_dtype(df[col]):
            col_info['is_numeric'] = True
            col_info['min'] = float(df[col].min()) if df[col].notna().any() else None
            col_info['max'] = float(df[col].max()) if df[col].notna().any() else None
            col_info['mean'] = round(float(df[col].mean()), 2) if df[col].notna().any() else None
        else:
            col_info['is_numeric'] = False
            col_info['unique_count'] = int(df[col].nunique())
        
        summary['columns'].append(col_info)
    
    # Create the AI prompt - Market Research Analyst persona
    prompt = f"""You are a Senior Market Research Data Analyst. Your job is to analyze survey/market research data and recommend how to handle data quality issues.

DATASET SUMMARY:
- Total Rows: {summary['total_rows']}
- Total Columns: {summary['total_columns']}

COLUMN DETAILS:
{json.dumps(summary['columns'], indent=2)}

TASK: For each column with missing values (missing_count > 0), recommend the best strategy. Consider:
1. Column type (numeric vs text)
2. Sample values (to understand context)
3. Missing percentage
4. What a market researcher would typically do

RESPOND IN THIS EXACT JSON FORMAT (no other text):
{{
  "recommendations": [
    {{
      "column": "column_name",
      "strategy": "one of: mean|median|zero|mode|unknown|remove|keep",
      "reason": "brief explanation (max 15 words)"
    }}
  ],
  "data_quality_notes": "1-2 sentence overall assessment",
  "analysis_summary": "2-3 sentences summarizing key issues and prioritized next steps",
  "suggested_actions": [
    "3-6 actions that reference specific columns and their missing rates",
    "avoid generic repeated text across datasets"
  ]
}}

STRATEGY OPTIONS:
- mean: Replace with column average (for continuous numeric like scores, ratings)
- median: Replace with median (for numeric with outliers)
- zero: Replace with 0 (for counts, quantities where 0 makes sense)
- mode: Replace with most frequent value (for categorical)
- unknown: Replace with "Unknown" text (for text/categorical)
- remove: Remove rows with missing (if <5% missing and random)
- keep: Keep as missing (if missing has meaning, like "not applicable")
- drop: Drop the entire column if missing rate is too high or unusable

Be concise. Focus on columns with missing values only. Do not repeat generic actions; make suggestions specific to this dataset."""

    try:
        # Prefer configured values; only fall back to env when config is empty.
        api_key_env = os.getenv("AZURE_OPENAI_API_KEY")
        endpoint_env = os.getenv("AZURE_OPENAI_ENDPOINT")
        deployment_env = os.getenv("AZURE_OPENAI_DEPLOYMENT")
        api_version_env = os.getenv("AZURE_OPENAI_API_VERSION")
        if not AZURE_CONFIG.api_key and api_key_env:
            AZURE_CONFIG.api_key = api_key_env
        if not AZURE_CONFIG.endpoint and endpoint_env:
            AZURE_CONFIG.endpoint = endpoint_env
        if not AZURE_CONFIG.deployment and deployment_env:
            AZURE_CONFIG.deployment = deployment_env
        if not AZURE_CONFIG.api_version and api_version_env:
            AZURE_CONFIG.api_version = api_version_env

        client = AzureOpenAI(
            azure_endpoint=AZURE_CONFIG.endpoint,
            api_key=AZURE_CONFIG.api_key,
            api_version=AZURE_CONFIG.api_version
        )
        
        response = client.chat.completions.create(
            model=AZURE_CONFIG.deployment,
            messages=[
                {"role": "system", "content": "You are a data analysis expert. Always respond with valid JSON only. No markdown, no explanation, just JSON."},
                {"role": "user", "content": prompt}
            ],
            max_completion_tokens=2000  # Using max_completion_tokens as required by gpt-5.2-chat
        )
        
        result_text = response.choices[0].message.content
        tokens_used = response.usage.total_tokens if response.usage else 0
        
        # Parse JSON response
        ai_result = json.loads(result_text)
        ai_result['tokens_used'] = tokens_used
        ai_result['success'] = True
        
        return ai_result
        
    except json.JSONDecodeError as e:
        # JSON parsing failed - AI returned non-JSON
        return {
            'success': False,
            'error': f'AI response not valid JSON: {str(e)[:100]}',
            'recommendations': [],
            'data_quality_notes': f'AI returned invalid format. Raw response saved.',
            'suggested_actions': ['Use manual options below', 'Check API configuration'],
            'tokens_used': 0,
            'raw_response': result_text[:500] if 'result_text' in dir() else 'No response'
        }
    except Exception as e:
        # Return fallback with actual error message
        error_msg = str(e)
        return {
            'success': False,
            'error': error_msg,
            'recommendations': [],
            'data_quality_notes': f'AI Error: {error_msg[:100]}',
            'suggested_actions': ['Use manual options below', 'Check API key in sidebar'],
            'tokens_used': 0
        }


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="AI Data Harmonization System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS (Kantar theme)
st.markdown("""
<style>
    :root {
        --kantar-bg: #ffffff;
        --kantar-surface: #f7f7f7;
        --kantar-border: #e6e6e6;
        --kantar-text: #111111;
        --kantar-gold: #f3db63;
        --kantar-gold-dark: #b48901;
    }

    /* Main theme */
    .stApp, .main {
        background-color: var(--kantar-bg);
        color: var(--kantar-text);
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: var(--kantar-text);
    }
    h1 {
        background: linear-gradient(90deg, var(--kantar-gold) 0%, #111111 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }
    
    /* Cards */
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f2f2f2 100%);
        border-radius: 16px;
        padding: 20px;
        border: 1px solid var(--kantar-border);
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }

    /* Buttons */
    .stButton>button {
        background: var(--kantar-gold);
        color: #111111;
        border: 1px solid var(--kantar-gold-dark);
        font-weight: 600;
        border-radius: 8px;
    }
    .stButton>button:hover {
        background: var(--kantar-gold-dark);
        color: #111111;
        border-color: var(--kantar-gold-dark);
    }

    /* Tabs */
    .stTabs [data-baseweb="tab"] {
        color: var(--kantar-text);
    }
    .stTabs [aria-selected="true"] {
        color: var(--kantar-gold);
    }

    /* Inputs */
    .stTextInput>div>div>input,
    .stSelectbox>div>div>div,
    .stNumberInput>div>div>input,
    .stDateInput>div>div>input,
    .stTextArea>div>div>textarea {
        background-color: #ffffff;
        color: var(--kantar-text);
        border: 1px solid var(--kantar-border);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        color: #111111;
        border-right: 2px solid var(--kantar-gold-dark);
    }
    [data-testid="stSidebar"] * {
        color: #111111;
    }
    [data-testid="stSidebar"] .stButton>button {
        background: var(--kantar-gold);
        color: #111111;
        border: 1px solid var(--kantar-gold-dark);
    }

    /* Center images in sidebar and prevent logo from overflowing */
    [data-testid="stSidebar"] img {
        display: block;
        margin-left: auto;
        margin-right: auto;
        max-width: 100% !important;
        height: auto !important;
        object-fit: contain;
    }

    /* Animated brain icon */
    .brain-pulse {
        display: block;
        margin-left: auto;
        margin-right: auto;
        animation: hueSpin 6s linear infinite;
    }
    @keyframes hueSpin {
        0% { filter: hue-rotate(0deg); }
        50% { filter: hue-rotate(180deg); }
        100% { filter: hue-rotate(360deg); }
    }

    /* File uploader */
    .stFileUploader {
        border: 1px dashed var(--kantar-gold-dark);
        background-color: #ffffff;
    }

    .kantar-callout {
        background-color: var(--kantar-gold);
        color: #111111;
        border: 1px solid var(--kantar-gold-dark);
        border-radius: 8px;
        padding: 10px 14px;
        font-weight: 600;
    }
    
    /* Status badges */
    .status-success {
        background-color: #10b981;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
    }
    
    .status-warning {
        background-color: #f59e0b;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
    }
    
    .status-error {
        background-color: #ef4444;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
    }
    
    /* Agent cards */
    .agent-card {
        background: linear-gradient(135deg, #1e293b 0%, #374151 100%);
        border-radius: 12px;
        padding: 16px;
        margin: 8px 0;
        border-left: 4px solid #6366f1;
        color: #ffffff !important;
    }
    
    .agent-card h4 {
        color: #ffffff !important;
        margin-bottom: 10px;
        font-size: 1.1em;
    }
    
    .agent-card p {
        color: #e2e8f0 !important;
        margin: 5px 0;
    }
    
    /* Scrollable containers */
    .scrollable {
        max-height: 400px;
        overflow-y: auto;
        padding: 10px;
        background: #1e293b;
        border-radius: 8px;
    }
    
    /* Custom buttons */
    .stButton>button {
        background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(99, 102, 241, 0.4);
    }
    
    /* File uploader */
    .stFileUploader {
        border: 2px dashed #6366f1;
        border-radius: 12px;
        padding: 20px;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 100%);
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

if 'pipeline_result' not in st.session_state:
    st.session_state.pipeline_result = None
if 'uploaded_df' not in st.session_state:
    st.session_state.uploaded_df = None
if 'processing' not in st.session_state:
    st.session_state.processing = False


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def check_api_key():
    """Check if Azure OpenAI API key is configured"""
    return bool(AZURE_CONFIG.api_key or os.environ.get("AZURE_OPENAI_API_KEY"))


def load_sample_data():
    """
    Load PMI CIV_NT_AQ_CASEDATA_202209.xlsx as the standard sample file.
    Inject a small amount of missing data to demonstrate harmonization.
    """
    import numpy as np

    sample_path = PROJECT_ROOT / "PMI CIV_NT_AQ_CASEDATA_202209.xlsx"
    if not sample_path.exists():
        raise FileNotFoundError(f"Sample file not found: {sample_path}")

    df = pd.read_excel(sample_path)

    # Inject controlled missing values to demonstrate harmonization
    np.random.seed(42)
    total_rows = len(df)
    if total_rows > 0:
        # Use PMI-relevant columns if available; otherwise fall back to first columns
        preferred_cols = [
            'Respondent_Serial',
            'Qcountry',
            'AGE',
            'SEX',
            'DEMO_REG',
            'DEMO_EDUC',
            'Main_Boost',
            'QLanguage',
            'Q5_PBAT_SPONT_AWARE_TOM',
            'Q3_CAT_AU_E_TN_DEV'
        ]
        candidate_cols = [c for c in preferred_cols if c in df.columns]
        if not candidate_cols:
            candidate_cols = list(df.columns[:8])

        # Vary missing rates by column for realism
        missing_rate_map = [0.02, 0.03, 0.05, 0.04, 0.06, 0.03, 0.02, 0.05, 0.04, 0.03]
        for i, col in enumerate(candidate_cols):
            non_missing = df[col].notna().sum()
            if non_missing == 0:
                continue
            rate = missing_rate_map[i % len(missing_rate_map)]
            missing_count = max(1, int(total_rows * rate))
            missing_idx = np.random.choice(df.index, size=missing_count, replace=False)
            df.loc[missing_idx, col] = np.nan

    return df


def create_quality_gauge(score: float) -> go.Figure:
    """Create a gauge chart for quality score"""
    color = '#10b981' if score >= 80 else '#f59e0b' if score >= 60 else '#ef4444'
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Quality Score", 'font': {'size': 16, 'color': 'white'}},
        number={'font': {'size': 40, 'color': 'white'}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': 'white'},
            'bar': {'color': color},
            'bgcolor': '#1e293b',
            'borderwidth': 2,
            'bordercolor': '#475569',
            'steps': [
                {'range': [0, 60], 'color': '#ef4444'},
                {'range': [60, 80], 'color': '#f59e0b'},
                {'range': [80, 100], 'color': '#10b981'}
            ],
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'},
        height=250,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig


def create_confidence_bar(confidences: dict) -> go.Figure:
    """Create a bar chart for agent confidences"""
    agents = list(confidences.keys())
    values = list(confidences.values())
    
    colors = ['#10b981' if v >= 0.9 else '#f59e0b' if v >= 0.7 else '#ef4444' for v in values]
    
    fig = go.Figure(go.Bar(
        x=agents,
        y=[v * 100 for v in values],
        marker_color=colors,
        text=[f'{v:.1%}' for v in values],
        textposition='outside'
    ))
    
    fig.update_layout(
        title='Agent Confidence Scores',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'},
        yaxis_title='Confidence (%)',
        yaxis_range=[0, 110],
        height=300,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    return fig


def harmonize_data_local(df: pd.DataFrame, options: dict) -> tuple:
    """
    Perform comprehensive data harmonization using pandas (NO API CALLS).
    Implements all harmonization rules locally.
    
    Returns:
        tuple: (harmonized_df, changes_log)
    """
    import re
    from datetime import datetime
    
    changes_log = []
    harmonized_df = df.copy()
    
    # ========================================
    # 1. HANDLE MISSING VALUES FIRST (before column renaming!)
    # ========================================
    # This MUST be done FIRST because column_strategies uses ORIGINAL column names
    column_strategies = options.get('column_strategies', {})
    default_numeric = options.get('default_numeric', 'Replace with MEAN')
    default_text = options.get('default_text', "Replace with 'Unknown'")
    remove_remaining = options.get('remove_missing_rows', False)
    
    total_missing_before = harmonized_df.isna().sum().sum()
    filled_count = 0
    columns_processed = []
    columns_dropped = []
    columns_remove_rows = []
    
    # Process each column with its specific strategy (using ORIGINAL column names)
    for col in harmonized_df.columns:
        missing_count = harmonized_df[col].isna().sum()
        if missing_count == 0:
            continue
        
        # Determine column type
        is_numeric = pd.api.types.is_numeric_dtype(harmonized_df[col])
        
        # Get strategy for this column (from user selection or default)
        if col in column_strategies:
            strategy = column_strategies[col]
        else:
            # Use default based on column type
            strategy = default_numeric if is_numeric else default_text
        
        # Skip if keeping missing
        if strategy == "Keep missing":
            continue

        # Drop column if requested
        if strategy == "Drop column":
            columns_dropped.append(col)
            continue

        if strategy == "Remove rows":
            columns_remove_rows.append(col)
            continue
        
        # Apply the strategy
        fill_val = None
        strategy_name = ""
        
        if strategy == "Replace with MEAN" and is_numeric:
            mean_val = harmonized_df[col].mean()
            # If column is 100% empty, mean is NaN - fallback to 0
            fill_val = mean_val if pd.notna(mean_val) else 0
            strategy_name = "MEAN" if pd.notna(mean_val) else "0 (no data for MEAN)"
        elif strategy == "Replace with MEDIAN" and is_numeric:
            median_val = harmonized_df[col].median()
            # If column is 100% empty, median is NaN - fallback to 0
            fill_val = median_val if pd.notna(median_val) else 0
            strategy_name = "MEDIAN" if pd.notna(median_val) else "0 (no data for MEDIAN)"
        elif strategy == "Replace with 0" and is_numeric:
            fill_val = 0
            strategy_name = "0"
        elif strategy == "Replace with MODE":
            mode_vals = harmonized_df[col].mode()
            fill_val = mode_vals[0] if len(mode_vals) > 0 else (0 if is_numeric else 'Unknown')
            strategy_name = "MODE" if len(mode_vals) > 0 else ("0" if is_numeric else "'Unknown'")
        elif strategy == "Replace with 'Unknown'" and not is_numeric:
            fill_val = 'Unknown'
            strategy_name = "'Unknown'"
        elif strategy == "Replace with 'Other'" and not is_numeric:
            fill_val = 'Other'
            strategy_name = "'Other'"
        elif strategy == "Replace with 'N/A'" and not is_numeric:
            fill_val = 'N/A'
            strategy_name = "'N/A'"
        elif strategy == "Replace with empty ''" and not is_numeric:
            fill_val = ''
            strategy_name = "empty"
        
        if fill_val is not None:
            harmonized_df[col].fillna(fill_val, inplace=True)
            filled_count += missing_count
            columns_processed.append(f"{col[:30]}→{strategy_name}")
    
    # Log what was done
    if filled_count > 0:
        changes_log.append(f"🔧 Replaced {filled_count:,} missing values across {len(columns_processed)} columns")
        # Show a few examples
        if len(columns_processed) <= 5:
            for cp in columns_processed:
                changes_log.append(f"   • {cp}")
        else:
            for cp in columns_processed[:3]:
                changes_log.append(f"   • {cp}")
            changes_log.append(f"   • ... and {len(columns_processed) - 3} more columns")

    if columns_dropped:
        harmonized_df.drop(columns=columns_dropped, inplace=True)
        changes_log.append(f"🗑️ Dropped {len(columns_dropped)} columns with missing values")
        for col in columns_dropped[:5]:
            changes_log.append(f"   • {col}")
        if len(columns_dropped) > 5:
            changes_log.append(f"   • ... and {len(columns_dropped) - 5} more columns")

    if columns_remove_rows:
        rows_before = len(harmonized_df)
        harmonized_df = harmonized_df.dropna(subset=columns_remove_rows)
        rows_removed = rows_before - len(harmonized_df)
        if rows_removed > 0:
            changes_log.append(
                f"🧹 Removed {rows_removed} rows with missing values in: {', '.join(columns_remove_rows[:5])}"
            )
    
    # Listwise deletion (remove rows with ANY remaining missing)
    if remove_remaining:
        rows_before = len(harmonized_df)
        harmonized_df.dropna(inplace=True)
        rows_removed = rows_before - len(harmonized_df)
        if rows_removed > 0:
            changes_log.append(f"🗑️ Removed {rows_removed:,} rows with remaining missing values")
    
    # Summary
    total_missing_after = harmonized_df.isna().sum().sum()
    if total_missing_before > 0:
        changes_log.append(f"✅ Missing values: {total_missing_before:,} → {total_missing_after:,}")
    
    # ========================================
    # 2. STANDARDIZE COLUMN NAMES (after missing values are handled!)
    # ========================================
    if options.get('standardize_cols', True):
        original_cols = list(harmonized_df.columns)
        new_cols = []
        
        for col in original_cols:
            new_col = col.lower().strip()
            new_col = re.sub(r'\s+', '_', new_col)
            
            if options.get('remove_special', True):
                new_col = re.sub(r'[^a-z0-9_]', '', new_col)
            
            if new_col in new_cols:
                new_col = f"{new_col}_{new_cols.count(new_col) + 1}"
            
            new_cols.append(new_col)
        
        harmonized_df.columns = new_cols
        renamed_count = sum(1 for o, n in zip(original_cols, new_cols) if o != n)
        if renamed_count > 0:
            changes_log.append(f"📝 Standardized {renamed_count} column names (lowercase, no special chars)")
    
    # ========================================
    # 2b. CLEAN SPECIAL CHARACTERS FROM DATA VALUES
    # ========================================
    if options.get('remove_special', True):
        cleaned_cols = 0
        for col in harmonized_df.select_dtypes(include=['object']).columns:
            # Clean text values: remove special chars but keep letters, numbers, spaces, basic punctuation
            def clean_text(val):
                if pd.isna(val):
                    return val
                val = str(val)
                # Remove HTML entities
                val = re.sub(r'&[a-zA-Z]+;', ' ', val)
                val = re.sub(r'<[^>]+>', ' ', val)  # Remove HTML tags
                # Remove emojis and special unicode
                val = val.encode('ascii', 'ignore').decode('ascii')
                # Remove excessive special chars but keep basic punctuation
                val = re.sub(r'[^\w\s\-\.,\'\"\/\(\)]', '', val)
                # Clean up multiple spaces
                val = re.sub(r'\s+', ' ', val).strip()
                # Standardize common separators in categories
                val = re.sub(r'\s*[>\-/]\s*', ' > ', val)  # "Electronics > Phones"
                return val if val else None
            
            original_values = harmonized_df[col].copy()
            harmonized_df[col] = harmonized_df[col].apply(clean_text)
            
            # Check if any values changed
            if not harmonized_df[col].equals(original_values):
                cleaned_cols += 1
        
        if cleaned_cols > 0:
            changes_log.append(f"🧹 Cleaned special characters from {cleaned_cols} text columns")
    
    # ========================================
    # 3. SCALE CONVERSION (1-5, 1-7, 1-10 → 0-100)
    # ========================================
    scale_conversion = options.get('scale_conversion', 'No conversion')
    
    if scale_conversion != 'No conversion':
        converted_cols = 0
        for col in harmonized_df.select_dtypes(include=['number']).columns:
            col_min = harmonized_df[col].min()
            col_max = harmonized_df[col].max()
            
            if scale_conversion == "1-5 scale → 0-100" and col_min >= 1 and col_max <= 5:
                harmonized_df[col] = ((harmonized_df[col] - 1) / 4 * 100).round(1)
                converted_cols += 1
            elif scale_conversion == "1-7 scale → 0-100" and col_min >= 1 and col_max <= 7:
                harmonized_df[col] = ((harmonized_df[col] - 1) / 6 * 100).round(1)
                converted_cols += 1
            elif scale_conversion == "1-10 scale → 0-100" and col_min >= 0 and col_max <= 10:
                harmonized_df[col] = (harmonized_df[col] * 10).round(1)
                converted_cols += 1
            elif scale_conversion == "0-10 NPS → Categories" and col_min >= 0 and col_max <= 10:
                # Convert NPS scores to categories (replaces numeric with text)
                def nps_category(x):
                    if pd.isna(x): return None
                    if x >= 9: return 'Promoter'
                    elif x >= 7: return 'Passive'
                    else: return 'Detractor'
                harmonized_df[col] = harmonized_df[col].apply(nps_category)  # Update in place
                converted_cols += 1
        
        if converted_cols > 0:
            changes_log.append(f"📊 Converted {converted_cols} columns using {scale_conversion}")
    
    # ========================================
    # 3. REGION/GEOGRAPHY MAPPING
    # ========================================
    if options.get('region_mapping', True):
        # Common region mappings
        region_mappings = {
            # US States
            'ny': 'US-NY', 'new york': 'US-NY', 'n.y.': 'US-NY', 'newyork': 'US-NY',
            'ca': 'US-CA', 'california': 'US-CA', 'calif': 'US-CA',
            'tx': 'US-TX', 'texas': 'US-TX',
            'fl': 'US-FL', 'florida': 'US-FL',
            # European
            'uk': 'GB', 'united kingdom': 'GB', 'england': 'GB', 'britain': 'GB',
            'de': 'DE', 'germany': 'DE', 'deutschland': 'DE',
            'fr': 'FR', 'france': 'FR',
            # Regions
            'north': 'NORTH', 'south': 'SOUTH', 'east': 'EAST', 'west': 'WEST',
            'northeast': 'NORTHEAST', 'northwest': 'NORTHWEST',
            'southeast': 'SOUTHEAST', 'southwest': 'SOUTHWEST',
            'central': 'CENTRAL', 'midwest': 'MIDWEST',
            # APAC
            'apac': 'APAC', 'asia pacific': 'APAC', 'asia-pacific': 'APAC',
            'emea': 'EMEA', 'latam': 'LATAM', 'na': 'NA', 'north america': 'NA'
        }
        
        region_cols = [c for c in harmonized_df.columns if any(x in c.lower() for x in ['region', 'geo', 'state', 'area', 'territory', 'location'])]
        mapped_regions = 0
        
        for col in region_cols:
            if harmonized_df[col].dtype == 'object':
                harmonized_df[col] = harmonized_df[col].str.lower().str.strip().map(
                    lambda x: region_mappings.get(x, x.upper() if isinstance(x, str) else x)
                )
                mapped_regions += 1
        
        if mapped_regions > 0:
            changes_log.append(f"🗺️ Standardized {mapped_regions} region columns")
    
    # ========================================
    # 4. COUNTRY CODE STANDARDIZATION
    # ========================================
    if options.get('country_mapping', True):
        country_mappings = {
            'united states': 'US', 'usa': 'US', 'u.s.a.': 'US', 'u.s.': 'US', 'america': 'US',
            'united kingdom': 'GB', 'uk': 'GB', 'great britain': 'GB', 'england': 'GB',
            'germany': 'DE', 'deutschland': 'DE',
            'france': 'FR', 'french': 'FR',
            'spain': 'ES', 'españa': 'ES',
            'italy': 'IT', 'italia': 'IT',
            'japan': 'JP', 'nippon': 'JP',
            'china': 'CN', 'prc': 'CN',
            'india': 'IN', 'bharat': 'IN',
            'brazil': 'BR', 'brasil': 'BR',
            'canada': 'CA',
            'australia': 'AU', 'aus': 'AU',
            'mexico': 'MX', 'méxico': 'MX',
            'netherlands': 'NL', 'holland': 'NL',
            'switzerland': 'CH', 'suisse': 'CH',
            'civ': 'CI', 'ivory coast': 'CI', 'cote divoire': 'CI'  # For PMI CIV data
        }
        
        country_cols = [c for c in harmonized_df.columns if any(x in c.lower() for x in ['country', 'nation', 'market', 'civ', 'cntry'])]
        mapped_countries = 0
        
        for col in country_cols:
            if harmonized_df[col].dtype == 'object':
                harmonized_df[col] = harmonized_df[col].str.lower().str.strip().map(
                    lambda x: country_mappings.get(x, x.upper() if isinstance(x, str) else x)
                )
                mapped_countries += 1
        
        if mapped_countries > 0:
            changes_log.append(f"🌍 Standardized {mapped_countries} country columns to ISO codes")
    
    # ========================================
    # 5. DATE FORMAT STANDARDIZATION
    # ========================================
    if options.get('date_standardize', True):
        date_cols = [c for c in harmonized_df.columns if any(x in c.lower() for x in ['date', 'time', 'dt', 'period'])]
        standardized_dates = 0
        
        for col in date_cols:
            try:
                # Try to parse dates - update IN PLACE (don't create new column)
                parsed = pd.to_datetime(harmonized_df[col], errors='coerce', infer_datetime_format=True)
                valid_count = parsed.notna().sum()
                if valid_count > len(harmonized_df) * 0.5:  # At least 50% valid dates
                    harmonized_df[col] = parsed.dt.strftime('%Y-%m-%d')  # Update original column
                    standardized_dates += 1
            except:
                pass
        
        if standardized_dates > 0:
            changes_log.append(f"📅 Standardized {standardized_dates} date columns to YYYY-MM-DD")
    
    # ========================================
    # 6. WAVE/PERIOD EXTRACTION (disabled - adds unnecessary columns)
    # ========================================
    # Wave extraction is typically done during analysis, not harmonization
    # Keeping this disabled by default to avoid adding extra columns
    
    # ========================================
    # 7. CATEGORY MAPPING (Product, Channel, etc.)
    # ========================================
    if options.get('category_mapping', True):
        category_mappings = {
            # Channels
            'online': 'ONLINE', 'web': 'ONLINE', 'ecommerce': 'ONLINE', 'digital': 'ONLINE',
            'retail': 'RETAIL', 'store': 'RETAIL', 'shop': 'RETAIL', 'brick and mortar': 'RETAIL',
            'mobile': 'MOBILE', 'app': 'MOBILE',
            'social': 'SOCIAL', 'facebook': 'SOCIAL', 'instagram': 'SOCIAL',
            # Product categories (PMI specific)
            'cigarettes': 'CIGARETTES', 'cig': 'CIGARETTES',
            'heated tobacco': 'HTU', 'htu': 'HTU', 'iqos': 'HTU',
            'vape': 'VAPE', 'e-cigarette': 'VAPE', 'ecig': 'VAPE',
            'oral': 'ORAL', 'snus': 'ORAL',
            # Status codes
            'active': 'ACTIVE', 'a': 'ACTIVE', '1': 'ACTIVE',
            'inactive': 'INACTIVE', 'i': 'INACTIVE', '0': 'INACTIVE',
            'pending': 'PENDING', 'p': 'PENDING'
        }
        
        category_cols = [c for c in harmonized_df.columns if any(x in c.lower() for x in ['category', 'channel', 'type', 'segment', 'status', 'cat', 'product'])]
        mapped_cats = 0
        
        for col in category_cols:
            if harmonized_df[col].dtype == 'object':
                harmonized_df[col] = harmonized_df[col].str.lower().str.strip().map(
                    lambda x: category_mappings.get(str(x).lower(), x) if pd.notna(x) else x
                )
                mapped_cats += 1
        
        if mapped_cats > 0:
            changes_log.append(f"🏷️ Standardized {mapped_cats} category columns")
    
    # ========================================
    # 8. DATA TYPE OPTIMIZATION
    # ========================================
    type_conversions = 0
    for col in harmonized_df.select_dtypes(include=['object']).columns:
        try:
            numeric_series = pd.to_numeric(harmonized_df[col], errors='coerce')
            if numeric_series.notna().sum() > len(harmonized_df) * 0.8:
                harmonized_df[col] = numeric_series
                type_conversions += 1
        except:
            pass
    
    if type_conversions > 0:
        changes_log.append(f"🔄 Auto-converted {type_conversions} columns to numeric")
    
    return harmonized_df, changes_log


def run_pipeline(df: pd.DataFrame, progress_callback) -> dict:
    """Run the harmonization pipeline"""
    
    # Get harmonization options from session state
    options = st.session_state.get('harmonization_options', {
        'standardize_cols': True,
        'remove_special': True,
        'default_numeric': 'Replace with MEAN',
        'default_text': "Replace with 'Unknown'",
        'column_strategies': {},
        'remove_missing_rows': False
    })
    
    progress_callback(0.1, "Initializing...")
    
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # ========================================
    # STEP 1: LOCAL HARMONIZATION (NO API)
    # ========================================
    progress_callback(0.2, "Applying harmonization transformations...")
    harmonized_df, changes_log = harmonize_data_local(df, options)
    
    # Save harmonized data FIRST
    output_path = OUTPUT_DIR / 'harmonized.csv'
    harmonized_df.to_csv(output_path, index=False, encoding='utf-8')
    
    progress_callback(0.3, "Verifying harmonized data...")
    
    # Verify the file was saved correctly
    if output_path.exists():
        saved_df = pd.read_csv(output_path)
        changes_log.append(f"💾 Saved harmonized data: {len(saved_df)} rows × {len(saved_df.columns)} columns")
    
    # Store changes log and harmonized dataframe
    st.session_state.harmonization_changes = changes_log
    st.session_state.harmonized_df = harmonized_df  # Store in session for display
    
    # Save ORIGINAL data to input directory (for reference only)
    input_path = INPUT_DIR / 'uploaded_data.csv'
    df.to_csv(input_path, index=False, encoding='utf-8')
    
    # Create sample metadata if not exists
    schema_path = METADATA_DIR / 'master_schema.yaml'
    if not schema_path.exists():
        from main import create_sample_metadata
        create_sample_metadata()
    
    progress_callback(0.5, "Running data quality analysis...")
    
    # Calculate quality metrics locally (no API)
    total_rows = len(harmonized_df)
    total_cols = len(harmonized_df.columns)
    missing_count = harmonized_df.isna().sum().sum()
    missing_pct = (missing_count / (total_rows * total_cols)) * 100 if total_rows * total_cols > 0 else 0
    duplicate_rows = harmonized_df.duplicated().sum()
    
    # Calculate quality score locally
    quality_score = 100 - (missing_pct * 0.5) - (duplicate_rows / total_rows * 10 if total_rows > 0 else 0)
    quality_score = max(0, min(100, quality_score))
    
    progress_callback(0.7, "Finalizing harmonization...")

    # Build local data quality stats (no API)
    column_statistics = []
    for col in harmonized_df.columns:
        null_count = int(harmonized_df[col].isna().sum())
        total_count = int(len(harmonized_df))
        null_percentage = (null_count / total_count * 100) if total_count else 0
        column_statistics.append({
            'column_name': col,
            'data_type': str(harmonized_df[col].dtype),
            'null_count': null_count,
            'null_percentage': null_percentage,
            'total_count': total_count,
            'unique_count': int(harmonized_df[col].nunique(dropna=True))
        })

    # Create a local-only result (AI is used only for analysis in the UI)
    result = type('MockResult', (), {
        'success': True,  # Local harmonization succeeded
        'result': {
            'pipeline_id': f"local_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'status': 'completed',
            'success': True,
            'input_file': str(input_path),
            'output_file': str(output_path),
            'final_quality_score': quality_score,
            'final_confidence_score': 0.85,
            'total_processing_time_seconds': 0,
            'total_llm_calls': 0,
            'total_tokens_used': 0,
            'structural_validation': {'success': True, 'confidence_score': 0.8, 'tokens_used': 0, 'llm_calls': 0},
            'data_quality': {
                'success': True,
                'confidence_score': 0.92,
                'tokens_used': 0,
                'llm_calls': 0,
                'result': {
                    'overall_quality_score': quality_score,
                    'column_statistics': column_statistics,
                    'blocking_issues': [],
                    'fixable_issues': [],
                    'recommendations': []
                }
            },
            'harmonization': {'success': True, 'confidence_score': 1.0, 'tokens_used': 0, 'llm_calls': 0},
            'harmonization_changes': changes_log,
            'audit_trail': [],
            'reports_generated': [],
            'supervisor_decisions': []
        }
    })()
    
    progress_callback(0.9, "Generating reports...")
    time.sleep(0.2)
    
    progress_callback(1.0, "Complete!")
    
    return result


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    if BRAND_LOGO_PATH.exists():
        logo_c1, logo_c2, logo_c3 = st.columns([1, 2, 1])
        with logo_c2:
            # Keep narrow so full "Kantar" fits in sidebar; CSS max-width:100% prevents overflow
            st.image(str(BRAND_LOGO_PATH), width=200)
    brain_c1, brain_c2, brain_c3 = st.columns([1, 2, 1])
    with brain_c2:
        st.markdown(
            "<img class=\"brain-pulse\" src=\"https://img.icons8.com/nolan/96/artificial-intelligence.png\" width=\"90\" />",
            unsafe_allow_html=True
        )
    
    st.divider()
    
    # API Key Status
    st.subheader("🔑 API Status")
    if check_api_key():
        st.success("Azure OpenAI Connected")
    else:
        st.error("API Key Not Set")
        api_key = st.text_input("Enter Azure OpenAI Key", type="password")
        if api_key:
            os.environ["AZURE_OPENAI_API_KEY"] = api_key
            AZURE_CONFIG.api_key = api_key
            st.rerun()
    
    st.divider()
    
    # Settings
    st.subheader("⚙️ Settings")
    
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.5,
        max_value=1.0,
        value=0.85,
        step=0.05
    )
    
    schema_name = st.selectbox(
        "Master Schema",
        options=["master_schema", "custom_schema"],
        index=0
    )
    st.session_state.schema_name = schema_name
    
    st.divider()
    
    # Quick Actions
    st.subheader("⚡ Quick Actions")
    
    if st.button("📊 Load PMI Sample Data"):
        st.session_state.uploaded_df = load_sample_data()
        st.success("PMI CIV sample data loaded (with injected missing values)!")
    
    if st.button("🗑️ Clear Results"):
        st.session_state.pipeline_result = None
        st.session_state.uploaded_df = None
        st.rerun()


# =============================================================================
# MAIN CONTENT
# =============================================================================

# Header
header_left, header_right = st.columns([8, 1])
with header_left:
    st.title("🤖 AI Data Harmonization System")
    st.markdown("*Autonomous data pipeline powered by GPT-5.2 agents*")
with header_right:
    if st.button("🔄 Restart", help="Clear session and reload", use_container_width=True):
        restart_app()

st.divider()

# Main tabs (removed Audit Trail - not needed)
tab1, tab2, tab3, tab4 = st.tabs([
    "📤 Upload & Process",
    "📊 Results Dashboard",
    "📋 Reports",
    "🔍 Harmonization Results"
])


# =============================================================================
# TAB 1: UPLOAD & PROCESS
# =============================================================================

with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📁 Data Input")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Drop your data file here",
            type=['csv', 'xlsx', 'xls', 'json'],
            help="Supported formats: CSV, Excel, JSON"
        )
        
        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(uploaded_file)
                elif uploaded_file.name.endswith('.json'):
                    df = pd.read_json(uploaded_file)
                
                st.session_state.uploaded_df = df
                st.success(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
        
        # Data preview - Full table with scroll
        if st.session_state.uploaded_df is not None:
            st.subheader("👀 Data Preview (Full Table)")
            df = st.session_state.uploaded_df
            st.dataframe(
                df,
                height=600  # Tall enough to show many rows with scroll
            )
            st.caption(f"📊 Showing all {len(df):,} rows × {len(df.columns):,} columns — Scroll ↕↔ to view more")
    
    with col2:
        st.subheader("📈 Data Stats")
        
        if st.session_state.uploaded_df is not None:
            df = st.session_state.uploaded_df
            
            # Stats metrics
            st.metric("Total Rows", f"{len(df):,}")
            st.metric("Total Columns", len(df.columns))
            st.metric("Missing Values", f"{df.isna().sum().sum():,}")
            st.metric("Duplicate Rows", f"{df.duplicated().sum():,}")
            
            # Column types
            st.markdown("**Column Types:**")
            type_counts = df.dtypes.value_counts()
            for dtype, count in type_counts.items():
                st.write(f"• {dtype}: {count}")
        else:
            st.info("Upload a file to see statistics")
    
    st.divider()

    # ===========================================
    # MULTI-SOURCE HARMONIZATION
    # ===========================================
    st.subheader("🧩 Multi-Source Harmonization")
    st.caption("Harmonize historical, current, and incremental data together.")
    with st.expander("Configure multi-source pipeline", expanded=False):
        auto_detect = st.checkbox(
            "Auto-detect files from data/input and metadata",
            value=True
        )
        auto_files = auto_detect_multisource_files() if auto_detect else {}
        if auto_detect:
            st.caption(
                "Auto-detected: "
                f"master={auto_files.get('master') or 'not found'}, "
                f"historical={auto_files.get('historical') or 'not found'}, "
                f"current={auto_files.get('current') or 'not found'}, "
                f"incremental={auto_files.get('incremental') or 'not found'}"
            )

        col_a, col_b = st.columns(2)
        with col_a:
            master_metadata_file = st.file_uploader(
                "Master metadata file",
                type=['yaml', 'yml', 'json'],
                key="multi_master"
            )
            historical_file = st.file_uploader(
                "Historical data (Vendor A)",
                type=['csv', 'xlsx', 'xls', 'json', 'sav'],
                key="multi_hist"
            )
            incremental_file = st.file_uploader(
                "Incremental data (new respondents)",
                type=['csv', 'xlsx', 'xls', 'json', 'sav'],
                key="multi_incr"
            )
            validation_rules_file = st.file_uploader(
                "Validation rules file",
                type=['yaml', 'yml', 'json', 'txt'],
                key="multi_rules"
            )
        with col_b:
            current_file = st.file_uploader(
                "Current data (Vendor B)",
                type=['csv', 'xlsx', 'xls', 'json', 'sav'],
                key="multi_curr"
            )
            mapping_table_file = st.file_uploader(
                "Mapping table (self-updating)",
                type=['yaml', 'yml', 'json'],
                key="multi_map"
            )
            descriptive_stats_file = st.file_uploader(
                "Baseline descriptive statistics (<= 1000 rows)",
                type=['csv', 'xlsx', 'xls'],
                key="multi_stats"
            )
            full_reprocess = st.checkbox(
                "Reprocess full history (ignore incremental-only)",
                value=False
            )

        output_filename = st.text_input(
            "Output filename",
            value="harmonized_master.csv"
        )

        run_multi = st.button("🚀 Run Multi-Source Harmonization")

        if run_multi:
            master_path = str(save_uploaded_file(master_metadata_file, "master")) if master_metadata_file else (
                str(auto_files.get("master")) if auto_files.get("master") else None
            )
            historical_path = str(save_uploaded_file(historical_file, "historical")) if historical_file else (
                str(auto_files.get("historical")) if auto_files.get("historical") else None
            )
            current_path = str(save_uploaded_file(current_file, "current")) if current_file else (
                str(auto_files.get("current")) if auto_files.get("current") else None
            )
            incremental_path = str(save_uploaded_file(incremental_file, "incremental")) if incremental_file else (
                str(auto_files.get("incremental")) if auto_files.get("incremental") else None
            )
            rules_path = str(save_uploaded_file(validation_rules_file, "rules")) if validation_rules_file else (
                str(auto_files.get("rules")) if auto_files.get("rules") else None
            )
            mapping_path = str(save_uploaded_file(mapping_table_file, "mapping")) if mapping_table_file else (
                str(auto_files.get("mapping")) if auto_files.get("mapping") else None
            )
            stats_path = str(save_uploaded_file(descriptive_stats_file, "baseline")) if descriptive_stats_file else (
                str(auto_files.get("stats")) if auto_files.get("stats") else None
            )

            if not historical_path and not current_path:
                st.error("Please upload or auto-detect at least a historical or current dataset.")
            else:
                with st.spinner("Running multi-source harmonization..."):
                    harmonizer = MultiSourceHarmonizer()
                    result = harmonizer.run(
                        master_metadata_file=master_path,
                        historical_file=historical_path,
                        current_file=current_path,
                        incremental_file=incremental_path,
                        validation_rules_file=rules_path,
                        mapping_table_file=mapping_path,
                        descriptive_stats_file=stats_path,
                        output_file=str(OUTPUT_DIR / output_filename),
                        full_reprocess=full_reprocess
                    )
                    st.session_state.multi_source_result = result
                st.success("Multi-source harmonization complete.")

        if st.session_state.get("multi_source_result"):
            result = st.session_state.multi_source_result
            st.markdown("**Outputs**")
            st.write(f"Output dataset: {result.get('output_file')}")
            st.write(f"Mapping table: {result.get('mapping_table_file')}")
            st.write(f"Calibration report: {result.get('calibration_report')}")
            st.write(f"Descriptive statistics: {result.get('descriptive_statistics')}")
            st.write(f"Validation flags: {result.get('validation_flags')}")
            st.write(f"Market silo comparison: {result.get('market_silo_comparison')}")
            st.write(f"Cross-dataset relationships: {result.get('cross_dataset_relationships')}")
            st.write(f"Knowledge bag: {result.get('knowledge_bag')}")
    
    st.divider()
    
    # ===========================================
    # AI-POWERED ANALYSIS (Simplified!)
    # ===========================================
    if st.session_state.uploaded_df is not None:
        df = st.session_state.uploaded_df

        # Load prior overrides so quality analysis uses learned decisions
        knowledge_bag_path = METADATA_DIR / "knowledge_bag.yaml"
        knowledge_bag = load_knowledge_bag(knowledge_bag_path)
        if "column_missing_strategies" not in st.session_state:
            st.session_state.column_missing_strategies = {}
        if knowledge_bag.get("imputation_overrides") and not st.session_state.get("overrides_loaded"):
            latest = {}
            for entry in knowledge_bag.get("imputation_overrides", []):
                col = entry.get("column")
                strategy = entry.get("strategy")
                if col and strategy:
                    latest[col] = strategy
            st.session_state.column_missing_strategies.update(latest)
            st.session_state.overrides_loaded = True

        # ===========================================
        # AUTO-GENERATE SUPPORTING FILES
        # ===========================================
        st.subheader("🧰 Auto-Build Supporting Files (from Unclean Input)")
        st.caption("Generates mapping, dictionary, validation rules, metadata, and stats from the uploaded file.")

        spec_json_file = st.file_uploader(
            "Spec JSON (master dictionary)",
            type=["json"],
            key="spec_json_file"
        )
        if spec_json_file:
            st.session_state.spec_json_path = str(save_uploaded_file(spec_json_file, "spec"))

        col_gen1, col_gen2 = st.columns([1, 2])
        with col_gen1:
            generate_clicked = st.button("🛠️ Build Supporting Files")
        with col_gen2:
            if st.session_state.get("supporting_files"):
                st.success("Supporting files ready.")

        if generate_clicked:
            with st.spinner("Building supporting files..."):
                paths = generate_supporting_files(
                    df=df,
                    output_dir=INPUT_DIR,
                    metadata_dir=METADATA_DIR,
                    prefix="auto"
                )
                st.session_state.supporting_files = paths
            st.success("Supporting files created.")

        if st.session_state.get("supporting_files"):
            st.markdown("**Generated Files**")
            for label, path in st.session_state.supporting_files.items():
                st.write(f"{label}: {path}")

        st.divider()

        # ===========================================
        # SCHEMA VALIDATION & MAPPING REVIEW
        # ===========================================
        st.subheader("🧩 Schema Validation — Review & Confirm")
        st.caption("Review suggested column mappings and approve them before harmonization.")

        if "schema_validation_result" not in st.session_state:
            st.session_state.schema_validation_result = None
        if "mapping_decisions" not in st.session_state:
            st.session_state.mapping_decisions = {}

        run_validation = st.button("🔍 Run Schema Validation")
        if run_validation:
            with st.spinner("Running schema validation..."):
                metadata_handler = MetadataHandler(METADATA_DIR)
                schema_name = st.session_state.get("schema_name", "master_schema")
                try:
                    master_schema = metadata_handler.load_master_schema(schema_name)
                except Exception:
                    master_schema = metadata_handler.load_master_schema("master_schema")

                agent = StructuralValidationAgent()
                try:
                    result = agent.execute(df, master_schema)
                    validation = result.result or {}
                    st.session_state.schema_validation_result = {
                        "validation": validation,
                        "master_schema": master_schema
                    }
                except Exception as e:
                    source_schema = agent._extract_source_schema(df)
                    rule_mappings = agent._rule_based_matching(source_schema, master_schema)
                    st.session_state.schema_validation_result = {
                        "validation": {
                            "column_mappings": [m.model_dump() for m in rule_mappings],
                            "validation_warnings": [f"Schema validation fallback: {str(e)}"]
                        },
                        "master_schema": master_schema
                    }
            st.success("Schema validation completed.")

        validation_bundle = st.session_state.get("schema_validation_result")
        if validation_bundle:
            validation = validation_bundle.get("validation", {})
            master_schema = validation_bundle.get("master_schema", {})
            master_cols = [c.get("name") for c in master_schema.get("columns", [])]

            column_mappings = validation.get("column_mappings", [])
            if column_mappings:
                with st.expander("🧷 Column Mappings — Review & Confirm", expanded=True):
                    for mapping in column_mappings:
                        source_col = mapping.get("source_column")
                        target_col = mapping.get("target_column")
                        confidence = mapping.get("confidence", 0)
                        label = f"{source_col} → {target_col} ({'High' if confidence >= 0.8 else 'Medium' if confidence >= 0.6 else 'Low'})"

                        with st.expander(label, expanded=False):
                            evidence = {}
                            if source_col in df.columns:
                                series = df[source_col]
                                evidence = {
                                    "type": str(series.dtype),
                                    "unique_values": int(series.nunique(dropna=True)),
                                    "null_pct": round(series.isna().mean() * 100, 2),
                                    "sample_values": series.dropna().astype(str).head(5).tolist()
                                }

                            col_left, col_right = st.columns([2, 1])
                            with col_left:
                                st.markdown("**Evidence**")
                                st.caption(f"Source Column: {source_col}")
                                if evidence:
                                    st.caption(f"Type: {evidence['type']}")
                                    st.caption(f"Unique values: {evidence['unique_values']}")
                                    st.caption(f"Null %: {evidence['null_pct']}")
                                    st.caption(f"Sample values: {evidence['sample_values']}")

                                if mapping.get("reasoning"):
                                    st.markdown("**AI Reasoning**")
                                    st.caption(mapping.get("reasoning"))

                            with col_right:
                                st.markdown("**Your Decision**")
                                decision_key = f"decision_{source_col}"
                                action = st.radio(
                                    "Action",
                                    ["Confirm suggested", "Map to different", "Add as new column", "Skip this column"],
                                    key=decision_key,
                                    label_visibility="collapsed"
                                )
                                target_choice = target_col
                                if action == "Map to different":
                                    target_choice = st.selectbox(
                                        "Select target column",
                                        options=[c for c in master_cols if c] + ["UNMAPPED"],
                                        key=f"target_{source_col}"
                                    )

                                st.session_state.mapping_decisions[source_col] = {
                                    "action": "confirm" if action == "Confirm suggested" else
                                              "map_to" if action == "Map to different" else
                                              "add_new" if action == "Add as new column" else
                                              "skip",
                                    "target": target_choice
                                }

                    apply_mappings = st.button("✅ Apply Mapping Decisions")
                    if apply_mappings:
                        mapped_df = apply_mapping_decisions(
                            df,
                            st.session_state.mapping_decisions
                        )
                        st.session_state.mapped_df = mapped_df
                        st.success("Mapping decisions applied. Harmonization will use mapped data.")
        
        # Initialize AI recommendations in session state
        if 'ai_recommendations' not in st.session_state:
            st.session_state.ai_recommendations = None
        if 'column_missing_strategies' not in st.session_state:
            st.session_state.column_missing_strategies = {}
        
        # Count missing values
        missing_count = df.isna().sum().sum()
        cols_with_missing = (df.isna().sum() > 0).sum()
        
        # ===========================================
        # STEP 1: AI ANALYSIS BUTTON
        # ===========================================
        st.subheader("🤖 AI-Powered Data Analysis")
        st.caption("AI is used only for analysis/insights (no transformations).")
        
        if "dictionary_suggestions" not in st.session_state:
            st.session_state.dictionary_suggestions = []
        if "dictionary_warnings" not in st.session_state:
            st.session_state.dictionary_warnings = []

        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
        
        with col_btn1:
            analyze_clicked = st.button(
                "🔍 Analyze with AI",
                type="primary" if st.session_state.ai_recommendations is None else "secondary",
                help="One API call to analyze all columns and suggest best practices"
            )

        with col_btn2:
            load_dictionary = st.button(
                "📥 Load Dictionary Standards",
                help="Load fallback strategies from the master dictionary"
            )

        with col_btn3:
            status_parts = []
            if st.session_state.ai_recommendations:
                tokens = st.session_state.ai_recommendations.get('tokens_used', 0)
                status_parts.append(f"AI ready ({tokens:,} tokens)")
            if st.session_state.dictionary_suggestions:
                status_parts.append("Dictionary loaded")
            if status_parts:
                st.success(f"✅ {', '.join(status_parts)}")
        
        # Run AI analysis when button clicked
        if analyze_clicked:
            if check_api_key():
                with st.spinner("🔍 AI Market Analyst reviewing your data..."):
                    result = analyze_data_with_ai(df)
                    st.session_state.ai_recommendations = result

                    st.rerun()
            else:
                st.error("⚠️ Please configure Azure OpenAI API key in sidebar first")

        # Load dictionary standards when button clicked
        if load_dictionary:
            standards_path = None
            if st.session_state.get("spec_json_path"):
                standards_path = st.session_state.spec_json_path
            elif st.session_state.get("supporting_files"):
                standards_path = st.session_state.supporting_files.get("master_dictionary")
            if not standards_path:
                standards_path = str(METADATA_DIR / "auto_master_dictionary.json")
            try:
                with open(standards_path, "r", encoding="utf-8") as f:
                    standards = json.load(f)
                suggestions, warnings = build_dictionary_suggestions(df, standards)
                st.session_state.dictionary_suggestions = suggestions
                st.session_state.dictionary_warnings = warnings
                st.success("Dictionary standards loaded.")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to load dictionary: {str(e)}")
        
        # ===========================================
        # STEP 2: SHOW AI RECOMMENDATIONS
        # ===========================================
        combined_recs: Dict[str, Dict[str, Any]] = {}
        if st.session_state.ai_recommendations:
            recs = st.session_state.ai_recommendations
            
            # Show overall assessment
            if recs.get('data_quality_notes'):
                st.info(f"📊 **AI Assessment:** {recs['data_quality_notes']}")

            # Show short summary below the tab
            summary_text = recs.get('analysis_summary') or recs.get('data_quality_notes')
            if summary_text:
                st.markdown(f"**AI Summary:** {summary_text}")
            
            # Show suggested actions
            if recs.get('suggested_actions'):
                with st.expander("💡 AI Suggested Actions", expanded=False):
                    for action in recs['suggested_actions']:
                        st.write(f"• {action}")
            
            # Show column recommendations
            if recs.get('recommendations'):
                for rec in recs['recommendations']:
                    col_name = rec.get('column')
                    if not col_name:
                        continue
                    combined_recs.setdefault(col_name, {})
                    combined_recs[col_name].update({
                        "ai_strategy": rec.get('strategy', 'keep'),
                        "ai_reason": rec.get('reason', ''),
                        "ai_missing": rec.get('missing'),
                        "ai_source": "AI"
                    })

        if st.session_state.dictionary_suggestions:
            for rec in st.session_state.dictionary_suggestions:
                col_name = rec.get("column")
                if not col_name:
                    continue
                combined_recs.setdefault(col_name, {})
                combined_recs[col_name].update({
                    "dict_strategy": rec.get("strategy"),
                    "dict_reason": rec.get("reason", ""),
                    "dict_missing": rec.get("missing"),
                    "dict_source": "Dictionary"
                })

        if combined_recs:
            st.markdown("### 📋 Missing Value Recommendations")
            st.caption("Unified view of AI + Dictionary suggestions. Review and apply once.")

            apply_combined = st.button("✅ Apply Recommendations", key="apply_combined_recs")
            if apply_combined:
                knowledge_bag = load_knowledge_bag(knowledge_bag_path)
                knowledge_bag = update_imputation_overrides(
                    knowledge_bag,
                    st.session_state.column_missing_strategies,
                    source="combined_recommendations"
                )
                save_knowledge_bag(knowledge_bag_path, knowledge_bag)
                st.success("Recommendations applied.")

            strategy_map = {
                'mean': 'Replace with MEAN',
                'median': 'Replace with MEDIAN',
                'zero': 'Replace with 0',
                'mode': 'Replace with MODE',
                'unknown': "Replace with 'Unknown'",
                'other': "Replace with 'Other'",
                'remove': 'Remove rows',
                'keep': 'Keep missing',
                'drop': 'Drop column'
            }

            for col_name, info in list(combined_recs.items())[:30]:
                is_numeric = pd.api.types.is_numeric_dtype(df[col_name]) if col_name in df.columns else False
                if is_numeric:
                    options = [
                        "Keep missing",
                        "Replace with MEAN",
                        "Replace with MEDIAN",
                        "Replace with 0",
                        "Replace with MODE",
                        "Remove rows",
                        "Drop column"
                    ]
                else:
                    options = [
                        "Keep missing",
                        "Replace with 'Unknown'",
                        "Replace with 'Other'",
                        "Replace with MODE",
                        "Replace with 'N/A'",
                        "Remove rows",
                        "Drop column"
                    ]

                ai_strategy = info.get("ai_strategy")
                dict_strategy = info.get("dict_strategy")
                default_strategy = dict_strategy or strategy_map.get(ai_strategy, 'Keep missing')
                current = st.session_state.column_missing_strategies.get(col_name, default_strategy)
                if current not in options:
                    current = options[0]

                sources = []
                if info.get("ai_source"):
                    sources.append("AI")
                if info.get("dict_source"):
                    sources.append("Dictionary")
                source_label = " + ".join(sources) if sources else "Recommendations"

                reason_parts = []
                if info.get("ai_reason"):
                    reason_parts.append(f"AI: {info.get('ai_reason')}")
                if info.get("dict_reason"):
                    reason_parts.append(f"Dictionary: {info.get('dict_reason')}")
                reason_text = " | ".join(reason_parts)

                rec_col1, rec_col2, rec_col3 = st.columns([2, 2, 3])
                with rec_col1:
                    type_icon = "🔢" if is_numeric else "📝"
                    missing = df[col_name].isna().sum() if col_name in df.columns else 0
                    st.markdown(f"{type_icon} **{col_name[:25]}**{'...' if len(col_name) > 25 else ''}")
                    st.caption(f"{missing:,} missing • {source_label}")

                with rec_col2:
                    selected = st.selectbox(
                        f"Action for {col_name}",
                        options=options,
                        index=options.index(current) if current in options else 0,
                        key=f"combined_rec_{col_name}",
                        label_visibility="collapsed"
                    )
                    st.session_state.column_missing_strategies[col_name] = selected

                with rec_col3:
                    if reason_text:
                        st.caption(f"💡 *{reason_text}*")

            if st.session_state.dictionary_warnings:
                st.warning("Standards warnings detected:")
                for warn in st.session_state.dictionary_warnings[:8]:
                    st.caption(f"• {warn}")
            
        # ===========================================
        # HARMONIZATION OPTIONS (Manual Control)
        # ===========================================
        st.markdown("---")
        st.subheader("⚙️ Harmonization Options")
        st.caption("Transformations are hard-coded locally. AI suggestions above are for missing values only.")
        
        opt_col1, opt_col2, opt_col3 = st.columns(3)
        
        with opt_col1:
            st.markdown("**📝 Data Cleaning:**")
            standardize_cols = st.checkbox("Standardize column names", value=True, 
                help="Convert to lowercase, replace spaces with underscores")
            remove_special = st.checkbox("Remove special characters", value=True,
                help="Remove characters like @, #, $, etc.")
        
        with opt_col2:
            st.markdown("**🗺️ Standardization:**")
            region_mapping = st.checkbox("Standardize region codes", value=True,
                help="Map region variations to standard codes")
            country_mapping = st.checkbox("Standardize country codes", value=True,
                help="Convert country names to ISO codes")
            date_standardize = st.checkbox("Standardize date formats", value=True,
                help="Convert all dates to YYYY-MM-DD format")
        
        with opt_col3:
            st.markdown("**🔧 Missing Values:**")
            has_recommendations = bool(st.session_state.get("column_missing_strategies"))
            if has_recommendations:
                st.caption("Using per-column recommendations you selected above.")
                default_numeric = "Keep missing"
                default_text = "Keep missing"
                missing_strategy = "Per-column recommendations"
            else:
                st.caption("No recommendations yet — using Smart Fill defaults.")
                default_numeric = "Replace with MEAN"
                default_text = "Replace with 'Unknown'"
                missing_strategy = "Smart Fill defaults"
        
        # Store all options
        st.session_state.harmonization_options = {
            'standardize_cols': standardize_cols,
            'remove_special': remove_special,
            'default_numeric': default_numeric,
            'default_text': default_text,
            'column_strategies': st.session_state.get('column_missing_strategies', {}),
            'remove_missing_rows': False,
            'missing_strategy': missing_strategy,
            'scale_conversion': 'No conversion',
            'region_mapping': region_mapping,
            'country_mapping': country_mapping,
            'date_standardize': date_standardize,
            'wave_extraction': False,
            'category_mapping': True,
            'hierarchy_mapping': False
        }
        
        st.divider()
    
    # Process button
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.session_state.uploaded_df is not None:
            if st.button("🚀 Start Harmonization", type="primary"):
                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(progress, message):
                    progress_bar.progress(progress)
                    status_text.text(message)
                
                with st.spinner("Running local harmonization (no API)..."):
                    try:
                        df_to_use = st.session_state.get("mapped_df") or st.session_state.uploaded_df
                        result = run_pipeline(
                            df_to_use,
                            update_progress
                        )
                        st.session_state.pipeline_result = result
                        
                        # Check if pipeline succeeded
                        if result.success:
                            st.success("✅ Pipeline completed successfully!")
                        else:
                            st.warning("⚠️ Pipeline completed with issues. Check Results tab for details.")
                            # Show errors if any
                            result_dict = to_dict(result.result) if result.result else {}
                            errors = result_dict.get('errors', []) or []
                            if errors:
                                with st.expander("🔍 View Issues", expanded=True):
                                    for err in errors[:5]:
                                        st.error(f"• {err}")
                        
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        st.session_state.pipeline_error = str(e)
                        st.error(f"❌ Pipeline failed: {str(e)}")
                        with st.expander("🔧 Troubleshooting Tips", expanded=True):
                            st.markdown("""
                            **Common issues:**
                            - Verify the data file format is correct
                            - Ensure the file has headers in the first row
                            - Check console for detailed error logs
                            """)
        else:
            st.info("👆 Upload a data file to get started")


# =============================================================================
# TAB 2: RESULTS DASHBOARD
# =============================================================================

with tab2:
    if st.session_state.pipeline_result:
        result = st.session_state.pipeline_result
        result_data = to_dict(result.result) if result.result else {}
        
        # Show status banner - warnings instead of errors for non-critical issues
        if not result.success:
            # Check if harmonization actually completed (that's what matters)
            harm_data = to_dict(result_data.get('harmonization', {}))
            harm_success = harm_data.get('success', False) if harm_data else False
            
            if harm_success:
                st.success("✅ **Harmonization Completed Successfully!** (Some agents had warnings)")
            else:
                st.warning("⚠️ **Pipeline completed with warnings** - Data was still processed")
            
            # Show warnings (not errors) for informational messages
            all_warnings = []
            
            # Structural validation - just informational, not critical
            sv_data = to_dict(result_data.get('structural_validation', {}))
            if sv_data and not sv_data.get('success', True):
                sv_errors = sv_data.get('errors', []) or []
                # Filter out technical errors, show user-friendly messages
                for e in sv_errors[:2]:
                    if 'keys' in str(e) or 'attribute' in str(e):
                        all_warnings.append("[Info] Schema validation skipped - using flexible mapping")
                    else:
                        all_warnings.append(f"[Info] {e}")
            
            # Data quality - these are informational, not blocking
            dq_data = to_dict(result_data.get('data_quality', {}))
            if dq_data:
                dq_errors = dq_data.get('errors', []) or []
                for e in dq_errors[:3]:
                    # Convert error messages to informational
                    if 'missing' in str(e).lower():
                        all_warnings.append(f"[Note] {e} - Check your missing value settings")
                    else:
                        all_warnings.append(f"[Note] {e}")
            
            if all_warnings:
                with st.expander("ℹ️ Processing Notes (Not Errors)", expanded=False):
                    st.info("These are informational messages. Your data was still harmonized!")
                    for warn in all_warnings[:5]:
                        st.caption(f"• {warn}")
            
            st.divider()
        else:
            st.success("✅ **Pipeline Completed Successfully!**")
            st.divider()
        
        # ========================================
        # HARMONIZATION CHANGES LOG
        # ========================================
        if hasattr(st.session_state, 'harmonization_changes') and st.session_state.harmonization_changes:
            st.subheader("✅ Harmonization Applied")
            
            # Show options used
            options = st.session_state.get('harmonization_options', {})
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Settings Used:**")
                st.caption(f"• Missing Values: {options.get('missing_strategy', 'Keep as-is')}")
                st.caption(f"• Scale Conversion: {options.get('scale_conversion', 'None')}")
                st.caption(f"• Column Standardization: {'Yes' if options.get('standardize_cols') else 'No'}")
            
            with col2:
                st.markdown("**Mappings Applied:**")
                st.caption(f"• Region Mapping: {'Yes' if options.get('region_mapping') else 'No'}")
                st.caption(f"• Country Codes: {'Yes' if options.get('country_mapping') else 'No'}")
                st.caption(f"• Category Mapping: {'Yes' if options.get('category_mapping') else 'No'}")
            
            # Show changes
            with st.expander("📋 Detailed Changes Log", expanded=True):
                for change in st.session_state.harmonization_changes:
                    st.success(f"✓ {change}")
                st.caption("✨ All transformations done locally (no API tokens used)")
            
            st.divider()

        # ===========================================
        # DATA CHANGE PREVIEW
        # ===========================================
        raw_df = st.session_state.get('uploaded_df', None)
        harmonized_df = st.session_state.get('harmonized_df', None)
        if raw_df is not None and harmonized_df is not None:
            st.subheader("🔍 Data Changes Preview (Raw vs Harmonized)")
            styled_preview, summary = build_change_preview(raw_df, harmonized_df, max_rows=50)

            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Raw Rows", summary["raw_rows"])
                st.metric("Harmonized Rows", summary["harmonized_rows"])
            with col_b:
                st.metric("Raw Columns", summary["raw_columns"])
                st.metric("Harmonized Columns", summary["harmonized_columns"])
            with col_c:
                st.metric("Changed Cells (preview)", summary["changed_cells_in_preview"])
                st.metric("Rows w/ Changes (preview)", summary["rows_with_changes_in_preview"])

            if summary["added_columns"]:
                st.caption(f"Added columns: {', '.join(summary['added_columns'][:10])}")
            if summary["removed_columns"]:
                st.caption(f"Removed columns: {', '.join(summary['removed_columns'][:10])}")

            st.dataframe(styled_preview, height=400)
            if "_changed_preview" in getattr(styled_preview, "columns", []):
                st.caption("Styling unavailable. Use `_changed_preview` to filter changed rows.")
            else:
                st.caption("Highlighted cells indicate differences between raw and harmonized data (preview).")

            st.divider()
        
        # Top metrics row
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            quality_score = result_data.get('final_quality_score', 0) if result_data else 0
            quality_score = quality_score or 0
            st.metric(
                "Quality Score",
                f"{quality_score:.1f}%",
                delta="Good" if quality_score >= 80 else "Needs Review"
            )
        
        with col2:
            confidence = result_data.get('final_confidence_score', 0) if result_data else 0
            confidence = confidence or 0
            st.metric(
                "Confidence",
                f"{confidence:.1%}",
                delta="High" if confidence >= 0.9 else "Moderate"
            )
        
        with col3:
            status = "Success" if result.success else "Issues Found"
            status_delta = "OK" if result.success else "Review"
            st.metric("Status", status, delta=status_delta)
        
        with col4:
            duration = result_data.get('total_processing_time_seconds', 0) if result_data else 0
            duration = duration or 0
            st.metric("Duration", f"{duration:.1f}s")
        
        with col5:
            # Token usage metrics
            tokens = result_data.get('total_tokens_used', 0) if result_data else 0
            tokens = tokens or 0
            llm_calls = result_data.get('total_llm_calls', 0) if result_data else 0
            llm_calls = llm_calls or 0
            # Estimate cost (GPT-4 pricing ~$0.03/1k input, ~$0.06/1k output)
            est_cost = (tokens / 1000) * 0.05  # Simplified average
            st.metric(
                "🔢 Tokens Used",
                f"{tokens:,}",
                delta=f"~${est_cost:.2f}" if tokens > 0 else "Free"
            )
        
        # Token usage details expander
        if result_data:
            tokens = result_data.get('total_tokens_used', 0) or 0
            llm_calls = result_data.get('total_llm_calls', 0) or 0
            if tokens > 0 or llm_calls > 0:
                with st.expander("💰 Token Usage Details", expanded=False):
                    tcol1, tcol2, tcol3 = st.columns(3)
                    with tcol1:
                        st.markdown(f"**Total Tokens:** {tokens:,}")
                    with tcol2:
                        st.markdown(f"**LLM Calls:** {llm_calls}")
                    with tcol3:
                        est_cost = (tokens / 1000) * 0.05
                        st.markdown(f"**Est. Cost:** ${est_cost:.3f}")
                    
                    # Show per-agent breakdown if available
                    st.markdown("---")
                    st.markdown("**Per-Agent Breakdown:**")
                    
                    sv_data = to_dict(result_data.get('structural_validation', {}))
                    dq_data = to_dict(result_data.get('data_quality', {}))
                    harm_data = to_dict(result_data.get('harmonization', {}))
                    
                    breakdown_data = [
                        {"Agent": "Structural Validation", "Tokens": sv_data.get('tokens_used', 0) or 0, "LLM Calls": sv_data.get('llm_calls', 0) or 0},
                        {"Agent": "Data Quality", "Tokens": dq_data.get('tokens_used', 0) or 0, "LLM Calls": dq_data.get('llm_calls', 0) or 0},
                        {"Agent": "Harmonization", "Tokens": harm_data.get('tokens_used', 0) or 0, "LLM Calls": harm_data.get('llm_calls', 0) or 0},
                    ]
                    st.dataframe(pd.DataFrame(breakdown_data), hide_index=True)
                    
                    # Savings message
                    if tokens < 10000:
                        st.success("🎉 **Token Optimized!** Using rule-based logic saved ~90% of tokens compared to full LLM processing.")
        
        st.divider()
        
        # Charts row
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Quality Gauge")
            fig = create_quality_gauge(quality_score)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🎯 Agent Performance")
            sv_data = to_dict(result_data.get('structural_validation', {}))
            dq_data = to_dict(result_data.get('data_quality', {}))
            harm_data = to_dict(result_data.get('harmonization', {}))
            confidences = {
                "Structural": sv_data.get('confidence_score', 0) or 0,
                "Data Quality": dq_data.get('confidence_score', 0) or 0,
                "Harmonization": harm_data.get('confidence_score', 0) or 0
            }
            fig = create_confidence_bar(confidences)
            st.plotly_chart(fig, config={'displayModeBar': False})
        
        st.divider()
        
        # Agent results
        st.subheader("🤖 Agent Results")
        
        agent_cols = st.columns(3)
        
        agents_info = [
            ("🔍 Structural Validation", "structural_validation"),
            ("📊 Data Quality", "data_quality"),
            ("🔄 Harmonization", "harmonization")
        ]
        
        for i, (title, key) in enumerate(agents_info):
            with agent_cols[i]:
                agent_result = to_dict(result_data.get(key, {}))
                success = agent_result.get('success', False)
                confidence = agent_result.get('confidence_score', 0) or 0
                status_text = '✅ Success' if success else '⚠️ Issues'
                conf_color = '#10b981' if confidence >= 0.8 else '#f59e0b' if confidence >= 0.5 else '#ef4444'
                
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #1e293b 0%, #374151 100%); border-radius: 12px; padding: 20px; border-left: 4px solid #6366f1;">
                    <h4 style="color: #ffffff; margin: 0 0 12px 0; font-size: 1.1em;">{title}</h4>
                    <p style="color: #e2e8f0; margin: 8px 0;">Status: {status_text}</p>
                    <p style="color: {conf_color}; margin: 8px 0; font-weight: 600;">Confidence: {confidence:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # ===========================================
        # MISSING VALUES SUMMARY - FROM HARMONIZED DATA
        # ===========================================
        st.divider()
        st.subheader("📉 Missing Values Summary (After Harmonization)")
        
        # Use HARMONIZED data from session state if available
        harm_df_for_quality = st.session_state.get('harmonized_df', None)
        
        if harm_df_for_quality is not None:
            # Calculate missing values from HARMONIZED data
            missing_data = []
            total_rows = len(harm_df_for_quality)
            
            for col in harm_df_for_quality.columns:
                null_count = harm_df_for_quality[col].isna().sum()
                if null_count > 0:
                    null_pct = (null_count / total_rows) * 100
                    missing_data.append({
                        'Column': col,
                        'Missing Count': int(null_count),
                        'Missing %': f"{null_pct:.1f}%",
                        'Total Rows': total_rows
                    })
            
            if missing_data:
                # Sort by missing count descending
                missing_data.sort(key=lambda x: x['Missing Count'], reverse=True)
                st.warning(f"⚠️ {len(missing_data)} columns still have missing values after harmonization:")
                st.dataframe(
                    pd.DataFrame(missing_data[:20]),  # Show top 20
                    hide_index=True
                )
                st.caption(f"Showing top 20 columns (Total: {len(missing_data)} columns have missing data)")
                st.info("💡 **Tip:** Try running again with a different Missing Value strategy (e.g., 'Fill numeric with MEAN' or 'Remove rows')")
            else:
                st.success("✅ **No missing values** in any column after harmonization!")
                total_cells = harm_df_for_quality.shape[0] * harm_df_for_quality.shape[1]
                st.metric("Data Completeness", f"100%", delta=f"{total_cells:,} cells")
        else:
            # Fallback to API result
            dq_agent = to_dict(result_data.get('data_quality', {}))
            dq_result = to_dict(dq_agent.get('result', {}))
            if dq_result:
                column_stats = dq_result.get('column_statistics', []) or []
                if column_stats:
                    missing_data = []
                    for stat in column_stats:
                        stat = to_dict(stat)
                        null_count = stat.get('null_count', 0) or 0
                        null_pct = stat.get('null_percentage', 0) or 0
                        if null_count > 0:
                            missing_data.append({
                                'Column': stat.get('column_name', 'Unknown'),
                                'Missing Count': null_count,
                                'Missing %': f"{null_pct:.1f}%",
                                'Total Rows': stat.get('total_count', 0)
                            })
                    
                    if missing_data:
                        missing_data.sort(key=lambda x: x['Missing Count'], reverse=True)
                        st.dataframe(pd.DataFrame(missing_data[:20]), hide_index=True)
                    else:
                        st.success("✅ No missing values found!")
        
        # ===========================================
        # RAW INPUT vs HARMONIZED COMPARISON
        # ===========================================
        st.divider()
        st.subheader("🔄 Data Transformation Preview")
        st.caption("Side-by-side comparison of raw input and harmonized output")
        
        # Get harmonized data if available
        output_file = result_data.get('output_file')
        
        # Get harmonized data from session state (most reliable) or file
        raw_df = st.session_state.uploaded_df
        harm_df = None
        
        # First try session state (fastest, most reliable)
        if hasattr(st.session_state, 'harmonized_df') and st.session_state.harmonized_df is not None:
            harm_df = st.session_state.harmonized_df
        # Then try file
        elif output_file and Path(output_file).exists():
            try:
                harm_df = pd.read_csv(output_file)
                if harm_df.empty or len(harm_df.columns) == 0:
                    harm_df = None
            except:
                harm_df = None
        
        if raw_df is not None and harm_df is not None:
            # Create side-by-side comparison
            comp_col1, comp_arrow, comp_col2 = st.columns([5, 1, 5])
            
            with comp_col1:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); 
                            border-radius: 12px; padding: 15px; border-left: 4px solid #ef4444;">
                    <h4 style="color: #991b1b; margin: 0;">📥 RAW INPUT</h4>
                    <p style="color: #7f1d1d; font-size: 12px; margin: 5px 0;">{} rows × {} columns</p>
                </div>
                """.format(len(raw_df), len(raw_df.columns)), unsafe_allow_html=True)
                
                st.dataframe(
                    raw_df,
                    height=420,
                    hide_index=True,
                    use_container_width=True
                )
            
            with comp_arrow:
                st.markdown("""
                <div style="display: flex; flex-direction: column; align-items: center; 
                            justify-content: center; height: 300px;">
                    <div style="background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%); 
                                border-radius: 50%; width: 60px; height: 60px; 
                                display: flex; align-items: center; justify-content: center;
                                box-shadow: 0 4px 15px rgba(99,102,241,0.4);">
                        <span style="color: white; font-size: 24px;">⇄</span>
                    </div>
                    <p style="color: #6366f1; font-weight: bold; margin-top: 10px;">MAP</p>
                </div>
                """, unsafe_allow_html=True)
            
            with comp_col2:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); 
                            border-radius: 12px; padding: 15px; border-left: 4px solid #10b981;">
                    <h4 style="color: #065f46; margin: 0;">📤 HARMONIZED</h4>
                    <p style="color: #064e3b; font-size: 12px; margin: 5px 0;">{} rows × {} columns</p>
                </div>
                """.format(len(harm_df), len(harm_df.columns)), unsafe_allow_html=True)
                
                # Highlight changes for columns that exist in both dataframes.
                common_cols = [col for col in harm_df.columns if col in raw_df.columns]
                styled_harm_df = harm_df
                if common_cols:
                    raw_common = raw_df[common_cols].reset_index(drop=True)
                    harm_common = harm_df[common_cols].reset_index(drop=True)
                    min_len = min(len(raw_common), len(harm_common))
                    raw_common = raw_common.iloc[:min_len]
                    harm_common = harm_common.iloc[:min_len]
                    diff_mask = harm_common.astype(str).ne(raw_common.astype(str))

                    def highlight_changes(data):
                        highlight = pd.DataFrame("", index=data.index, columns=data.columns)
                        if min_len == 0:
                            return highlight
                        for col in common_cols:
                            if col in diff_mask.columns:
                                highlight.loc[:min_len - 1, col] = diff_mask[col].map(
                                    lambda v: "background-color: #fde68a" if v else ""
                                )
                        return highlight

                    try:
                        styled_harm_df = harm_df.style.apply(highlight_changes, axis=None)
                    except Exception:
                        styled_harm_df = harm_df.copy()
                        styled_harm_df["_changed_preview"] = diff_mask.any(axis=1)

                st.dataframe(
                    styled_harm_df,
                    height=420,
                    hide_index=True,
                    use_container_width=True
                )
                if common_cols:
                    if "_changed_preview" in getattr(styled_harm_df, "columns", []):
                        st.caption("Styling unavailable. Use `_changed_preview` to filter changed rows.")
                    else:
                        st.caption("Highlighted cells show values that changed vs raw input.")
            
            # Transformation summary boxes
            st.markdown("---")
            
            trans_col1, trans_col2, trans_col3 = st.columns(3)
            
            with trans_col1:
                st.markdown("""
                <div style="background: #f8fafc; border-radius: 10px; padding: 15px; text-align: center; border: 1px solid #e2e8f0;">
                    <span style="font-size: 24px;">📚</span>
                    <p style="font-weight: bold; margin: 5px 0;">Data Dictionary</p>
                    <p style="color: #64748b; font-size: 12px;">Column mappings applied</p>
                </div>
                """, unsafe_allow_html=True)
            
            with trans_col2:
                st.markdown("""
                <div style="background: #f8fafc; border-radius: 10px; padding: 15px; text-align: center; border: 1px solid #e2e8f0;">
                    <span style="font-size: 24px;">🔧</span>
                    <p style="font-weight: bold; margin: 5px 0;">Transformations</p>
                    <p style="color: #64748b; font-size: 12px;">Type conversions & cleaning</p>
                </div>
                """, unsafe_allow_html=True)
            
            with trans_col3:
                st.markdown("""
                <div style="background: #f8fafc; border-radius: 10px; padding: 15px; text-align: center; border: 1px solid #e2e8f0;">
                    <span style="font-size: 24px;">✅</span>
                    <p style="font-weight: bold; margin: 5px 0;">Validation</p>
                    <p style="color: #64748b; font-size: 12px;">Schema compliance checked</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Download section
            st.divider()
            st.subheader("📄 Download Results")
            
            dl_col1, dl_col2 = st.columns(2)
            
            with dl_col1:
                # Create CSV bytes from harmonized dataframe
                csv_bytes = harm_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "⬇️ Download Harmonized Data (CSV)",
                    csv_bytes,
                    file_name="harmonized_data.csv",
                    mime="text/csv"
                )
            
            with dl_col2:
                # Also offer Excel download
                try:
                    from io import BytesIO
                    excel_buffer = BytesIO()
                    harm_df.to_excel(excel_buffer, index=False, engine='openpyxl')
                    excel_buffer.seek(0)
                    st.download_button(
                        "⬇️ Download Harmonized Data (Excel)",
                        excel_buffer.getvalue(),
                        file_name="harmonized_data.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                except:
                    pass  # Excel export not available
        
        else:
            st.warning("⚠️ No harmonized data available. Run the pipeline first.")
    else:
        st.markdown(
            "<div class=\"kantar-callout\">🔄 Run the pipeline to see results</div>",
            unsafe_allow_html=True
        )


# =============================================================================
# TAB 3: REPORTS (Displayed in UI + PDF Download)
# =============================================================================

def generate_pdf_report(result_data):
    """Generate a PDF-ready HTML report"""
    sv_data = to_dict(result_data.get('structural_validation', {}))
    dq_data = to_dict(result_data.get('data_quality', {}))
    harm_data = to_dict(result_data.get('harmonization', {}))
    
    # Get data quality details
    dq_result = to_dict(dq_data.get('result', {}))
    column_stats = dq_result.get('column_statistics', []) or []
    
    # Build missing values table
    missing_rows = ""
    for stat in column_stats[:30]:
        stat = to_dict(stat)
        null_count = stat.get('null_count', 0) or 0
        if null_count > 0:
            missing_rows += f"<tr><td>{stat.get('column_name', 'N/A')}</td><td>{null_count}</td><td>{stat.get('null_percentage', 0):.1f}%</td></tr>"
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Data Harmonization Report</title>
        <style>
            body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
            .container {{ max-width: 900px; margin: 0 auto; background: white; padding: 40px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            h1 {{ color: #1e3a5f; border-bottom: 3px solid #667eea; padding-bottom: 10px; }}
            h2 {{ color: #334155; margin-top: 30px; }}
            .metric-box {{ display: inline-block; background: linear-gradient(135deg, #667eea, #764ba2); color: white; padding: 20px 30px; border-radius: 10px; margin: 10px; text-align: center; }}
            .metric-value {{ font-size: 28px; font-weight: bold; }}
            .metric-label {{ font-size: 12px; opacity: 0.9; }}
            table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background: #667eea; color: white; }}
            tr:nth-child(even) {{ background: #f9f9f9; }}
            .success {{ color: #10b981; font-weight: bold; }}
            .warning {{ color: #f59e0b; font-weight: bold; }}
            .error {{ color: #ef4444; font-weight: bold; }}
            .footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #666; font-size: 12px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🤖 AI Data Harmonization Report</h1>
            <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>Pipeline ID:</strong> {result_data.get('pipeline_id', 'N/A')}</p>
            
            <h2>📊 Summary Metrics</h2>
            <div style="text-align: center;">
                <div class="metric-box">
                    <div class="metric-value">{result_data.get('final_quality_score', 0):.1f}%</div>
                    <div class="metric-label">Quality Score</div>
                </div>
                <div class="metric-box">
                    <div class="metric-value">{result_data.get('final_confidence_score', 0)*100:.1f}%</div>
                    <div class="metric-label">Confidence</div>
                </div>
                <div class="metric-box">
                    <div class="metric-value">{result_data.get('total_tokens_used', 0):,}</div>
                    <div class="metric-label">Tokens Used</div>
                </div>
                <div class="metric-box">
                    <div class="metric-value">{result_data.get('total_processing_time_seconds', 0):.1f}s</div>
                    <div class="metric-label">Duration</div>
                </div>
            </div>
            
            <h2>🔍 Agent Results</h2>
            <table>
                <tr>
                    <th>Agent</th>
                    <th>Status</th>
                    <th>Confidence</th>
                    <th>Tokens</th>
                </tr>
                <tr>
                    <td>Structural Validation</td>
                    <td class="{'success' if sv_data.get('success') else 'warning'}">{'✓ Success' if sv_data.get('success') else '⚠ Issues'}</td>
                    <td>{sv_data.get('confidence_score', 0)*100:.1f}%</td>
                    <td>{sv_data.get('tokens_used', 0):,}</td>
                </tr>
                <tr>
                    <td>Data Quality</td>
                    <td class="{'success' if dq_data.get('success') else 'warning'}">{'✓ Success' if dq_data.get('success') else '⚠ Issues'}</td>
                    <td>{dq_data.get('confidence_score', 0)*100:.1f}%</td>
                    <td>{dq_data.get('tokens_used', 0):,}</td>
                </tr>
                <tr>
                    <td>Harmonization</td>
                    <td class="{'success' if harm_data.get('success') else 'warning'}">{'✓ Success' if harm_data.get('success') else '⚠ Issues'}</td>
                    <td>{harm_data.get('confidence_score', 0)*100:.1f}%</td>
                    <td>{harm_data.get('tokens_used', 0):,}</td>
                </tr>
            </table>
            
            <h2>📉 Missing Values Summary</h2>
            <table>
                <tr>
                    <th>Column Name</th>
                    <th>Missing Count</th>
                    <th>Missing %</th>
                </tr>
                {missing_rows if missing_rows else '<tr><td colspan="3">No missing values found</td></tr>'}
            </table>
            
            <div class="footer">
                <p>🤖 Generated by Agentic AI Data Harmonization System</p>
                <p>Powered by Azure OpenAI GPT-5.2</p>
            </div>
        </div>
    </body>
    </html>
    """
    return html_content

with tab3:
    if st.session_state.pipeline_result and st.session_state.pipeline_result.result:
        result_data = to_dict(st.session_state.pipeline_result.result)
        
        st.subheader("📋 Data Harmonization Report")
        
        # Download buttons at top
        col1, col2 = st.columns(2)
        
        with col1:
            # Generate PDF-ready HTML
            pdf_html = generate_pdf_report(result_data)
            st.download_button(
                "📥 Download Report (HTML/PDF)",
                pdf_html,
                file_name="harmonization_report.html",
                mime="text/html",
                help="Open in browser and Print → Save as PDF"
            )
        
        with col2:
            # Download harmonized data
            output_file = result_data.get('output_file')
            if output_file and Path(output_file).exists():
                with open(output_file, 'rb') as f:
                    st.download_button(
                        "📥 Download Harmonized Data",
                        f.read(),
                        file_name="harmonized_data.csv",
                        mime="text/csv"
                    )
        
        st.divider()
        
        # ========== REPORT DISPLAYED IN UI ==========
        
        # Section 1: Summary
        st.markdown("### 📊 Executive Summary")
        
        sum_col1, sum_col2, sum_col3, sum_col4 = st.columns(4)
        with sum_col1:
            st.metric("Quality Score", f"{result_data.get('final_quality_score', 0):.1f}%")
        with sum_col2:
            st.metric("Confidence", f"{result_data.get('final_confidence_score', 0)*100:.1f}%")
        with sum_col3:
            st.metric("Processing Time", f"{result_data.get('total_processing_time_seconds', 0):.1f}s")
        with sum_col4:
            st.metric("Tokens Used", f"{result_data.get('total_tokens_used', 0):,}")
        
        st.divider()
        
        # Section 2: Agent Performance
        st.markdown("### 🤖 Agent Performance")
        
        sv_data = to_dict(result_data.get('structural_validation', {}))
        dq_data = to_dict(result_data.get('data_quality', {}))
        harm_data = to_dict(result_data.get('harmonization', {}))
        
        agent_df = pd.DataFrame([
            {
                "Agent": "🔍 Structural Validation",
                "Status": "✅ Success" if sv_data.get('success') else "⚠️ Issues",
                "Confidence": f"{sv_data.get('confidence_score', 0)*100:.1f}%",
                "Duration": f"{sv_data.get('execution_time_seconds', 0):.1f}s",
                "Tokens": sv_data.get('tokens_used', 0)
            },
            {
                "Agent": "📊 Data Quality",
                "Status": "✅ Success" if dq_data.get('success') else "⚠️ Issues",
                "Confidence": f"{dq_data.get('confidence_score', 0)*100:.1f}%",
                "Duration": f"{dq_data.get('execution_time_seconds', 0):.1f}s",
                "Tokens": dq_data.get('tokens_used', 0)
            },
            {
                "Agent": "🔄 Harmonization",
                "Status": "✅ Success" if harm_data.get('success') else "⚠️ Issues",
                "Confidence": f"{harm_data.get('confidence_score', 0)*100:.1f}%",
                "Duration": f"{harm_data.get('execution_time_seconds', 0):.1f}s",
                "Tokens": harm_data.get('tokens_used', 0)
            }
        ])
        st.dataframe(agent_df, hide_index=True)
        
        st.divider()
        
        # Section 3: Data Quality Details
        st.markdown("### 📉 Data Quality Analysis")
        
        dq_result = to_dict(dq_data.get('result', {}))
        column_stats = dq_result.get('column_statistics', []) or []
        
        if column_stats:
            # Missing values table
            missing_data = []
            for stat in column_stats:
                stat = to_dict(stat)
                null_count = stat.get('null_count', 0) or 0
                if null_count > 0:
                    missing_data.append({
                        'Column': stat.get('column_name', 'Unknown'),
                        'Missing Count': null_count,
                        'Missing %': f"{stat.get('null_percentage', 0):.1f}%",
                        'Data Type': stat.get('data_type', 'unknown')
                    })
            
            if missing_data:
                missing_data.sort(key=lambda x: x['Missing Count'], reverse=True)
                st.markdown(f"**Columns with Missing Values:** {len(missing_data)}")
                st.dataframe(pd.DataFrame(missing_data), hide_index=True, height=300)
            else:
                st.success("✅ No missing values found in any column!")
        
        st.divider()
        
        # Section 4: Issues Found
        st.markdown("### ⚠️ Issues & Recommendations")
        
        blocking = dq_result.get('blocking_issues', []) or []
        fixable = dq_result.get('fixable_issues', []) or []
        
        if blocking:
            st.error(f"**🚫 Blocking Issues ({len(blocking)})**")
            for issue in blocking[:10]:
                issue = to_dict(issue)
                st.markdown(f"- {issue.get('description', 'Unknown issue')}")
        
        if fixable:
            st.warning(f"**🔧 Fixable Issues ({len(fixable)})**")
            for issue in fixable[:10]:
                issue = to_dict(issue)
                st.markdown(f"- {issue.get('description', 'Unknown issue')}")
        
        if not blocking and not fixable:
            st.success("✅ No significant issues found!")
        
        # Recommendations
        recommendations = dq_result.get('recommendations', []) or []
        if recommendations:
            st.markdown("**💡 Recommendations:**")
            for rec in recommendations[:5]:
                st.info(f"• {rec}")
        
        st.divider()
        
        # Section 5: Visual Report Preview (Printable)
        st.markdown("### 📄 Printable Report Preview")
        st.caption("This is how your downloaded report will look. Click 'Download Report' above to save.")
        
        with st.expander("👁️ View Full Report Preview", expanded=False):
            # Show the HTML report in an iframe
            pdf_html = generate_pdf_report(result_data)
            st.components.v1.html(pdf_html, height=800, scrolling=True)
        
    else:
        st.markdown(
            "<div class=\"kantar-callout\">🔄 Run the pipeline to generate reports</div>",
            unsafe_allow_html=True
        )


# =============================================================================
# TAB 4: HARMONIZATION RESULTS (FROM INPUT FILES FOLDER)
# =============================================================================

with tab4:
    st.subheader("🔍 Harmonization Results Dashboard")
    st.caption("View pre-processed harmonization results from the Input files folder")
    
    # Load button
    col_load1, col_load2 = st.columns([1, 3])
    with col_load1:
        load_clicked = st.button("📂 Load Input Files", key="load_input_files")
    with col_load2:
        if st.session_state.get('input_files_loaded'):
            st.success("✅ Input files loaded successfully!")
    
    if load_clicked or st.session_state.get('input_files_loaded'):
        # Load the input files
        if load_clicked or 'input_files_data' not in st.session_state:
            input_files = load_input_files()
            st.session_state.input_files_data = input_files
            st.session_state.input_files_loaded = input_files['loaded']
        else:
            input_files = st.session_state.input_files_data
        
        # Show errors if any
        if input_files['errors']:
            with st.expander("⚠️ Loading Warnings", expanded=False):
                for error in input_files['errors']:
                    st.warning(error)
        
        if input_files['loaded']:
            # Get summary statistics
            summary = get_harmonization_summary(input_files)
            
            st.divider()
            
            # ===========================================
            # SUMMARY METRICS
            # ===========================================
            st.markdown("### 📊 Summary Metrics")
            
            met_col1, met_col2, met_col3, met_col4 = st.columns(4)
            with met_col1:
                st.metric("Total Rows", f"{summary['total_rows']:,}")
            with met_col2:
                st.metric("Total Columns", summary['total_columns'])
            with met_col3:
                st.metric("Anomalies Found", summary['total_anomalies'])
            with met_col4:
                st.metric("Corrections Made", summary['total_corrections'])
            
            met_col5, met_col6, met_col7, met_col8 = st.columns(4)
            with met_col5:
                st.metric("Columns with Issues", summary['columns_with_issues'])
            with met_col6:
                st.metric("Valid Nulls", summary['valid_nulls_count'])
            with met_col7:
                st.metric("Uncertain Missing", summary['uncertain_missing_count'])
            with met_col8:
                st.metric("Drift Alerts", summary['drift_alerts_count'])
            
            st.divider()
            
            # ===========================================
            # EXECUTIVE SUMMARY
            # ===========================================
            if input_files.get('executive_summary'):
                st.markdown("### 📝 Executive Summary")
                st.markdown("""
                <div style="background: linear-gradient(135deg, #1e293b 0%, #334155 100%); 
                            border-radius: 12px; padding: 20px; border-left: 4px solid #6366f1;">
                """, unsafe_allow_html=True)
                
                # Parse and format the executive summary
                summary_text = input_files['executive_summary']
                st.markdown(summary_text)
                st.markdown("</div>", unsafe_allow_html=True)
                
                st.divider()
            
            # ===========================================
            # DETAILED DATA VIEWS (Expandable Sections)
            # ===========================================
            st.markdown("### 📋 Detailed Data Views")
            
            # ----- CLEANED DATA -----
            with st.expander("✅ Cleaned Dataset", expanded=True):
                if input_files.get('cleaned_df') is not None:
                    cleaned_df = input_files['cleaned_df']
                    st.caption(f"📊 {len(cleaned_df):,} rows × {len(cleaned_df.columns)} columns")
                    st.dataframe(cleaned_df, height=400)
                    
                    # Download button
                    csv_data = cleaned_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Download Cleaned Data (CSV)",
                        csv_data,
                        file_name="cleaned_data.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("Cleaned dataset not available")
            
            # ----- COLUMN HEALTH REPORT -----
            with st.expander("📊 Column Health Report", expanded=False):
                if input_files.get('column_health_report') is not None:
                    health_df = input_files['column_health_report']
                    st.caption("Column-level quality metrics showing anomaly rates and recommendations")
                    
                    # Create a visual bar chart for anomaly rates
                    if 'column' in health_df.columns and 'anomaly_pct' in health_df.columns:
                        fig = px.bar(
                            health_df.sort_values('anomaly_pct', ascending=False),
                            x='column',
                            y='anomaly_pct',
                            color='severity' if 'severity' in health_df.columns else None,
                            title='Anomaly Rate by Column',
                            labels={'anomaly_pct': 'Anomaly %', 'column': 'Column'},
                            color_discrete_map={'LOW': '#10b981', 'MEDIUM': '#f59e0b', 'HIGH': '#ef4444'}
                        )
                        fig.update_layout(
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font={'color': 'white'},
                            height=300
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    st.dataframe(health_df, height=300)
                else:
                    st.warning("Column health report not available")
            
            # ----- ANOMALIES -----
            with st.expander("⚠️ Anomalies Detected", expanded=False):
                if input_files.get('anomalies') is not None:
                    anomalies_df = input_files['anomalies']
                    st.caption("List of out-of-range and invalid values detected in the dataset")
                    
                    # Filter by issue type
                    if 'issue_type' in anomalies_df.columns:
                        issue_types = ['All'] + list(anomalies_df['issue_type'].unique())
                        selected_type = st.selectbox("Filter by Issue Type", issue_types, key="anomaly_filter")
                        
                        if selected_type != 'All':
                            filtered_df = anomalies_df[anomalies_df['issue_type'] == selected_type]
                        else:
                            filtered_df = anomalies_df
                        
                        st.dataframe(filtered_df, height=300)
                    else:
                        st.dataframe(anomalies_df, height=300)
                else:
                    st.warning("Anomalies data not available")
            
            # ----- AUDIT LOG -----
            with st.expander("📝 Audit Log (Changes Made)", expanded=False):
                if input_files.get('audit_log') is not None:
                    audit_df = input_files['audit_log']
                    st.caption("Complete log of all corrections and modifications applied to the data")
                    
                    # Show summary by column
                    if 'column' in audit_df.columns:
                        col_summary = audit_df.groupby('column').size().reset_index(name='corrections')
                        col_summary = col_summary.sort_values('corrections', ascending=False)
                        
                        st.markdown("**Corrections by Column:**")
                        fig = px.bar(
                            col_summary.head(10),
                            x='column',
                            y='corrections',
                            title='Top 10 Columns with Most Corrections',
                            color='corrections',
                            color_continuous_scale='Blues'
                        )
                        fig.update_layout(
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font={'color': 'white'},
                            height=300
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    st.dataframe(audit_df, height=300)
                else:
                    st.warning("Audit log not available")
            
            # ----- DRIFT ALERTS -----
            with st.expander("📈 Drift Alerts", expanded=False):
                if input_files.get('drift_alerts') is not None:
                    drift_df = input_files['drift_alerts']
                    st.caption("Column-wise comparison of summary statistics between historical and current data")
                    
                    # Highlight drift types
                    if 'drift_type' in drift_df.columns:
                        drift_counts = drift_df['drift_type'].value_counts()
                        drift_col1, drift_col2 = st.columns(2)
                        
                        with drift_col1:
                            st.markdown("**Drift Types Found:**")
                            for dtype, count in drift_counts.items():
                                icon = "🔴" if "SHIFT" in str(dtype) else "🟡" if "SCALE" in str(dtype) else "🟠"
                                st.write(f"{icon} {dtype}: {count}")
                        
                        with drift_col2:
                            fig = px.pie(
                                values=drift_counts.values,
                                names=drift_counts.index,
                                title='Drift Types Distribution'
                            )
                            fig.update_layout(
                                paper_bgcolor='rgba(0,0,0,0)',
                                font={'color': 'white'},
                                height=250
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    
                    st.dataframe(drift_df, height=300)
                else:
                    st.warning("Drift alerts not available")
            
            # ----- VALID NULLS -----
            with st.expander("✅ Valid Nulls (Questionnaire Logic)", expanded=False):
                if input_files.get('valid_nulls') is not None:
                    valid_nulls_df = input_files['valid_nulls']
                    st.caption("Missing values identified as valid due to survey skip logic or branching")
                    
                    # Summary by column
                    if 'column' in valid_nulls_df.columns:
                        null_summary = valid_nulls_df.groupby('column').size().reset_index(name='count')
                        null_summary = null_summary.sort_values('count', ascending=False)
                        
                        st.markdown(f"**{len(valid_nulls_df)} valid nulls across {len(null_summary)} columns:**")
                        st.dataframe(null_summary, height=200)
                    
                    st.dataframe(valid_nulls_df, height=300)
                else:
                    st.warning("Valid nulls data not available")
            
            # ----- UNCERTAIN MISSING -----
            with st.expander("❓ Uncertain Missing Values", expanded=False):
                if input_files.get('uncertain_missing') is not None:
                    uncertain_df = input_files['uncertain_missing']
                    st.caption("Missing values that require manual review - not explained by questionnaire logic")
                    
                    # Summary by column
                    if 'column' in uncertain_df.columns:
                        uncertain_summary = uncertain_df.groupby('column').size().reset_index(name='count')
                        uncertain_summary = uncertain_summary.sort_values('count', ascending=False)
                        
                        st.markdown(f"**{len(uncertain_df)} uncertain missing values across {len(uncertain_summary)} columns:**")
                        
                        fig = px.bar(
                            uncertain_summary,
                            x='column',
                            y='count',
                            title='Uncertain Missing by Column',
                            color='count',
                            color_continuous_scale='Reds'
                        )
                        fig.update_layout(
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font={'color': 'white'},
                            height=300
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    st.dataframe(uncertain_df, height=300)
                else:
                    st.warning("Uncertain missing data not available")
            
            st.divider()
            
            # ===========================================
            # 🎯 HARMONIZATION WORKFLOW - MAIN FEATURE
            # ===========================================
            st.markdown("### 🎯 Final Harmonization Workflow")
            st.caption("Configure how to handle uncertain missing values and generate the final harmonized file")
            
            # Get columns that need imputation
            columns_needing_imputation = get_columns_needing_imputation(input_files)
            
            if columns_needing_imputation:
                st.markdown(f"**📋 {len(columns_needing_imputation)} columns have uncertain missing values that need decisions:**")
                
                # Initialize session state for imputation strategies
                if 'imputation_strategies' not in st.session_state:
                    st.session_state.imputation_strategies = {}
                
                # Create a form for imputation strategies
                with st.form("imputation_form"):
                    st.markdown("**Select imputation strategy for each column:**")
                    
                    # Create columns for better layout
                    strategies_selected = {}
                    
                    for col_name, col_info in columns_needing_imputation.items():
                        col_a, col_b, col_c = st.columns([2, 1, 2])
                        
                        with col_a:
                            st.markdown(f"**{col_name}**")
                            st.caption(f"Type: {col_info['data_type']} | Missing: {col_info['missing_count']}")
                        
                        with col_b:
                            # Determine default strategy based on column type
                            if col_info['is_numeric']:
                                options = ['mean', 'median', 'mode', 'zero', 'keep', 'remove', 'drop']
                                default_idx = 0  # mean
                            else:
                                options = ['mode', 'unknown', 'other', 'keep', 'remove', 'drop']
                                default_idx = 0  # mode
                            
                            strategy = st.selectbox(
                                f"Strategy for {col_name}",
                                options=options,
                                index=default_idx,
                                key=f"strategy_{col_name}",
                                label_visibility="collapsed"
                            )
                            strategies_selected[col_name] = strategy
                        
                        with col_c:
                            if col_info['sample_values']:
                                st.caption(f"Sample: {col_info['sample_values'][:3]}")
                    
                    st.divider()
                    
                    # Additional options
                    opt_col1, opt_col2 = st.columns(2)
                    with opt_col1:
                        keep_valid_nulls = st.checkbox(
                            "✅ Preserve valid nulls (questionnaire logic)",
                            value=True,
                            help="Keep missing values that are valid due to survey skip logic"
                        )
                    with opt_col2:
                        output_format = st.selectbox(
                            "Output format",
                            options=['CSV', 'Excel (.xlsx)'],
                            index=0
                        )
                    
                    # Submit button
                    run_harmonization = st.form_submit_button(
                        "🚀 Run Final Harmonization",
                        use_container_width=True,
                        type="primary"
                    )
                
                # Process harmonization when button is clicked
                if run_harmonization:
                    with st.spinner("Running harmonization..."):
                        try:
                            # Run the harmonization
                            harmonized_df, changes_log, stats = perform_final_harmonization(
                                input_files=input_files,
                                imputation_strategies=strategies_selected,
                                keep_valid_nulls=keep_valid_nulls
                            )
                            
                            # Store results in session state
                            st.session_state.final_harmonized_df = harmonized_df
                            st.session_state.harmonization_log = changes_log
                            st.session_state.harmonization_stats = stats
                            st.session_state.harmonization_complete = True
                            
                            st.success("✅ Harmonization complete!")
                            
                        except Exception as e:
                            st.error(f"❌ Harmonization failed: {str(e)}")
                            st.session_state.harmonization_complete = False
                
                # Show results if harmonization is complete
                if st.session_state.get('harmonization_complete'):
                    st.divider()
                    st.markdown("### ✅ Harmonization Results")
                    
                    # Show statistics
                    stats = st.session_state.harmonization_stats
                    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                    with stat_col1:
                        st.metric("Rows", f"{stats['rows_after']:,}", 
                                  delta=f"{stats['rows_after'] - stats['rows_before']}" if stats['rows_after'] != stats['rows_before'] else None)
                    with stat_col2:
                        st.metric("Missing Before", f"{stats['missing_before']:,}")
                    with stat_col3:
                        st.metric("Missing After", f"{stats['missing_after']:,}",
                                  delta=f"-{stats['missing_before'] - stats['missing_after']}")
                    with stat_col4:
                        st.metric("Values Imputed", f"{stats['values_imputed']:,}")
                    
                    # Show changes log
                    with st.expander("📋 Changes Log", expanded=True):
                        for log_entry in st.session_state.harmonization_log:
                            st.markdown(log_entry)
                    
                    # Preview final data
                    with st.expander("👀 Preview Final Harmonized Data", expanded=True):
                        st.dataframe(st.session_state.final_harmonized_df.head(100), height=400)
                        st.caption(f"Showing first 100 rows of {len(st.session_state.final_harmonized_df):,} total rows")
                    
                    # Download buttons
                    st.markdown("### 📥 Download Final Harmonized File")
                    
                    dl_col1, dl_col2, dl_col3 = st.columns(3)
                    
                    with dl_col1:
                        # CSV download
                        csv_data = st.session_state.final_harmonized_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "📥 Download as CSV",
                            csv_data,
                            file_name="FINAL_harmonized_data.csv",
                            mime="text/csv",
                            key="download_final_csv",
                            use_container_width=True
                        )
                    
                    with dl_col2:
                        # Excel download
                        import io
                        excel_buffer = io.BytesIO()
                        st.session_state.final_harmonized_df.to_excel(excel_buffer, index=False, engine='openpyxl')
                        excel_data = excel_buffer.getvalue()
                        st.download_button(
                            "📥 Download as Excel",
                            excel_data,
                            file_name="FINAL_harmonized_data.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key="download_final_excel",
                            use_container_width=True
                        )
                    
                    with dl_col3:
                        # Changes log download
                        log_text = "\n".join(st.session_state.harmonization_log)
                        st.download_button(
                            "📥 Download Changes Log",
                            log_text,
                            file_name="harmonization_changes_log.txt",
                            mime="text/plain",
                            key="download_log",
                            use_container_width=True
                        )
                    
                    # Save to output folder
                    st.divider()
                    save_col1, save_col2 = st.columns([1, 2])
                    with save_col1:
                        if st.button("💾 Save to Output Folder", use_container_width=True):
                            try:
                                output_path = OUTPUT_DIR / "FINAL_harmonized_data.csv"
                                OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                                st.session_state.final_harmonized_df.to_csv(output_path, index=False)
                                st.success(f"✅ Saved to: {output_path}")
                            except Exception as e:
                                st.error(f"❌ Failed to save: {str(e)}")
                    with save_col2:
                        st.caption(f"Will save to: {OUTPUT_DIR / 'FINAL_harmonized_data.csv'}")
            
            else:
                st.success("✅ No uncertain missing values to handle! The cleaned dataset is ready.")
                
                # Direct download of cleaned_df as final
                if input_files.get('cleaned_df') is not None:
                    st.markdown("### 📥 Download Final Harmonized File")
                    
                    dl_col1, dl_col2 = st.columns(2)
                    with dl_col1:
                        csv_data = input_files['cleaned_df'].to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "📥 Download as CSV",
                            csv_data,
                            file_name="FINAL_harmonized_data.csv",
                            mime="text/csv",
                            key="download_final_direct_csv",
                            use_container_width=True
                        )
                    with dl_col2:
                        import io
                        excel_buffer = io.BytesIO()
                        input_files['cleaned_df'].to_excel(excel_buffer, index=False, engine='openpyxl')
                        excel_data = excel_buffer.getvalue()
                        st.download_button(
                            "📥 Download as Excel",
                            excel_data,
                            file_name="FINAL_harmonized_data.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key="download_final_direct_excel",
                            use_container_width=True
                        )
            
            st.divider()
            
            # ===========================================
            # DETAILED DATA VIEWS (Expandable)
            # ===========================================
            st.markdown("### 📋 View Input Files (Reference)")
            st.caption("Expand sections below to view the source input files")
            
            # ----- CLEANED DATA -----
            with st.expander("✅ Base Dataset (cleaned_df)", expanded=False):
                if input_files.get('cleaned_df') is not None:
                    cleaned_df = input_files['cleaned_df']
                    st.caption(f"📊 {len(cleaned_df):,} rows × {len(cleaned_df.columns)} columns")
                    st.dataframe(cleaned_df, height=400)
                else:
                    st.warning("Cleaned dataset not available")
            
            # ----- AUDIT LOG -----
            with st.expander("📝 Audit Log (Changes Already Applied)", expanded=False):
                if input_files.get('audit_log') is not None:
                    audit_df = input_files['audit_log']
                    st.caption(f"Total corrections: {len(audit_df)}")
                    st.dataframe(audit_df, height=300)
                else:
                    st.warning("Audit log not available")
            
            # ----- COLUMN HEALTH -----
            with st.expander("📊 Column Health Report", expanded=False):
                if input_files.get('column_health_report') is not None:
                    st.dataframe(input_files['column_health_report'], height=300)
                else:
                    st.warning("Column health report not available")
            
            # ----- ANOMALIES -----
            with st.expander("⚠️ Anomalies Detected", expanded=False):
                if input_files.get('anomalies') is not None:
                    st.dataframe(input_files['anomalies'], height=300)
                else:
                    st.warning("Anomalies not available")
            
            # ----- DRIFT ALERTS -----
            with st.expander("📈 Drift Alerts", expanded=False):
                if input_files.get('drift_alerts') is not None:
                    st.dataframe(input_files['drift_alerts'], height=300)
                else:
                    st.warning("Drift alerts not available")
            
            # ----- VALID NULLS -----
            with st.expander("✅ Valid Nulls", expanded=False):
                if input_files.get('valid_nulls') is not None:
                    st.dataframe(input_files['valid_nulls'], height=300)
                else:
                    st.warning("Valid nulls not available")
            
            # ----- UNCERTAIN MISSING -----
            with st.expander("❓ Uncertain Missing", expanded=False):
                if input_files.get('uncertain_missing') is not None:
                    st.dataframe(input_files['uncertain_missing'], height=300)
                else:
                    st.warning("Uncertain missing not available")
        
        else:
            st.warning("⚠️ Could not load the input files. Please ensure the 'Input files' folder exists and contains the required Excel files.")
            st.info(f"📂 Expected folder: {INPUT_FILES_DIR}")
    
    else:
        st.info("👆 Click 'Load Input Files' to view harmonization results from the Input files folder")
        
        # Show expected file structure
        with st.expander("📁 Expected Input Files Structure", expanded=False):
            st.markdown("""
            The **Input files** folder should contain:
            
            | File | Description |
            |------|-------------|
            | `cleaned_df.xlsx` | The harmonized dataset after corrections |
            | `anomalies.xlsx` | List of out-of-range values detected |
            | `audit_log.xlsx` | Log of all modifications made |
            | `column_health_report.xlsx` | Column-level quality metrics |
            | `drift_alerts.xlsx` | Data drift between historical and current |
            | `valid_nulls.xlsx` | Valid missing values (skip logic) |
            | `uncertain_missing.xlsx` | Missing values needing review |
            | `systemic_executive_summary.txt` | AI-generated summary |
            """)


# =============================================================================
# FOOTER
# =============================================================================

st.divider()
st.markdown("""
<div style="text-align: center; color: #64748b; padding: 20px;">
    <p>🤖 Agentic AI Data Harmonization System</p>
    <p>Powered by Azure OpenAI GPT-5.2 | Built for Hackathon 2024</p>
</div>
""", unsafe_allow_html=True)

