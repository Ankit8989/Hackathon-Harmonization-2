"""
Helper utilities for multi-source harmonization workflows.
Provides profiling, mapping, calibration, and validation helpers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml


@dataclass
class MappingCandidate:
    source_column: str
    target_column: str
    confidence: float
    name_similarity: float
    label_similarity: float
    distribution_similarity: float


def _infer_series_type(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_integer_dtype(series):
        return "integer"
    if pd.api.types.is_float_dtype(series):
        return "float"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_categorical_dtype(series):
        return "categorical"
    if pd.api.types.is_object_dtype(series):
        non_null = series.dropna()
        if len(non_null) > 0:
            unique_ratio = non_null.nunique() / len(non_null)
            if unique_ratio < 0.05:
                return "categorical"
        return "string"
    return "unknown"


def profile_dataframe(
    df: pd.DataFrame,
    sample_rows: int = 5,
    max_levels: int = 20
) -> Dict[str, Any]:
    """Generate a structural profile for a dataframe."""
    profile: Dict[str, Any] = {
        "shape": {"rows": int(len(df)), "columns": int(len(df.columns))},
        "head": df.head(sample_rows).to_dict(orient="records"),
        "tail": df.tail(sample_rows).to_dict(orient="records"),
        "columns": {},
        "missing_patterns": {
            "missing_by_column": {},
            "missing_by_row": {}
        },
        "relationships": {"numeric_correlations": []}
    }

    missing_by_row = df.isna().sum(axis=1).value_counts().head(10).to_dict()
    profile["missing_patterns"]["missing_by_row"] = {
        str(k): int(v) for k, v in missing_by_row.items()
    }

    for col in df.columns:
        series = df[col]
        inferred_type = _infer_series_type(series)
        col_profile: Dict[str, Any] = {
            "dtype": str(series.dtype),
            "inferred_type": inferred_type,
            "unique_count": int(series.nunique(dropna=True)),
            "missing_count": int(series.isna().sum()),
            "missing_pct": round(series.isna().sum() / len(df) * 100, 2) if len(df) else 0,
            "sample_values": series.dropna().head(5).astype(str).tolist()
        }

        if pd.api.types.is_numeric_dtype(series):
            non_null = series.dropna()
            if len(non_null) > 0:
                col_profile.update({
                    "min": float(non_null.min()),
                    "max": float(non_null.max()),
                    "mean": float(non_null.mean()),
                    "median": float(non_null.median()),
                    "std": float(non_null.std()) if len(non_null) > 1 else 0.0,
                    "scale_range": float(non_null.max() - non_null.min())
                })
        else:
            value_counts = series.value_counts(dropna=True).head(max_levels)
            col_profile["levels"] = {
                str(k): int(v) for k, v in value_counts.items()
            }

        profile["columns"][col] = col_profile
        profile["missing_patterns"]["missing_by_column"][col] = col_profile["missing_pct"]

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr()
        pairs = []
        for i, col_a in enumerate(numeric_cols):
            for col_b in numeric_cols[i + 1:]:
                if pd.notna(corr.loc[col_a, col_b]):
                    pairs.append({
                        "column_a": col_a,
                        "column_b": col_b,
                        "correlation": float(corr.loc[col_a, col_b])
                    })
        pairs = sorted(pairs, key=lambda x: abs(x["correlation"]), reverse=True)
        profile["relationships"]["numeric_correlations"] = pairs[:10]

    return profile


def load_validation_rules(rules_path: Optional[str]) -> List[Dict[str, Any]]:
    """Load validation rules from JSON, YAML, or line-delimited text."""
    if not rules_path:
        return []
    path = Path(rules_path)
    if not path.exists():
        return []

    if path.suffix.lower() in {".json"}:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    elif path.suffix.lower() in {".yaml", ".yml"}:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    else:
        with open(path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        return [{"rule": line} for line in lines]

    if isinstance(data, dict) and "rules" in data:
        return data["rules"] or []
    if isinstance(data, list):
        return data
    return []


def apply_validation_rules(
    df: pd.DataFrame,
    rules: List[Dict[str, Any]],
    dataset_name: str,
    record_id_column: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Apply validation rules and return validation flags."""
    flags: List[Dict[str, Any]] = []
    if not rules:
        return flags

    record_id_series = df[record_id_column] if record_id_column and record_id_column in df.columns else None

    for rule in rules:
        if "min_records" in rule:
            min_records = int(rule["min_records"])
            if len(df) < min_records:
                flags.append({
                    "dataset": dataset_name,
                    "record_id": None,
                    "column": None,
                    "issue_type": "min_records",
                    "details": f"Dataset has {len(df)} rows, minimum is {min_records}"
                })
            continue

        if "max_records" in rule:
            max_records = int(rule["max_records"])
            if len(df) > max_records:
                flags.append({
                    "dataset": dataset_name,
                    "record_id": None,
                    "column": None,
                    "issue_type": "max_records",
                    "details": f"Dataset has {len(df)} rows, maximum is {max_records}"
                })
            continue

        column = rule.get("column")
        if not column or column not in df.columns:
            continue

        series = df[column]
        min_val = rule.get("min")
        max_val = rule.get("max")
        allowed = rule.get("allowed_values")
        required = rule.get("required", False)

        if required:
            missing_mask = series.isna()
            for idx in series[missing_mask].index:
                flags.append({
                    "dataset": dataset_name,
                    "record_id": record_id_series.loc[idx] if record_id_series is not None else None,
                    "column": column,
                    "issue_type": "required_missing",
                    "details": "Required value missing"
                })

        if min_val is not None or max_val is not None:
            numeric = pd.to_numeric(series, errors="coerce")
            if min_val is not None:
                bad_idx = numeric[numeric < min_val].index
                for idx in bad_idx:
                    flags.append({
                        "dataset": dataset_name,
                        "record_id": record_id_series.loc[idx] if record_id_series is not None else None,
                        "column": column,
                        "issue_type": "below_min",
                        "details": f"Value {series.loc[idx]} below min {min_val}"
                    })
            if max_val is not None:
                bad_idx = numeric[numeric > max_val].index
                for idx in bad_idx:
                    flags.append({
                        "dataset": dataset_name,
                        "record_id": record_id_series.loc[idx] if record_id_series is not None else None,
                        "column": column,
                        "issue_type": "above_max",
                        "details": f"Value {series.loc[idx]} above max {max_val}"
                    })

        if allowed:
            allowed_set = {str(v) for v in allowed}
            bad_idx = series[series.notna() & ~series.astype(str).isin(allowed_set)].index
            for idx in bad_idx:
                flags.append({
                    "dataset": dataset_name,
                    "record_id": record_id_series.loc[idx] if record_id_series is not None else None,
                    "column": column,
                    "issue_type": "invalid_value",
                    "details": f"Value {series.loc[idx]} not in allowed list"
                })

    return flags


def build_default_value_mappings() -> Dict[str, Dict[str, str]]:
    """Default level mappings for age groups and markets."""
    age_map = {
        "18-24": "18-24",
        "18_24": "18-24",
        "18 to 24": "18-24",
        "18-25": "18-24",
        "25-34": "25-34",
        "25_34": "25-34",
        "25 to 34": "25-34",
        "35-44": "35-44",
        "35_44": "35-44",
        "35 to 44": "35-44",
        "45-54": "45-54",
        "45_54": "45-54",
        "45 to 54": "45-54",
        "55+": "55+",
        "55 plus": "55+",
        "56-65": "55+",
        "65+": "55+",
        "under 18": "<18",
        "<18": "<18"
    }

    market_map = {
        "US": "USA",
        "USA": "USA",
        "UNITED STATES": "USA",
        "UK": "UK",
        "UNITED KINGDOM": "UK",
        "GB": "UK",
        "DE": "Germany",
        "GERMANY": "Germany",
        "FR": "France",
        "FRANCE": "France",
        "ES": "Spain",
        "SPAIN": "Spain",
        "IT": "Italy",
        "ITALY": "Italy",
        "NL": "Netherlands",
        "NETHERLANDS": "Netherlands",
        "BE": "Belgium",
        "BELGIUM": "Belgium",
        "JP": "Japan",
        "JAPAN": "Japan",
        "AU": "Australia",
        "AUSTRALIA": "Australia",
        "CA": "Canada",
        "CANADA": "Canada",
        "BR": "Brazil",
        "BRAZIL": "Brazil",
        "IN": "India",
        "INDIA": "India",
        "CN": "China",
        "CHINA": "China"
    }

    return {
        "age": age_map,
        "age_group": age_map,
        "market": market_map,
        "country": market_map
    }


def impute_missing_values(
    df: pd.DataFrame,
    dataset_name: str
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Impute missing values using type-aware strategies."""
    df_imputed = df.copy()
    imputation_summary: List[Dict[str, Any]] = []

    for col in df_imputed.columns:
        series = df_imputed[col]
        missing_count = int(series.isna().sum())
        if missing_count == 0:
            continue

        inferred_type = _infer_series_type(series)
        strategy = None
        fill_value = None

        if pd.api.types.is_numeric_dtype(series):
            skew = series.dropna().skew() if series.dropna().shape[0] > 1 else 0
            if abs(skew) > 1:
                strategy = "median"
                fill_value = series.median()
            else:
                strategy = "mean"
                fill_value = series.mean()
        elif inferred_type in {"categorical", "string"}:
            strategy = "mode"
            mode_vals = series.mode(dropna=True)
            fill_value = mode_vals.iloc[0] if not mode_vals.empty else "Unknown"
        elif inferred_type == "boolean":
            strategy = "mode"
            mode_vals = series.mode(dropna=True)
            fill_value = mode_vals.iloc[0] if not mode_vals.empty else False
        else:
            strategy = "mode"
            mode_vals = series.mode(dropna=True)
            fill_value = mode_vals.iloc[0] if not mode_vals.empty else "Unknown"

        df_imputed[col] = series.fillna(fill_value)
        imputation_summary.append({
            "dataset": dataset_name,
            "column": col,
            "strategy": strategy,
            "fill_value": None if pd.isna(fill_value) else str(fill_value),
            "missing_count": missing_count
        })

    return df_imputed, imputation_summary


def generate_descriptive_stats(
    df: pd.DataFrame,
    dataset_name: str
) -> pd.DataFrame:
    """Generate descriptive statistics per column."""
    rows: List[Dict[str, Any]] = []

    scale_bands = {
        "0-1": (0.0, 1.0),
        "1-5": (1.0, 5.0),
        "0-10": (0.0, 10.0),
        "0-100": (0.0, 100.0)
    }

    for col in df.columns:
        series = df[col]
        inferred_type = _infer_series_type(series)
        row: Dict[str, Any] = {
            "dataset": dataset_name,
            "column": col,
            "inferred_type": inferred_type,
            "missing_pct": round(series.isna().sum() / len(df) * 100, 2) if len(df) else 0,
            "unique_count": int(series.nunique(dropna=True))
        }

        if pd.api.types.is_numeric_dtype(series):
            numeric = series.dropna()
            if len(numeric) > 0:
                row.update({
                    "mean": float(numeric.mean()),
                    "median": float(numeric.median()),
                    "mode": float(numeric.mode().iloc[0]) if not numeric.mode().empty else None,
                    "min": float(numeric.min()),
                    "max": float(numeric.max()),
                    "std": float(numeric.std()) if len(numeric) > 1 else 0.0
                })
                z_scores = (numeric - numeric.mean()) / (numeric.std() if numeric.std() else 1)
                row["outlier_count"] = int((abs(z_scores) > 3).sum())

                scale_match = "mixed"
                for label, (low, high) in scale_bands.items():
                    if numeric.min() >= low and numeric.max() <= high:
                        scale_match = label
                        break
                row["scale_consistency"] = scale_match
            else:
                row.update({
                    "mean": None,
                    "median": None,
                    "mode": None,
                    "min": None,
                    "max": None,
                    "std": None,
                    "outlier_count": 0,
                    "scale_consistency": "unknown"
                })
        else:
            mode_vals = series.mode(dropna=True)
            row["mode"] = str(mode_vals.iloc[0]) if not mode_vals.empty else None
            top_values = series.value_counts(dropna=True).head(5).to_dict()
            row["distribution"] = json.dumps({str(k): int(v) for k, v in top_values.items()})

        rows.append(row)

    return pd.DataFrame(rows)


def _sequence_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _distribution_similarity(a: pd.Series, b: pd.Series) -> float:
    if a.dropna().empty or b.dropna().empty:
        return 0.0
    if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
        a_mean, b_mean = a.mean(), b.mean()
        a_std, b_std = a.std(), b.std()
        mean_diff = abs(a_mean - b_mean)
        std_diff = abs(a_std - b_std)
        denom = (abs(a_mean) + abs(b_mean) + 1e-6)
        mean_sim = 1 - min(mean_diff / denom, 1)
        std_sim = 1 - min(std_diff / (abs(a_std) + abs(b_std) + 1e-6), 1)
        return round((mean_sim + std_sim) / 2, 3)

    a_vals = set(a.dropna().astype(str).unique())
    b_vals = set(b.dropna().astype(str).unique())
    if not a_vals or not b_vals:
        return 0.0
    overlap = len(a_vals & b_vals) / len(a_vals | b_vals)
    return round(overlap, 3)


def map_columns_between_datasets(
    df_source: pd.DataFrame,
    df_target: pd.DataFrame,
    source_labels: Optional[Dict[str, str]] = None,
    target_labels: Optional[Dict[str, str]] = None,
    confidence_threshold: float = 0.8
) -> Tuple[List[MappingCandidate], List[Dict[str, Any]]]:
    """Auto-map columns between datasets using name, label, and distribution similarity."""
    source_labels = source_labels or {}
    target_labels = target_labels or {}
    candidates: List[MappingCandidate] = []
    updates: List[Dict[str, Any]] = []

    for source_col in df_source.columns:
        best_candidate: Optional[MappingCandidate] = None

        for target_col in df_target.columns:
            name_sim = _sequence_similarity(source_col, target_col)
            label_sim = _sequence_similarity(
                source_labels.get(source_col, ""),
                target_labels.get(target_col, "")
            )
            dist_sim = _distribution_similarity(df_source[source_col], df_target[target_col])
            confidence = round(0.5 * name_sim + 0.2 * label_sim + 0.3 * dist_sim, 3)

            candidate = MappingCandidate(
                source_column=source_col,
                target_column=target_col,
                confidence=confidence,
                name_similarity=round(name_sim, 3),
                label_similarity=round(label_sim, 3),
                distribution_similarity=round(dist_sim, 3)
            )

            if best_candidate is None or candidate.confidence > best_candidate.confidence:
                best_candidate = candidate

        if best_candidate:
            candidates.append(best_candidate)
            updates.append({
                "source_column": best_candidate.source_column,
                "target_column": best_candidate.target_column,
                "confidence": best_candidate.confidence,
                "status": "auto" if best_candidate.confidence >= confidence_threshold else "review",
                "name_similarity": best_candidate.name_similarity,
                "label_similarity": best_candidate.label_similarity,
                "distribution_similarity": best_candidate.distribution_similarity,
                "updated_at": datetime.utcnow().isoformat()
            })

    return candidates, updates


def update_mapping_table(
    mapping_table: Dict[str, Any],
    table_name: str,
    updates: Iterable[Dict[str, Any]]
) -> Dict[str, Any]:
    """Update mapping table with new mapping entries."""
    mapping_table = mapping_table or {}
    mapping_table.setdefault("column_mappings", {})
    mapping_table["column_mappings"].setdefault(table_name, {})

    for update in updates:
        mapping_table["column_mappings"][table_name][update["source_column"]] = update

    return mapping_table


def calibrate_to_reference(
    source_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    group_columns: List[str]
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Calibrate numeric columns in source_df to match reference_df means/stds."""
    calibrated = source_df.copy()
    report: List[Dict[str, Any]] = []

    numeric_cols = [
        col for col in source_df.columns
        if col in reference_df.columns and pd.api.types.is_numeric_dtype(source_df[col])
    ]

    if not numeric_cols:
        return calibrated, report

    if not group_columns:
        group_columns = []

    if group_columns:
        source_groups = source_df.groupby(group_columns)
        reference_groups = reference_df.groupby(group_columns)
    else:
        source_groups = [(None, source_df)]
        reference_groups = [(None, reference_df)]

    reference_lookup = {
        group: ref_group for group, ref_group in reference_groups
    }

    for group_key, group_df in source_groups:
        ref_group = reference_lookup.get(group_key)
        if ref_group is None or len(ref_group) < 5 or len(group_df) < 5:
            continue

        for col in numeric_cols:
            src_mean = group_df[col].mean()
            src_std = group_df[col].std() if group_df[col].std() else 0
            ref_mean = ref_group[col].mean()
            ref_std = ref_group[col].std() if ref_group[col].std() else 0

            if pd.isna(src_mean) or pd.isna(ref_mean):
                continue

            if src_std == 0 or ref_std == 0:
                adjusted = group_df[col] - src_mean + ref_mean
                method = "mean_shift"
            else:
                adjusted = (group_df[col] - src_mean) / src_std * ref_std + ref_mean
                method = "mean_std"

            calibrated.loc[group_df.index, col] = adjusted

            report.append({
                "group": group_key if group_key is not None else "overall",
                "column": col,
                "source_mean": float(src_mean),
                "reference_mean": float(ref_mean),
                "source_std": float(src_std),
                "reference_std": float(ref_std),
                "method": method
            })

    return calibrated, report


def build_market_silo_comparison(
    historical_df: Optional[pd.DataFrame],
    current_df: Optional[pd.DataFrame],
    market_column_candidates: List[str]
) -> pd.DataFrame:
    """Create market-wise comparison between historical and current datasets."""
    if historical_df is None or current_df is None:
        return pd.DataFrame()

    market_col = next(
        (col for col in market_column_candidates if col in historical_df.columns and col in current_df.columns),
        None
    )
    if not market_col:
        return pd.DataFrame()

    numeric_cols = [
        col for col in historical_df.columns
        if col in current_df.columns and pd.api.types.is_numeric_dtype(historical_df[col])
    ]
    if not numeric_cols:
        return pd.DataFrame()

    hist_summary = historical_df.groupby(market_col)[numeric_cols].mean().reset_index()
    hist_summary["dataset"] = "historical"
    curr_summary = current_df.groupby(market_col)[numeric_cols].mean().reset_index()
    curr_summary["dataset"] = "current"

    return pd.concat([hist_summary, curr_summary], ignore_index=True)
