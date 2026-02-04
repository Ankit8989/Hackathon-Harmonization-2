"""
Generate supporting metadata files from a raw (unclean) dataset.
"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

from utils.harmonization_helpers import build_default_value_mappings, generate_descriptive_stats


def _infer_type(series: pd.Series) -> str:
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
        if len(non_null) > 0 and non_null.nunique() / len(non_null) < 0.1:
            return "categorical"
        return "string"
    return "string"


def generate_supporting_files(
    df: pd.DataFrame,
    output_dir: Path,
    metadata_dir: Path,
    prefix: str = "auto",
    max_levels: int = 25
) -> Dict[str, str]:
    """Generate master metadata, mapping table, data dictionary, and validation rules."""
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    # Save raw unclean input
    raw_path = output_dir / f"{prefix}_unclean_{timestamp}.csv"
    df.to_csv(raw_path, index=False)

    columns = []
    data_dictionary = []
    standards = {
        "version": "1.0.0",
        "generated_at": datetime.utcnow().isoformat(),
        "columns": {}
    }
    validation_rules = {"rules": [{"min_records": 10}]}
    mapping_table: Dict[str, Dict[str, str]] = build_default_value_mappings()

    for col in df.columns:
        series = df[col]
        inferred = _infer_type(series)
        sample_values = series.dropna().astype(str).head(5).tolist()

        column_entry = {
            "name": col,
            "canonical_name": col,
            "data_type": inferred,
            "description": "",
            "aliases": []
        }

        if pd.api.types.is_numeric_dtype(series):
            col_min = series.min()
            col_max = series.max()
            if pd.notna(col_min) and pd.notna(col_max):
                column_entry["validation_rules"] = {
                    "min_value": float(col_min),
                    "max_value": float(col_max)
                }
                validation_rules["rules"].append({
                    "column": col,
                    "min": float(col_min),
                    "max": float(col_max)
                })
                standards["columns"][col] = {
                    "data_type": inferred,
                    "min": float(col_min),
                    "max": float(col_max),
                    "impute_strategy": "mean"
                }
        else:
            uniques = series.dropna().astype(str).unique().tolist()
            if 0 < len(uniques) <= max_levels:
                validation_rules["rules"].append({
                    "column": col,
                    "allowed_values": uniques
                })
                mapping_table.setdefault(col, {})
                for val in uniques:
                    mapping_table[col][val] = val
                standards["columns"][col] = {
                    "data_type": inferred,
                    "allowed_values": uniques,
                    "impute_strategy": "mode"
                }
            else:
                standards["columns"][col] = {
                    "data_type": inferred,
                    "impute_strategy": "mode"
                }

        columns.append(column_entry)
        data_dictionary.append({
            "column": col,
            "data_type": inferred,
            "sample_values": sample_values,
            "description": ""
        })

    master_metadata = {
        "schema_name": f"{prefix}_master_metadata",
        "version": "1.0.0",
        "description": "Auto-generated master metadata from unclean input",
        "created_at": datetime.utcnow().isoformat(),
        "columns": columns
    }

    master_path = metadata_dir / f"{prefix}_master_metadata.yaml"
    mapping_path = metadata_dir / f"{prefix}_mapping_table.yaml"
    rules_path = metadata_dir / f"{prefix}_validation_rules.yaml"
    dictionary_path = metadata_dir / f"{prefix}_data_dictionary.yaml"
    standards_path = metadata_dir / f"{prefix}_master_dictionary.json"

    with open(master_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(master_metadata, f, sort_keys=False)

    with open(mapping_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(mapping_table, f, sort_keys=False)

    with open(rules_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(validation_rules, f, sort_keys=False)

    with open(dictionary_path, "w", encoding="utf-8") as f:
        yaml.safe_dump({"fields": data_dictionary}, f, sort_keys=False)

    with open(standards_path, "w", encoding="utf-8") as f:
        json.dump(standards, f, indent=2)

    stats = generate_descriptive_stats(df, "unclean_input")
    stats_path = output_dir / f"{prefix}_descriptive_stats_{timestamp}.csv"
    stats.to_csv(stats_path, index=False)

    return {
        "raw_input": str(raw_path),
        "master_metadata": str(master_path),
        "mapping_table": str(mapping_path),
        "validation_rules": str(rules_path),
        "data_dictionary": str(dictionary_path),
        "master_dictionary": str(standards_path),
        "descriptive_stats": str(stats_path)
    }
