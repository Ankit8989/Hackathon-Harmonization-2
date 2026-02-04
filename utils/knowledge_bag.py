"""
Self-updating knowledge bag for schema changes and learned mappings.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


def load_knowledge_bag(path: Path) -> Dict[str, Any]:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data
    return {
        "version": "1.0",
        "last_updated": None,
        "datasets": {},
        "schema_diffs": [],
        "learned_mappings": [],
        "level_suggestions": [],
        "imputation_overrides": []
    }


def save_knowledge_bag(path: Path, knowledge_bag: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    knowledge_bag["last_updated"] = datetime.utcnow().isoformat()
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(knowledge_bag, f, sort_keys=False)


def update_imputation_overrides(
    knowledge_bag: Dict[str, Any],
    overrides: Dict[str, str],
    source: str
) -> Dict[str, Any]:
    """Store approved imputation strategies for reuse."""
    knowledge_bag.setdefault("imputation_overrides", [])
    for column, strategy in overrides.items():
        knowledge_bag["imputation_overrides"].append({
            "column": column,
            "strategy": strategy,
            "source": source,
            "timestamp": datetime.utcnow().isoformat()
        })
    return knowledge_bag


def update_knowledge_bag(
    knowledge_bag: Dict[str, Any],
    dataset_name: str,
    profile: Dict[str, Any],
    master_schema: Dict[str, Any],
    mapping_table: Dict[str, Any],
    column_mappings: List[Dict[str, Any]],
    confidence_threshold: float
) -> Dict[str, Any]:
    knowledge_bag.setdefault("datasets", {})
    knowledge_bag["datasets"][dataset_name] = {
        "last_seen": datetime.utcnow().isoformat(),
        "shape": profile.get("shape"),
        "columns": list(profile.get("columns", {}).keys())
    }

    schema_diff = _build_schema_diff(profile, master_schema, dataset_name)
    if schema_diff:
        knowledge_bag.setdefault("schema_diffs", [])
        knowledge_bag["schema_diffs"].append(schema_diff)

    learned = _build_learned_mappings(column_mappings, dataset_name, confidence_threshold)
    if learned:
        knowledge_bag.setdefault("learned_mappings", [])
        knowledge_bag["learned_mappings"].extend(learned)

    level_suggestions = _build_level_suggestions(profile, mapping_table, dataset_name)
    if level_suggestions:
        knowledge_bag.setdefault("level_suggestions", [])
        knowledge_bag["level_suggestions"].extend(level_suggestions)

    return knowledge_bag


def _build_schema_diff(
    profile: Dict[str, Any],
    master_schema: Dict[str, Any],
    dataset_name: str
) -> Dict[str, Any]:
    source_cols = set(profile.get("columns", {}).keys())
    master_cols = {col.get("name") for col in master_schema.get("columns", [])}
    missing = sorted(master_cols - source_cols)
    extra = sorted(source_cols - master_cols)
    return {
        "dataset": dataset_name,
        "timestamp": datetime.utcnow().isoformat(),
        "missing_in_source": missing,
        "extra_in_source": extra
    }


def _build_learned_mappings(
    column_mappings: List[Dict[str, Any]],
    dataset_name: str,
    confidence_threshold: float
) -> List[Dict[str, Any]]:
    learned = []
    for mapping in column_mappings:
        learned.append({
            "dataset": dataset_name,
            "source_column": mapping.get("source_column"),
            "target_column": mapping.get("target_column"),
            "confidence": mapping.get("confidence", 0),
            "status": "auto" if mapping.get("confidence", 0) >= confidence_threshold else "review",
            "timestamp": datetime.utcnow().isoformat()
        })
    return learned


def _build_level_suggestions(
    profile: Dict[str, Any],
    mapping_table: Dict[str, Any],
    dataset_name: str
) -> List[Dict[str, Any]]:
    suggestions: List[Dict[str, Any]] = []
    for col, details in profile.get("columns", {}).items():
        levels = details.get("levels")
        if not levels:
            continue
        known_map = mapping_table.get(col, {})
        for level in levels.keys():
            if str(level) not in known_map and level not in known_map:
                suggestions.append({
                    "dataset": dataset_name,
                    "column": col,
                    "raw_value": level,
                    "suggested_value": str(level).strip(),
                    "status": "review",
                    "timestamp": datetime.utcnow().isoformat()
                })
    return suggestions
