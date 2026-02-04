"""
Multi-source harmonization pipeline to align historical, current, and incremental data.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from agents.data_quality_agent import DataQualityAgent
from agents.harmonization_agent import HarmonizationAgent
from agents.structural_validation_agent import StructuralValidationAgent
from config import METADATA_DIR, OUTPUT_DIR, REPORTS_DIR
from utils.file_handlers import FileHandler, MetadataHandler
from utils.harmonization_helpers import (
    apply_validation_rules,
    build_default_value_mappings,
    calibrate_to_reference,
    generate_descriptive_stats,
    impute_missing_values,
    load_validation_rules,
    map_columns_between_datasets,
    profile_dataframe,
    update_mapping_table,
    build_market_silo_comparison
)
from utils.logger import get_logger
from utils.knowledge_bag import load_knowledge_bag, save_knowledge_bag, update_knowledge_bag


class MultiSourceHarmonizer:
    """End-to-end harmonization across historical, current, and incremental files."""

    def __init__(self) -> None:
        self.file_handler = FileHandler()
        self.metadata_handler = MetadataHandler(METADATA_DIR)
        self.structural_agent = StructuralValidationAgent()
        self.harmonization_agent = HarmonizationAgent()
        self.quality_agent = DataQualityAgent()
        self.logger = get_logger("MultiSourceHarmonizer")

    def run(
        self,
        master_metadata_file: Optional[str] = None,
        historical_file: Optional[str] = None,
        current_file: Optional[str] = None,
        incremental_file: Optional[str] = None,
        validation_rules_file: Optional[str] = None,
        mapping_table_file: Optional[str] = None,
        descriptive_stats_file: Optional[str] = None,
        output_file: Optional[str] = None,
        mapping_confidence_threshold: float = 0.85,
        full_reprocess: bool = False
    ) -> Dict[str, Any]:
        """Run the multi-source harmonization pipeline."""
        master_schema = self._load_master_schema(master_metadata_file)
        mapping_table, mapping_table_path = self._load_mapping_table(mapping_table_file)
        default_mappings = build_default_value_mappings()
        mapping_table.update(default_mappings)
        knowledge_bag_path = METADATA_DIR / "knowledge_bag.yaml"
        knowledge_bag = load_knowledge_bag(knowledge_bag_path)

        validation_rules = load_validation_rules(validation_rules_file)
        baseline_stats = self._load_baseline_stats(descriptive_stats_file)

        datasets: Dict[str, Optional[str]] = {
            "historical": historical_file,
            "current": current_file,
            "incremental": incremental_file
        }

        harmonized_frames: Dict[str, pd.DataFrame] = {}
        profiles: Dict[str, Dict[str, Any]] = {}
        validation_flags: List[Dict[str, Any]] = []
        imputation_summaries: List[Dict[str, Any]] = []

        for name, path in datasets.items():
            if not path:
                continue
            df, metadata = self.file_handler.read_file(path)

            profile = profile_dataframe(df)
            profiles[name] = profile
            self._write_json_report(profile, REPORTS_DIR / f"structural_scan_{name}.json")

            sv_result = self.structural_agent.execute(df, master_schema, metadata)
            column_mappings = sv_result.result.get("column_mappings", []) if sv_result.result else []

            mapping_updates = self._build_source_to_master_updates(
                column_mappings, mapping_confidence_threshold
            )
            mapping_table = update_mapping_table(
                mapping_table, f"{name}_to_master", mapping_updates
            )
            knowledge_bag = update_knowledge_bag(
                knowledge_bag,
                dataset_name=name,
                profile=profile,
                master_schema=master_schema,
                mapping_table=mapping_table,
                column_mappings=column_mappings,
                confidence_threshold=mapping_confidence_threshold
            )

            output_path = OUTPUT_DIR / f"harmonized_{name}.csv"
            self.harmonization_agent.execute(
                df=df,
                column_mappings=column_mappings,
                master_schema=master_schema,
                mapping_tables=mapping_table,
                output_path=output_path
            )
            harmonized_df, _ = self.file_handler.read_file(output_path)

            harmonized_df, impute_summary = impute_missing_values(harmonized_df, name)
            imputation_summaries.extend(impute_summary)

            record_id_col = self._find_record_id_column(harmonized_df)
            validation_flags.extend(
                self._flag_missingness(harmonized_df, name, record_id_col)
            )
            validation_flags.extend(
                apply_validation_rules(harmonized_df, validation_rules, name, record_id_col)
            )

            harmonized_frames[name] = harmonized_df

        historical_df = harmonized_frames.get("historical")
        current_df = harmonized_frames.get("current")

        calibration_report: List[Dict[str, Any]] = []
        if historical_df is not None and current_df is not None:
            group_cols = self._find_calibration_groups(historical_df, current_df)
            historical_df, calibration_report = calibrate_to_reference(
                historical_df, current_df, group_cols
            )
            harmonized_frames["historical"] = historical_df

        cross_relationships: List[Dict[str, Any]] = []
        if historical_df is not None and current_df is not None:
            candidates, updates = map_columns_between_datasets(
                historical_df,
                current_df,
                confidence_threshold=mapping_confidence_threshold
            )
            mapping_table = update_mapping_table(
                mapping_table, "historical_to_current", updates
            )
            cross_relationships = [c.__dict__ for c in candidates]
            self._write_json_report(
                {"cross_dataset_relationships": cross_relationships},
                REPORTS_DIR / "cross_dataset_relationships.json"
            )

        output_path = output_file or str(OUTPUT_DIR / "harmonized_master.csv")
        combined_df = self._combine_outputs(
            harmonized_frames,
            output_path,
            incremental_only=bool(incremental_file and not full_reprocess)
        )

        descriptive_stats = self._build_descriptive_stats(harmonized_frames, baseline_stats)

        market_silo = build_market_silo_comparison(
            historical_df, current_df, ["market", "country", "region"]
        )

        output_file_path = self._write_outputs(
            combined_df,
            mapping_table,
            mapping_table_path,
            calibration_report,
            descriptive_stats,
            validation_flags,
            market_silo,
            output_path
        )
        save_knowledge_bag(knowledge_bag_path, knowledge_bag)

        return {
            "output_file": str(output_file_path),
            "mapping_table_file": str(mapping_table_path),
            "calibration_report": str(REPORTS_DIR / "calibration_report.json"),
            "descriptive_statistics": str(REPORTS_DIR / "descriptive_statistics.csv"),
            "validation_flags": str(REPORTS_DIR / "validation_flags.csv"),
            "market_silo_comparison": str(REPORTS_DIR / "market_silo_comparison.csv"),
            "cross_dataset_relationships": str(REPORTS_DIR / "cross_dataset_relationships.json"),
            "knowledge_bag": str(knowledge_bag_path),
            "structural_scans": [
                str(REPORTS_DIR / f"structural_scan_{name}.json")
                for name in profiles.keys()
            ],
            "imputation_summary": imputation_summaries
        }

    def _load_master_schema(self, path: Optional[str]) -> Dict[str, Any]:
        if path:
            return self._load_yaml_or_json(Path(path))
        return self.metadata_handler.load_master_schema("master_schema")

    def _load_mapping_table(
        self,
        path: Optional[str]
    ) -> Tuple[Dict[str, Any], Path]:
        if path:
            table = self._load_yaml_or_json(Path(path))
            return table or {}, Path(path)
        default_path = METADATA_DIR / "mapping_tables.yaml"
        if default_path.exists():
            return self.metadata_handler.load_mapping_tables("mapping_tables"), default_path
        return {}, default_path

    def _load_yaml_or_json(self, path: Path) -> Dict[str, Any]:
        if path.suffix.lower() in {".yaml", ".yml"}:
            return self.file_handler.read_yaml(path)
        if path.suffix.lower() == ".json":
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        raise ValueError(f"Unsupported metadata file: {path}")

    def _load_baseline_stats(self, path: Optional[str]) -> Optional[pd.DataFrame]:
        if not path:
            return None
        df, _ = self.file_handler.read_file(path)
        return df

    def _build_source_to_master_updates(
        self,
        column_mappings: List[Dict[str, Any]],
        threshold: float
    ) -> List[Dict[str, Any]]:
        updates: List[Dict[str, Any]] = []
        for mapping in column_mappings:
            confidence = mapping.get("confidence", 0)
            updates.append({
                "source_column": mapping.get("source_column"),
                "target_column": mapping.get("target_column"),
                "confidence": confidence,
                "status": "auto" if confidence >= threshold else "review",
                "updated_at": datetime.utcnow().isoformat()
            })
        return updates

    def _find_record_id_column(self, df: pd.DataFrame) -> Optional[str]:
        candidates = ["record_id", "respondent_id", "id", "case_id"]
        for col in candidates:
            if col in df.columns:
                return col
        return None

    def _flag_missingness(
        self,
        df: pd.DataFrame,
        dataset_name: str,
        record_id_column: Optional[str]
    ) -> List[Dict[str, Any]]:
        flags: List[Dict[str, Any]] = []
        record_id_series = df[record_id_column] if record_id_column and record_id_column in df.columns else None
        for col in df.columns:
            missing_pct = df[col].isna().mean() * 100
            if missing_pct == 0:
                continue
            issue_type = "structural_missing" if missing_pct == 100 else "random_missing"
            flags.append({
                "dataset": dataset_name,
                "record_id": None,
                "column": col,
                "issue_type": issue_type,
                "details": f"{missing_pct:.1f}% missing"
            })
        if record_id_series is not None:
            # Record-level missing flags for required tracking
            row_missing = df.isna().sum(axis=1)
            for idx in row_missing[row_missing > 0].index[:500]:
                flags.append({
                    "dataset": dataset_name,
                    "record_id": record_id_series.loc[idx],
                    "column": "multiple",
                    "issue_type": "row_missing",
                    "details": f"{int(row_missing.loc[idx])} fields missing"
                })
        return flags

    def _find_calibration_groups(
        self,
        historical_df: pd.DataFrame,
        current_df: pd.DataFrame
    ) -> List[str]:
        candidates = ["market", "country", "region", "vendor", "source_vendor", "supplier", "wave", "date"]
        return [c for c in candidates if c in historical_df.columns and c in current_df.columns]

    def _combine_outputs(
        self,
        frames: Dict[str, pd.DataFrame],
        output_file: str,
        incremental_only: bool
    ) -> pd.DataFrame:
        if incremental_only and output_file and Path(output_file).exists():
            existing, _ = self.file_handler.read_file(output_file)
            incremental = frames.get("incremental")
            if incremental is None:
                return existing
            combined = pd.concat([existing, incremental], ignore_index=True, sort=False)
            return combined

        dfs = [df for df in frames.values() if df is not None]
        if not dfs:
            return pd.DataFrame()
        return pd.concat(dfs, ignore_index=True, sort=False)

    def _build_descriptive_stats(
        self,
        frames: Dict[str, pd.DataFrame],
        baseline_stats: Optional[pd.DataFrame]
    ) -> pd.DataFrame:
        stats_frames = []
        for name, df in frames.items():
            stats_frames.append(generate_descriptive_stats(df, name))
        combined = pd.concat(stats_frames, ignore_index=True) if stats_frames else pd.DataFrame()
        if baseline_stats is not None and not baseline_stats.empty:
            baseline_stats = baseline_stats.copy()
            baseline_stats["dataset"] = "baseline"
            combined = pd.concat([combined, baseline_stats], ignore_index=True)
        return combined.head(1000)

    def _write_outputs(
        self,
        combined_df: pd.DataFrame,
        mapping_table: Dict[str, Any],
        mapping_table_path: Path,
        calibration_report: List[Dict[str, Any]],
        descriptive_stats: pd.DataFrame,
        validation_flags: List[Dict[str, Any]],
        market_silo: pd.DataFrame,
        output_path: str
    ) -> Path:
        output_path_obj = Path(output_path)
        self.file_handler.write_file(combined_df, output_path_obj)

        self.file_handler.write_yaml(mapping_table, mapping_table_path)
        self._write_json_report(
            {"calibration": calibration_report}, REPORTS_DIR / "calibration_report.json"
        )
        self.file_handler.write_file(descriptive_stats, REPORTS_DIR / "descriptive_statistics.csv")
        self.file_handler.write_file(pd.DataFrame(validation_flags), REPORTS_DIR / "validation_flags.csv")
        if not market_silo.empty:
            self.file_handler.write_file(market_silo, REPORTS_DIR / "market_silo_comparison.csv")

        return output_path_obj

    def _write_json_report(self, data: Dict[str, Any], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
