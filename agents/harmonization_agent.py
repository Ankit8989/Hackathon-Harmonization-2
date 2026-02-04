"""
Harmonization Agent for the Agentic AI Data Harmonization System.
Transforms data to conform to canonical schema.
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable

import pandas as pd
import numpy as np

from agents.base_agent import BaseAgent
from agents.llm_reasoning_agent import get_llm_reasoning_agent
from config import HARMONIZATION_CONFIG, HARMONIZATION_SETTINGS, OUTPUT_DIR
from models.schemas import (
    AgentResponse,
    ColumnMapping,
    ColumnTransformationSummary,
    HarmonizationResult,
    ProcessingStatus,
    TransformationRecord
)
from utils.file_handlers import FileHandler, compare_dataframes


class HarmonizationAgent(BaseAgent):
    """
    Agent responsible for data harmonization and transformation.
    
    Capabilities:
    - Apply column mappings
    - Normalize scales (e.g., 1-10 to 0-100)
    - Standardize categorical values
    - Align date/time formats
    - Standardize hierarchies (region, brand, SKU)
    - Generate before/after comparison
    """
    
    PROMPT_TEMPLATE = """You are an expert data engineer specializing in data transformation and harmonization.

TASK: Generate transformation logic for harmonizing source data to canonical schema.

SOURCE COLUMN: {source_column}
SOURCE DATA TYPE: {source_type}
SOURCE SAMPLE VALUES: {source_samples}
SOURCE VALUE DISTRIBUTION: {value_distribution}

TARGET COLUMN: {target_column}
TARGET DATA TYPE: {target_type}
TARGET CONSTRAINTS: {target_constraints}

MAPPING TABLES AVAILABLE: {mapping_tables}

Generate transformation logic that:
1. Converts data types appropriately
2. Maps values using provided mapping tables
3. Handles edge cases and nulls
4. Preserves data integrity

RESPOND IN STRICT JSON FORMAT:
{{
    "transformation_type": "scale_normalize|value_map|type_cast|date_format|string_standardize|custom",
    "transformation_description": "human readable description",
    "pandas_code": "df['target'] = df['source'].apply(lambda x: ...)",
    "value_mappings": {{
        "source_value": "target_value"
    }},
    "null_handling": "keep|fill_default|drop",
    "default_value": null,
    "scale_params": {{
        "source_min": null,
        "source_max": null,
        "target_min": 0,
        "target_max": 100
    }},
    "date_format": {{
        "source_format": null,
        "target_format": "%Y-%m-%d"
    }},
    "string_operations": ["lowercase", "strip", "replace_special"],
    "warnings": ["list of potential issues"],
    "confidence": 0.95
}}"""

    def __init__(self):
        """Initialize the Harmonization Agent"""
        super().__init__(
            name=HARMONIZATION_CONFIG.name,
            confidence_threshold=HARMONIZATION_CONFIG.confidence_threshold,
            max_retries=HARMONIZATION_CONFIG.max_retries
        )
        self.llm_agent = get_llm_reasoning_agent()
        self.file_handler = FileHandler()
        self.settings = HARMONIZATION_SETTINGS
        
        # Transformation registry
        self.transformation_registry: Dict[str, Callable] = {
            "scale_normalize": self._apply_scale_normalization,
            "value_map": self._apply_value_mapping,
            "type_cast": self._apply_type_cast,
            "date_format": self._apply_date_format,
            "string_standardize": self._apply_string_standardization,
            "custom": self._apply_custom_transformation
        }
    
    def get_prompt_template(self) -> str:
        """Get the agent's prompt template"""
        return self.PROMPT_TEMPLATE
    
    def execute(
        self,
        df: pd.DataFrame,
        column_mappings: List[Dict[str, Any]],
        master_schema: Dict[str, Any],
        mapping_tables: Optional[Dict[str, Any]] = None,
        output_path: Optional[Path] = None
    ) -> AgentResponse:
        """
        Execute data harmonization.
        
        Args:
            df: Input DataFrame to harmonize
            column_mappings: Column mappings from structural validation
            master_schema: Master schema definition
            mapping_tables: Optional value mapping tables
            output_path: Optional output file path
            
        Returns:
            AgentResponse containing HarmonizationResult
        """
        self.start_execution()
        
        try:
            # Store original for comparison
            df_original = df.copy()
            
            # Step 1: Initialize harmonized DataFrame
            self.add_audit_entry(
                action="Initializing harmonization",
                status=ProcessingStatus.IN_PROGRESS
            )
            df_harmonized = pd.DataFrame()
            
            # Step 2: Process column mappings
            self.add_audit_entry(
                action="Processing column mappings",
                status=ProcessingStatus.IN_PROGRESS
            )
            transformation_summaries = []
            columns_renamed = {}
            llm_decisions = []
            issues = []
            warnings = []
            
            # Convert mapping dicts to proper format if needed
            mappings = self._normalize_mappings(column_mappings)
            
            for mapping in mappings:
                source_col = mapping.get("source_column")
                target_col = mapping.get("target_column")
                
                if target_col == "UNMAPPED" or not target_col:
                    warnings.append(f"Skipping unmapped column: {source_col}")
                    continue
                
                if source_col not in df.columns:
                    issues.append(f"Source column not found: {source_col}")
                    continue
                
                try:
                    # Get transformation logic
                    transform_result = self._determine_transformation(
                        df, source_col, target_col, master_schema, mapping_tables
                    )
                    
                    if transform_result.get("llm_used"):
                        llm_decisions.append({
                            "source": source_col,
                            "target": target_col,
                            "decision": transform_result.get("transformation_type"),
                            "confidence": transform_result.get("confidence", 0)
                        })
                    
                    # Apply transformation
                    transformed_series, summary = self._apply_transformation(
                        df[source_col],
                        source_col,
                        target_col,
                        transform_result
                    )
                    
                    df_harmonized[target_col] = transformed_series
                    transformation_summaries.append(summary)
                    
                    if source_col != target_col:
                        columns_renamed[source_col] = target_col
                    
                except Exception as e:
                    self.logger.error(f"Error transforming {source_col}: {str(e)}")
                    issues.append(f"Transformation failed for {source_col}: {str(e)}")
                    # Copy original column as fallback
                    df_harmonized[target_col] = df[source_col]
            
            # Step 3: Add any required columns not in source
            self.add_audit_entry(
                action="Adding missing required columns",
                status=ProcessingStatus.IN_PROGRESS
            )
            columns_added = self._add_missing_columns(
                df_harmonized, master_schema, mappings
            )
            
            # Step 4: Reorder columns to match schema
            df_harmonized = self._reorder_columns(df_harmonized, master_schema)
            
            # Step 5: Generate comparison
            self.add_audit_entry(
                action="Generating before/after comparison",
                status=ProcessingStatus.IN_PROGRESS
            )
            comparison = compare_dataframes(df_original, df_harmonized)
            
            # Step 6: Save output
            output_file_path = None
            if output_path:
                output_file_path = str(output_path)
            else:
                output_file_path = str(OUTPUT_DIR / "harmonized.csv")
            
            self.file_handler.write_file(df_harmonized, output_file_path)
            
            # Calculate confidence
            successful_transforms = sum(1 for s in transformation_summaries if s.records_failed == 0)
            confidence = successful_transforms / len(transformation_summaries) if transformation_summaries else 0
            
            # Build result
            harmonization_result = HarmonizationResult(
                success=len(issues) == 0,
                confidence_score=confidence,
                input_records=len(df_original),
                output_records=len(df_harmonized),
                records_modified=comparison.get("total_cells_changed", 0),
                records_dropped=len(df_original) - len(df_harmonized),
                column_transformations=transformation_summaries,
                applied_mappings=[ColumnMapping(**m) if isinstance(m, dict) else m for m in mappings],
                columns_added=columns_added,
                columns_removed=comparison.get("columns_removed", []),
                columns_renamed=columns_renamed,
                harmonization_issues=issues,
                warnings=warnings,
                llm_decisions=llm_decisions,
                output_file_path=output_file_path,
                processing_time_seconds=(datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
                timestamp=datetime.now()
            )
            
            success = harmonization_result.success and confidence >= self.confidence_threshold
            self.end_execution(success)
            
            return self.create_response(
                success=success,
                confidence_score=confidence,
                result=harmonization_result.model_dump(),
                errors=issues,
                warnings=warnings
            )
            
        except Exception as e:
            self.logger.error(f"Harmonization failed: {str(e)}")
            self.end_execution(False)
            
            return self.create_response(
                success=False,
                confidence_score=0.0,
                result=None,
                errors=[str(e)]
            )
    
    def _normalize_mappings(
        self,
        mappings: List[Any]
    ) -> List[Dict[str, Any]]:
        """Convert mappings to consistent dictionary format"""
        normalized = []
        for m in mappings:
            if isinstance(m, dict):
                normalized.append(m)
            elif hasattr(m, 'model_dump'):
                normalized.append(m.model_dump())
            elif hasattr(m, '__dict__'):
                normalized.append(m.__dict__)
            else:
                normalized.append({"source_column": str(m), "target_column": str(m)})
        return normalized
    
    def _determine_transformation(
        self,
        df: pd.DataFrame,
        source_col: str,
        target_col: str,
        master_schema: Dict[str, Any],
        mapping_tables: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Determine the transformation needed for a column.
        
        Args:
            df: Source DataFrame
            source_col: Source column name
            target_col: Target column name
            master_schema: Master schema definition
            mapping_tables: Value mapping tables
            
        Returns:
            Transformation specification dictionary
        """
        source_series = df[source_col]
        source_type = str(source_series.dtype)
        
        # Get target column spec from schema
        target_spec = self._get_column_spec(master_schema, target_col)
        target_type = target_spec.get("data_type", "string")
        target_constraints = target_spec.get("validation_rules", {})
        
        # Check if we have a mapping table for this column
        if mapping_tables and target_col in mapping_tables:
            return {
                "transformation_type": "value_map",
                "value_mappings": mapping_tables[target_col],
                "null_handling": "keep",
                "confidence": 0.95,
                "llm_used": False
            }
        
        # Determine transformation type based on types
        if self._needs_scale_normalization(source_series, target_constraints):
            return self._get_scale_params(source_series, target_constraints)
        
        if self._is_date_conversion_needed(source_type, target_type):
            return {
                "transformation_type": "date_format",
                "date_format": {
                    "source_format": self._detect_date_format(source_series),
                    "target_format": self.settings.target_date_format
                },
                "null_handling": "keep",
                "confidence": 0.90,
                "llm_used": False
            }
        
        if self._needs_type_cast(source_type, target_type):
            return {
                "transformation_type": "type_cast",
                "target_type": target_type,
                "null_handling": "keep",
                "confidence": 0.85,
                "llm_used": False
            }
        
        # Use LLM for complex transformations
        return self._llm_determine_transformation(
            df, source_col, target_col, target_spec, mapping_tables
        )
    
    def _get_column_spec(
        self,
        master_schema: Dict[str, Any],
        column_name: str
    ) -> Dict[str, Any]:
        """Get column specification from master schema"""
        columns = master_schema.get("columns", [])
        if not columns and "schema" in master_schema:
            columns = master_schema["schema"].get("columns", [])
        
        for col in columns:
            if col.get("name") == column_name or col.get("canonical_name") == column_name:
                return col
        
        return {}
    
    def _needs_scale_normalization(
        self,
        series: pd.Series,
        constraints: Dict[str, Any]
    ) -> bool:
        """Check if scale normalization is needed"""
        if not pd.api.types.is_numeric_dtype(series):
            return False
        
        target_min = constraints.get("min_value")
        target_max = constraints.get("max_value")
        
        if target_min is None or target_max is None:
            return False
        
        source_min = series.min()
        source_max = series.max()
        
        # Check if ranges differ significantly
        return (source_min != target_min or source_max != target_max) and \
               (source_max - source_min) > 0
    
    def _get_scale_params(
        self,
        series: pd.Series,
        constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get scale normalization parameters"""
        scale_method = "linear"
        if len(series.dropna()) >= 5:
            p01 = series.quantile(0.01)
            p99 = series.quantile(0.99)
            if series.max() != series.min() and (p99 - p01) / (series.max() - series.min()) < 0.7:
                scale_method = "percentile"

        mixed_scale = self._detect_mixed_scale(series)
        return {
            "transformation_type": "scale_normalize",
            "scale_params": {
                "source_min": float(series.min()),
                "source_max": float(series.max()),
                "target_min": constraints.get("min_value", 0),
                "target_max": constraints.get("max_value", 100)
            },
            "scale_method": scale_method,
            "mixed_scale": mixed_scale,
            "null_handling": "keep",
            "confidence": 0.95,
            "llm_used": False
        }
    
    def _is_date_conversion_needed(
        self,
        source_type: str,
        target_type: str
    ) -> bool:
        """Check if date format conversion is needed"""
        date_types = ["datetime", "date", "datetime64"]
        return any(dt in target_type.lower() for dt in date_types) or \
               any(dt in source_type.lower() for dt in date_types)
    
    def _detect_date_format(self, series: pd.Series) -> Optional[str]:
        """Detect the date format in a series"""
        sample = series.dropna().head(10)
        common_formats = [
            "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y",
            "%Y-%m-%d %H:%M:%S", "%d-%m-%Y", "%Y%m%d"
        ]
        
        for fmt in common_formats:
            try:
                pd.to_datetime(sample, format=fmt)
                return fmt
            except (ValueError, TypeError):
                continue
        
        return None
    
    def _needs_type_cast(self, source_type: str, target_type: str) -> bool:
        """Check if type casting is needed"""
        source_type = source_type.lower()
        target_type = target_type.lower()
        
        if source_type == target_type:
            return False
        
        # Map pandas types to general types
        type_map = {
            "int64": "integer", "int32": "integer", "int": "integer",
            "float64": "float", "float32": "float", "float": "float",
            "object": "string", "str": "string", "string": "string",
            "bool": "boolean", "boolean": "boolean"
        }
        
        source_general = type_map.get(source_type, source_type)
        target_general = type_map.get(target_type, target_type)
        
        return source_general != target_general
    
    def _llm_determine_transformation(
        self,
        df: pd.DataFrame,
        source_col: str,
        target_col: str,
        target_spec: Dict[str, Any],
        mapping_tables: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Determine transformation logic WITHOUT LLM for common cases.
        Uses hardcoded rules to save API tokens.
        """
        source_series = df[source_col]
        source_dtype = str(source_series.dtype)
        target_type = target_spec.get("data_type", "string")
        
        # OPTIMIZATION: Use rule-based logic instead of LLM
        # This saves ~5000 tokens per column!
        
        # Determine transformation type based on source/target types
        transformation_type = "direct_copy"
        null_handling = "keep"
        
        # Type conversion rules (hardcoded)
        if "int" in source_dtype and target_type == "float":
            transformation_type = "type_convert"
        elif "float" in source_dtype and target_type == "integer":
            transformation_type = "type_convert"
        elif "object" in source_dtype and target_type in ["integer", "float"]:
            transformation_type = "type_convert"
        elif "datetime" in source_dtype.lower() or "date" in target_type.lower():
            transformation_type = "date_format"
        elif target_type == "categorical":
            transformation_type = "string_standardize"
        elif "object" in source_dtype:
            transformation_type = "string_standardize"
        
        # Check for value mapping
        if mapping_tables and target_col in str(mapping_tables):
            transformation_type = "value_mapping"
        
        # Handle nulls based on target requirements
        if target_spec.get("is_required", False):
            null_handling = "flag"  # Flag but don't remove
        
        result = {
            "transformation_type": transformation_type,
            "null_handling": null_handling,
            "confidence": 0.85,
            "llm_used": False,  # No LLM used!
            "reasoning": f"Rule-based: {source_dtype} -> {target_type}"
        }
        
        self.logger.info(f"Transformation for {source_col}: {transformation_type} (NO LLM)")
        
        return result
    
    def _apply_transformation(
        self,
        series: pd.Series,
        source_col: str,
        target_col: str,
        transform_spec: Dict[str, Any]
    ) -> Tuple[pd.Series, ColumnTransformationSummary]:
        """
        Apply transformation to a series.
        
        Args:
            series: Input series
            source_col: Source column name
            target_col: Target column name
            transform_spec: Transformation specification
            
        Returns:
            Tuple of (transformed series, summary)
        """
        transform_type = transform_spec.get("transformation_type", "string_standardize")
        
        # Get transformer function
        transformer = self.transformation_registry.get(
            transform_type,
            self._apply_string_standardization
        )
        
        try:
            transformed, records = transformer(series, transform_spec)
            failed = 0
        except Exception as e:
            self.logger.warning(f"Transformation failed for {source_col}: {str(e)}")
            transformed = series.copy()
            failed = len(series)
            records = []
        
        summary = ColumnTransformationSummary(
            source_column=source_col,
            target_column=target_col,
            transformation_type=transform_type,
            records_transformed=len(series) - failed,
            records_failed=failed,
            sample_transformations=records[:5]
        )
        
        return transformed, summary
    
    def _apply_scale_normalization(
        self,
        series: pd.Series,
        spec: Dict[str, Any]
    ) -> Tuple[pd.Series, List[TransformationRecord]]:
        """Apply scale normalization"""
        params = spec.get("scale_params", {})
        source_min = params.get("source_min", series.min())
        source_max = params.get("source_max", series.max())
        target_min = params.get("target_min", 0)
        target_max = params.get("target_max", 100)
        scale_method = spec.get("scale_method", "linear")
        mixed_scale = spec.get("mixed_scale")
        
        # Normalize
        if mixed_scale:
            transformed = self._apply_mixed_scale(series, mixed_scale, target_min, target_max)
        elif scale_method == "percentile":
            ranks = series.rank(pct=True)
            transformed = ranks * (target_max - target_min) + target_min
        elif source_max - source_min != 0:
            normalized = (series - source_min) / (source_max - source_min)
            transformed = normalized * (target_max - target_min) + target_min
        else:
            transformed = series
        
        # Sample records
        records = []
        for i in range(min(5, len(series))):
            if pd.notna(series.iloc[i]):
                records.append(TransformationRecord(
                    column_name=series.name or "unknown",
                    transformation_type="scale_normalize",
                    original_value=float(series.iloc[i]),
                    transformed_value=float(transformed.iloc[i]),
                    transformation_rule=f"({source_min}-{source_max}) -> ({target_min}-{target_max})"
                ))
        
        return transformed, records

    def _detect_mixed_scale(self, series: pd.Series) -> Optional[Dict[str, Tuple[float, float]]]:
        """Detect mixed scales within a numeric series."""
        values = series.dropna()
        if values.empty:
            return None

        scale_bands = {
            "0_1": (0.0, 1.05),
            "1_5": (1.0, 5.05),
            "0_10": (0.0, 10.05),
            "0_100": (0.0, 100.05)
        }

        counts = {}
        total = len(values)
        for label, (low, high) in scale_bands.items():
            counts[label] = int(((values >= low) & (values <= high)).sum())

        significant = [label for label, cnt in counts.items() if cnt / total >= 0.15]
        if len(significant) <= 1:
            return None

        return {label: scale_bands[label] for label in significant}

    def _apply_mixed_scale(
        self,
        series: pd.Series,
        mixed_scale: Dict[str, Tuple[float, float]],
        target_min: float,
        target_max: float
    ) -> pd.Series:
        """Apply piecewise scaling when mixed scales are detected."""
        def scale_value(x):
            if pd.isna(x):
                return x
            for _, (low, high) in mixed_scale.items():
                if low <= x <= high and high - low != 0:
                    return ((x - low) / (high - low)) * (target_max - target_min) + target_min
            return x

        return series.apply(scale_value)
    
    def _apply_value_mapping(
        self,
        series: pd.Series,
        spec: Dict[str, Any]
    ) -> Tuple[pd.Series, List[TransformationRecord]]:
        """Apply value mapping"""
        mappings = spec.get("value_mappings", {})
        default = spec.get("default_value")
        
        def map_value(x):
            if pd.isna(x):
                return x
            str_x = str(x)
            return mappings.get(str_x, mappings.get(x, default if default else x))
        
        transformed = series.apply(map_value)
        
        # Sample records
        records = []
        for i in range(min(5, len(series))):
            if pd.notna(series.iloc[i]):
                records.append(TransformationRecord(
                    column_name=series.name or "unknown",
                    transformation_type="value_map",
                    original_value=series.iloc[i],
                    transformed_value=transformed.iloc[i],
                    transformation_rule=f"Map: {len(mappings)} value mappings"
                ))
        
        return transformed, records
    
    def _apply_type_cast(
        self,
        series: pd.Series,
        spec: Dict[str, Any]
    ) -> Tuple[pd.Series, List[TransformationRecord]]:
        """Apply type casting"""
        target_type = spec.get("target_type", "string")
        
        type_map = {
            "integer": "Int64",  # Nullable integer
            "float": "float64",
            "string": "string",
            "boolean": "boolean"
        }
        
        pandas_type = type_map.get(target_type, "object")
        
        try:
            if target_type == "integer":
                transformed = pd.to_numeric(series, errors='coerce').astype("Int64")
            elif target_type == "float":
                transformed = pd.to_numeric(series, errors='coerce')
            elif target_type == "boolean":
                transformed = series.astype(bool)
            else:
                transformed = series.astype(str)
        except Exception:
            transformed = series
        
        records = []
        for i in range(min(5, len(series))):
            records.append(TransformationRecord(
                column_name=series.name or "unknown",
                transformation_type="type_cast",
                original_value=series.iloc[i],
                transformed_value=transformed.iloc[i],
                transformation_rule=f"Cast to {target_type}"
            ))
        
        return transformed, records
    
    def _apply_date_format(
        self,
        series: pd.Series,
        spec: Dict[str, Any]
    ) -> Tuple[pd.Series, List[TransformationRecord]]:
        """Apply date format transformation"""
        date_spec = spec.get("date_format", {})
        source_format = date_spec.get("source_format")
        target_format = date_spec.get("target_format", "%Y-%m-%d")
        
        try:
            if source_format:
                parsed = pd.to_datetime(series, format=source_format, errors='coerce')
            else:
                parsed = pd.to_datetime(series, errors='coerce')
            
            transformed = parsed.dt.strftime(target_format)
            transformed = transformed.where(parsed.notna(), None)
        except Exception:
            transformed = series
        
        records = []
        for i in range(min(5, len(series))):
            if pd.notna(series.iloc[i]):
                records.append(TransformationRecord(
                    column_name=series.name or "unknown",
                    transformation_type="date_format",
                    original_value=str(series.iloc[i]),
                    transformed_value=str(transformed.iloc[i]) if pd.notna(transformed.iloc[i]) else None,
                    transformation_rule=f"Format: {source_format or 'auto'} -> {target_format}"
                ))
        
        return transformed, records
    
    def _apply_string_standardization(
        self,
        series: pd.Series,
        spec: Dict[str, Any]
    ) -> Tuple[pd.Series, List[TransformationRecord]]:
        """Apply string standardization"""
        operations = spec.get("string_operations", ["strip"])
        
        transformed = series.copy()
        
        if pd.api.types.is_object_dtype(transformed) or pd.api.types.is_string_dtype(transformed):
            transformed = transformed.astype(str)
            
            for op in operations:
                if op == "lowercase":
                    transformed = transformed.str.lower()
                elif op == "uppercase":
                    transformed = transformed.str.upper()
                elif op == "strip":
                    transformed = transformed.str.strip()
                elif op == "replace_special":
                    transformed = transformed.str.replace(r'[^\w\s]', '', regex=True)
            
            # Handle nulls that became 'nan' string
            transformed = transformed.replace('nan', None)
        
        records = []
        for i in range(min(5, len(series))):
            if pd.notna(series.iloc[i]):
                records.append(TransformationRecord(
                    column_name=series.name or "unknown",
                    transformation_type="string_standardize",
                    original_value=str(series.iloc[i]),
                    transformed_value=str(transformed.iloc[i]) if pd.notna(transformed.iloc[i]) else None,
                    transformation_rule=f"Operations: {operations}"
                ))
        
        return transformed, records
    
    def _apply_custom_transformation(
        self,
        series: pd.Series,
        spec: Dict[str, Any]
    ) -> Tuple[pd.Series, List[TransformationRecord]]:
        """Apply custom transformation using LLM-generated code"""
        pandas_code = spec.get("pandas_code", "")
        
        if pandas_code:
            try:
                # Create a safe execution environment
                local_vars = {"df_col": series, "pd": pd, "np": np}
                exec(f"result = {pandas_code.replace('df[source]', 'df_col')}", {}, local_vars)
                transformed = local_vars.get("result", series)
            except Exception as e:
                self.logger.warning(f"Custom transformation failed: {str(e)}")
                transformed = series
        else:
            transformed = series
        
        records = []
        for i in range(min(5, len(series))):
            records.append(TransformationRecord(
                column_name=series.name or "unknown",
                transformation_type="custom",
                original_value=series.iloc[i],
                transformed_value=transformed.iloc[i] if i < len(transformed) else None,
                transformation_rule=pandas_code[:50] + "..." if len(pandas_code) > 50 else pandas_code
            ))
        
        return transformed, records
    
    def _add_missing_columns(
        self,
        df: pd.DataFrame,
        master_schema: Dict[str, Any],
        mappings: List[Dict[str, Any]]
    ) -> List[str]:
        """Add missing required columns with default values"""
        columns = master_schema.get("columns", [])
        mapped_targets = {m.get("target_column") for m in mappings}
        added = []
        
        for col_spec in columns:
            col_name = col_spec.get("name") or col_spec.get("canonical_name")
            if col_name and col_name not in df.columns and col_name not in mapped_targets:
                if col_spec.get("is_required", False):
                    default = col_spec.get("default_value")
                    df[col_name] = default
                    added.append(col_name)
        
        return added
    
    def _reorder_columns(
        self,
        df: pd.DataFrame,
        master_schema: Dict[str, Any]
    ) -> pd.DataFrame:
        """Reorder columns to match master schema order"""
        columns = master_schema.get("columns", [])
        schema_order = [
            col.get("name") or col.get("canonical_name")
            for col in columns
        ]
        
        # Get columns in schema order, then add any extras at the end
        ordered = []
        for col in schema_order:
            if col in df.columns:
                ordered.append(col)
        
        # Add remaining columns
        for col in df.columns:
            if col not in ordered:
                ordered.append(col)
        
        return df[ordered]
    
    def harmonize_single_column(
        self,
        series: pd.Series,
        source_name: str,
        target_name: str,
        target_type: str,
        mapping_table: Optional[Dict[str, Any]] = None
    ) -> pd.Series:
        """
        Harmonize a single column.
        
        Args:
            series: Input series
            source_name: Source column name
            target_name: Target column name
            target_type: Target data type
            mapping_table: Optional value mappings
            
        Returns:
            Transformed series
        """
        spec = {
            "transformation_type": "value_map" if mapping_table else "string_standardize",
            "value_mappings": mapping_table or {},
            "target_type": target_type,
            "null_handling": "keep"
        }
        
        if mapping_table:
            transformed, _ = self._apply_value_mapping(series, spec)
        else:
            transformed, _ = self._apply_string_standardization(series, spec)
        
        return transformed

