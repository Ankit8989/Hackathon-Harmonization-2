"""
Structural Validation Agent for the Agentic AI Data Harmonization System.
Validates dataset structure against master metadata schema.
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from agents.base_agent import BaseAgent
from agents.llm_reasoning_agent import get_llm_reasoning_agent
from config import STRUCTURAL_VALIDATION_CONFIG
from models.schemas import (
    AgentResponse,
    ColumnMapping,
    MappingConfidence,
    ProcessingStatus,
    SchemaDrift,
    SchemaValidationResult,
    ColumnDataType
)
from utils.file_handlers import get_dataframe_summary


class StructuralValidationAgent(BaseAgent):
    """
    Agent responsible for validating dataset structure against master schema.
    
    Capabilities:
    - Compare columns against master metadata
    - Detect missing, extra, and mismatched fields
    - Use LLM to infer likely mappings
    - Generate schema drift reports
    """
    
    PROMPT_TEMPLATE = """You are an expert data engineer specializing in schema validation and data harmonization.

TASK: Validate the structure of a source dataset against a master schema definition.

SOURCE DATASET SCHEMA:
{source_schema}

MASTER SCHEMA DEFINITION:
{master_schema}

Analyze and provide:
1. Column-by-column mapping with confidence scores
2. Missing required columns
3. Extra columns not in master schema
4. Type mismatches
5. Ambiguous mappings that need resolution
6. Overall schema drift assessment

Consider:
- Column name similarities (exact match, case differences, underscores vs spaces)
- Semantic meaning of column names
- Data type compatibility
- Common abbreviations and synonyms

RESPOND IN STRICT JSON FORMAT:
{{
    "mappings": [
        {{
            "source_column": "col_name",
            "target_column": "mapped_master_col or UNMAPPED",
            "confidence": 0.95,
            "confidence_level": "high|medium|low|uncertain",
            "reasoning": "why this mapping",
            "transformation": "transformation needed or null",
            "requires_review": false
        }}
    ],
    "missing_required_columns": ["list of required master columns not found"],
    "extra_source_columns": ["list of source columns with no match"],
    "type_mismatches": {{
        "column_name": {{"source_type": "string", "expected_type": "integer"}}
    }},
    "ambiguous_mappings": ["columns with multiple possible matches"],
    "schema_drift_score": 0.15,
    "schema_drift_severity": "blocking|fixable|ignorable",
    "is_valid": true,
    "validation_errors": ["list of critical validation errors"],
    "validation_warnings": ["list of non-critical warnings"],
    "overall_confidence": 0.85,
    "analysis_summary": "detailed summary of the validation"
}}"""

    def __init__(self):
        """Initialize the Structural Validation Agent"""
        super().__init__(
            name=STRUCTURAL_VALIDATION_CONFIG.name,
            confidence_threshold=STRUCTURAL_VALIDATION_CONFIG.confidence_threshold,
            max_retries=STRUCTURAL_VALIDATION_CONFIG.max_retries
        )
        self.llm_agent = get_llm_reasoning_agent()
    
    def get_prompt_template(self) -> str:
        """Get the agent's prompt template"""
        return self.PROMPT_TEMPLATE
    
    def execute(
        self,
        df: pd.DataFrame,
        master_schema: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """
        Execute structural validation of the dataset.
        
        Args:
            df: Input DataFrame to validate
            master_schema: Master schema definition
            metadata: Optional additional metadata
            
        Returns:
            AgentResponse containing SchemaValidationResult
        """
        self.start_execution()
        
        try:
            # Step 1: Extract source schema information
            self.add_audit_entry(
                action="Extracting source schema",
                status=ProcessingStatus.IN_PROGRESS
            )
            source_schema = self._extract_source_schema(df, metadata)
            
            # Step 2: Perform rule-based matching first
            self.add_audit_entry(
                action="Performing rule-based column matching",
                status=ProcessingStatus.IN_PROGRESS
            )
            rule_based_mappings = self._rule_based_matching(source_schema, master_schema)
            
            # Step 3: Use LLM for unmapped and ambiguous columns
            self.add_audit_entry(
                action="Using LLM for intelligent mapping inference",
                status=ProcessingStatus.IN_PROGRESS
            )
            llm_result = self._llm_enhanced_validation(source_schema, master_schema)
            
            # Step 4: Merge and reconcile mappings
            self.add_audit_entry(
                action="Merging rule-based and LLM mappings",
                status=ProcessingStatus.IN_PROGRESS
            )
            final_mappings = self._merge_mappings(rule_based_mappings, llm_result)
            
            # Step 5: Calculate schema drift
            schema_drift = self._calculate_schema_drift(
                source_schema,
                master_schema,
                final_mappings,
                llm_result
            )
            
            # Step 6: Build validation result
            validation_result = self._build_validation_result(
                final_mappings,
                schema_drift,
                llm_result
            )
            
            # Determine success
            success = validation_result.is_valid or schema_drift.drift_severity != "blocking"
            confidence = validation_result.confidence_score
            
            self.end_execution(success)
            
            return self.create_response(
                success=success,
                confidence_score=confidence,
                result=validation_result.model_dump(),
                errors=validation_result.validation_errors,
                warnings=validation_result.validation_warnings
            )
            
        except Exception as e:
            self.logger.error(f"Structural validation failed: {str(e)}")
            self.end_execution(False)
            
            return self.create_response(
                success=False,
                confidence_score=0.0,
                result=None,
                errors=[str(e)]
            )
    
    def _extract_source_schema(
        self,
        df: pd.DataFrame,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Extract schema information from the source DataFrame.
        
        Args:
            df: Input DataFrame
            metadata: Optional metadata from file loading
            
        Returns:
            Dictionary containing schema information
        """
        schema = {
            "columns": [],
            "row_count": len(df),
            "column_count": len(df.columns)
        }
        
        for col in df.columns:
            col_info = {
                "name": col,
                "dtype": str(df[col].dtype),
                "inferred_type": self._infer_column_type(df[col]),
                "non_null_count": int(df[col].notna().sum()),
                "null_count": int(df[col].isna().sum()),
                "unique_count": int(df[col].nunique()),
                "sample_values": df[col].dropna().head(5).tolist()
            }
            
            # Add SPSS labels if available
            if metadata and "variable_labels" in metadata:
                labels = metadata["variable_labels"]
                if col in labels:
                    col_info["label"] = labels[col]
            
            if metadata and "value_labels" in metadata:
                value_labels = metadata["value_labels"]
                if col in value_labels:
                    col_info["value_labels"] = value_labels[col]
            
            schema["columns"].append(col_info)
        
        return schema
    
    def _infer_column_type(self, series: pd.Series) -> str:
        """
        Infer the semantic data type of a column.
        
        Args:
            series: Pandas Series
            
        Returns:
            Inferred type string
        """
        dtype = series.dtype
        
        if pd.api.types.is_bool_dtype(dtype):
            return ColumnDataType.BOOLEAN.value
        elif pd.api.types.is_integer_dtype(dtype):
            return ColumnDataType.INTEGER.value
        elif pd.api.types.is_float_dtype(dtype):
            return ColumnDataType.FLOAT.value
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            return ColumnDataType.DATETIME.value
        elif pd.api.types.is_categorical_dtype(dtype):
            return ColumnDataType.CATEGORICAL.value
        elif pd.api.types.is_object_dtype(dtype):
            # Check if it could be categorical
            non_null = series.dropna()
            if len(non_null) > 0:
                unique_ratio = non_null.nunique() / len(non_null)
                if unique_ratio < 0.05:  # Less than 5% unique values
                    return ColumnDataType.CATEGORICAL.value
            return ColumnDataType.STRING.value
        else:
            return ColumnDataType.UNKNOWN.value
    
    def _rule_based_matching(
        self,
        source_schema: Dict[str, Any],
        master_schema: Dict[str, Any]
    ) -> List[ColumnMapping]:
        """
        Perform rule-based column matching.
        
        Args:
            source_schema: Source dataset schema
            master_schema: Master schema definition
            
        Returns:
            List of column mappings
        """
        mappings = []
        master_columns = self._get_master_columns(master_schema)
        
        # Create lookup dictionaries
        master_lookup = {col["name"].lower(): col for col in master_columns}
        master_canonical = {col.get("canonical_name", col["name"]).lower(): col for col in master_columns}
        
        # Also create alias lookups
        alias_lookup = {}
        for col in master_columns:
            for alias in col.get("aliases", []):
                alias_lookup[alias.lower()] = col
        
        for source_col in source_schema["columns"]:
            source_name = source_col["name"]
            source_name_lower = source_name.lower()
            source_name_normalized = source_name_lower.replace(" ", "_").replace("-", "_")
            
            mapping = None
            confidence = 0.0
            reasoning = ""
            
            # Check exact match
            if source_name_lower in master_lookup:
                target = master_lookup[source_name_lower]
                mapping = target["name"]
                confidence = 1.0
                reasoning = "Exact name match"
            
            # Check canonical name match
            elif source_name_normalized in master_canonical:
                target = master_canonical[source_name_normalized]
                mapping = target["name"]
                confidence = 0.95
                reasoning = "Normalized name match"
            
            # Check alias match
            elif source_name_lower in alias_lookup:
                target = alias_lookup[source_name_lower]
                mapping = target["name"]
                confidence = 0.90
                reasoning = f"Alias match: '{source_name}' is alias for '{target['name']}'"
            
            # Check for partial matches
            else:
                for master_name, master_col in master_lookup.items():
                    if source_name_normalized in master_name or master_name in source_name_normalized:
                        mapping = master_col["name"]
                        confidence = 0.70
                        reasoning = f"Partial name match: '{source_name}' ~ '{master_col['name']}'"
                        break
            
            # Create mapping object
            if mapping:
                conf_level = MappingConfidence.HIGH if confidence >= 0.9 else \
                            MappingConfidence.MEDIUM if confidence >= 0.7 else \
                            MappingConfidence.LOW if confidence >= 0.5 else \
                            MappingConfidence.UNCERTAIN
                
                mappings.append(ColumnMapping(
                    source_column=source_name,
                    target_column=mapping,
                    confidence=confidence,
                    confidence_level=conf_level,
                    transformation=None,
                    llm_reasoning=None,
                    is_auto_mapped=True,
                    requires_review=confidence < 0.8
                ))
            else:
                # Unmapped column
                mappings.append(ColumnMapping(
                    source_column=source_name,
                    target_column="UNMAPPED",
                    confidence=0.0,
                    confidence_level=MappingConfidence.UNCERTAIN,
                    transformation=None,
                    llm_reasoning="No rule-based match found",
                    is_auto_mapped=False,
                    requires_review=True
                ))
        
        return mappings
    
    def _get_master_columns(self, master_schema: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract column definitions from master schema"""
        columns = master_schema.get("columns", [])
        if not columns and "schema" in master_schema:
            columns = master_schema["schema"].get("columns", [])
        return columns
    
    def _llm_enhanced_validation(
        self,
        source_schema: Dict[str, Any],
        master_schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Use LLM ONLY for ambiguous columns that couldn't be matched by rules.
        Most validation is done without API calls to save tokens.
        
        Args:
            source_schema: Source dataset schema
            master_schema: Master schema definition
            
        Returns:
            LLM analysis result (or hardcoded result if no ambiguous columns)
        """
        # Get unmapped columns from rule-based matching
        source_cols = list(source_schema.get('columns', {}).keys())
        master_cols = [c.get('name', '') for c in master_schema.get('columns', [])]
        
        # Check if we have any ambiguous columns that need LLM
        # If rule-based matching covered most columns, skip LLM entirely
        unmapped_count = len([c for c in source_cols if not self._find_exact_match(c, master_cols)])
        
        # OPTIMIZATION: Only use LLM if we have truly ambiguous columns (< 20 unmapped)
        # For large datasets, skip LLM to save tokens
        if unmapped_count > 50 or len(source_cols) > 500:
            self.logger.info(f"Skipping LLM call - too many columns ({len(source_cols)}). Using rule-based only.")
            return {
                "column_mappings": [],
                "validation_issues": [],
                "confidence_score": 0.7,
                "recommendations": ["Large dataset - using rule-based matching only to save API costs"]
            }
        
        # Only send unmapped columns to LLM (not the entire schema)
        unmapped_cols = [c for c in source_cols[:30] if not self._find_exact_match(c, master_cols)]  # Limit to 30
        
        if not unmapped_cols:
            self.logger.info("All columns matched by rules. Skipping LLM call.")
            return {
                "column_mappings": [],
                "validation_issues": [],
                "confidence_score": 0.95,
                "recommendations": []
            }
        
        # Simplified prompt - only ask about unmapped columns
        simplified_prompt = f"""Match these source columns to the closest master column:

Source columns (unmapped): {unmapped_cols[:20]}

Master columns available: {master_cols}

Return JSON with format:
{{"column_mappings": [{{"source": "col_name", "target": "master_col", "confidence": 0.8}}], "confidence_score": 0.8}}"""
        
        messages = [
            {"role": "system", "content": "You are a data engineer. Return only valid JSON, no explanation."},
            {"role": "user", "content": simplified_prompt}
        ]
        
        response, tokens = self.call_llm(messages, purpose="structural_validation")
        
        # Parse response
        try:
            result = json.loads(response)
        except json.JSONDecodeError:
            import re
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                result = json.loads(json_match.group())
            else:
                result = {"column_mappings": [], "confidence_score": 0.7}
        
        return result
    
    def _find_exact_match(self, source_col: str, master_cols: List[str]) -> bool:
        """Check if source column has an exact or close match in master columns"""
        source_lower = source_col.lower().replace('_', '').replace('-', '').replace(' ', '')
        for master in master_cols:
            master_lower = master.lower().replace('_', '').replace('-', '').replace(' ', '')
            if source_lower == master_lower or source_lower in master_lower or master_lower in source_lower:
                return True
        return False
    
    def _merge_mappings(
        self,
        rule_based: List[ColumnMapping],
        llm_result: Dict[str, Any]
    ) -> List[ColumnMapping]:
        """
        Merge rule-based and LLM mappings, preferring higher confidence.
        
        Args:
            rule_based: Rule-based mapping results
            llm_result: LLM analysis results
            
        Returns:
            Merged list of column mappings
        """
        # Create lookup from rule-based mappings
        rule_lookup = {m.source_column: m for m in rule_based}
        
        # Get LLM mappings
        llm_mappings = llm_result.get("mappings", [])
        
        final_mappings = []
        processed_columns = set()
        
        for llm_map in llm_mappings:
            source_col = llm_map.get("source_column")
            if not source_col:
                continue
            
            processed_columns.add(source_col)
            
            rule_map = rule_lookup.get(source_col)
            llm_confidence = llm_map.get("confidence", 0)
            
            # If rule-based has higher confidence, use it
            if rule_map and rule_map.confidence >= llm_confidence and rule_map.target_column != "UNMAPPED":
                final_mappings.append(rule_map)
            else:
                # Use LLM mapping
                conf_level_str = llm_map.get("confidence_level", "uncertain")
                try:
                    conf_level = MappingConfidence(conf_level_str)
                except ValueError:
                    conf_level = MappingConfidence.UNCERTAIN
                
                final_mappings.append(ColumnMapping(
                    source_column=source_col,
                    target_column=llm_map.get("target_column", "UNMAPPED"),
                    confidence=llm_confidence,
                    confidence_level=conf_level,
                    transformation=llm_map.get("transformation"),
                    llm_reasoning=llm_map.get("reasoning"),
                    is_auto_mapped=llm_confidence >= 0.8,
                    requires_review=llm_map.get("requires_review", llm_confidence < 0.8)
                ))
        
        # Add any rule-based mappings not in LLM results
        for source_col, rule_map in rule_lookup.items():
            if source_col not in processed_columns:
                final_mappings.append(rule_map)
        
        return final_mappings
    
    def _calculate_schema_drift(
        self,
        source_schema: Dict[str, Any],
        master_schema: Dict[str, Any],
        mappings: List[ColumnMapping],
        llm_result: Dict[str, Any]
    ) -> SchemaDrift:
        """
        Calculate schema drift metrics.
        
        Args:
            source_schema: Source dataset schema
            master_schema: Master schema definition
            mappings: Final column mappings
            llm_result: LLM analysis results
            
        Returns:
            SchemaDrift object
        """
        master_columns = self._get_master_columns(master_schema)
        master_col_names = {col["name"] for col in master_columns}
        required_cols = {col["name"] for col in master_columns if col.get("is_required", False)}
        
        # Find mapped target columns
        mapped_targets = {m.target_column for m in mappings if m.target_column != "UNMAPPED"}
        
        # Calculate missing and extra
        missing_columns = list(required_cols - mapped_targets)
        unmapped_sources = [m.source_column for m in mappings if m.target_column == "UNMAPPED"]
        ambiguous = llm_result.get("ambiguous_mappings", [])
        type_mismatches = llm_result.get("type_mismatches", {})
        
        # Calculate drift score
        total_expected = len(master_col_names)
        total_source = len(source_schema["columns"])
        matched = len(mapped_targets)
        
        if total_expected > 0:
            drift_score = 1 - (matched / max(total_expected, total_source))
        else:
            drift_score = 0.0
        
        # Determine severity
        if missing_columns or drift_score > 0.3:
            drift_severity = "blocking"
        elif ambiguous or drift_score > 0.1:
            drift_severity = "fixable"
        else:
            drift_severity = "ignorable"
        
        from models.schemas import IssueSeverity
        severity_map = {
            "blocking": IssueSeverity.BLOCKING,
            "fixable": IssueSeverity.FIXABLE,
            "ignorable": IssueSeverity.IGNORABLE
        }
        
        return SchemaDrift(
            missing_columns=missing_columns,
            extra_columns=unmapped_sources,
            type_mismatches=type_mismatches,
            ambiguous_columns=ambiguous,
            drift_score=drift_score,
            drift_severity=severity_map.get(drift_severity, IssueSeverity.WARNING)
        )
    
    def _build_validation_result(
        self,
        mappings: List[ColumnMapping],
        schema_drift: SchemaDrift,
        llm_result: Dict[str, Any]
    ) -> SchemaValidationResult:
        """
        Build the final validation result.
        
        Args:
            mappings: Column mappings
            schema_drift: Schema drift analysis
            llm_result: LLM analysis results
            
        Returns:
            SchemaValidationResult object
        """
        # Calculate overall confidence
        if mappings:
            avg_confidence = sum(m.confidence for m in mappings) / len(mappings)
        else:
            avg_confidence = 0.0
        
        # Check if valid
        is_valid = (
            len(schema_drift.missing_columns) == 0 and
            schema_drift.drift_severity != "blocking" and
            avg_confidence >= self.confidence_threshold
        )
        
        # Get errors and warnings from LLM
        errors = llm_result.get("validation_errors", [])
        warnings = llm_result.get("validation_warnings", [])
        
        # Add drift-related errors
        if schema_drift.missing_columns:
            errors.append(f"Missing required columns: {schema_drift.missing_columns}")
        
        if schema_drift.type_mismatches:
            for col, mismatch in schema_drift.type_mismatches.items():
                warnings.append(
                    f"Type mismatch for '{col}': expected {mismatch.get('expected_type')}, "
                    f"got {mismatch.get('source_type')}"
                )
        
        execution_time = 0.0
        if self.start_time:
            execution_time = (datetime.now() - self.start_time).total_seconds()
        
        return SchemaValidationResult(
            is_valid=is_valid,
            confidence_score=avg_confidence,
            column_mappings=mappings,
            schema_drift=schema_drift,
            validation_errors=errors,
            validation_warnings=warnings,
            llm_analysis=llm_result.get("analysis_summary"),
            processing_time_seconds=execution_time,
            timestamp=datetime.now()
        )
    
    def validate_single_column(
        self,
        source_column: Dict[str, Any],
        master_schema: Dict[str, Any]
    ) -> ColumnMapping:
        """
        Validate a single column against the master schema.
        
        Args:
            source_column: Source column information
            master_schema: Master schema definition
            
        Returns:
            ColumnMapping for the column
        """
        # Use LLM to find best match
        result, _ = self.llm_agent.infer_column_mappings(
            [source_column],
            self._get_master_columns(master_schema)
        )
        
        if result.get("mappings"):
            mapping_info = result["mappings"][0]
            return ColumnMapping(
                source_column=source_column["name"],
                target_column=mapping_info.get("target_column", "UNMAPPED"),
                confidence=mapping_info.get("confidence", 0),
                confidence_level=MappingConfidence.MEDIUM,
                transformation=mapping_info.get("transformation"),
                llm_reasoning=mapping_info.get("reasoning"),
                is_auto_mapped=True,
                requires_review=mapping_info.get("confidence", 0) < 0.8
            )
        
        return ColumnMapping(
            source_column=source_column["name"],
            target_column="UNMAPPED",
            confidence=0.0,
            confidence_level=MappingConfidence.UNCERTAIN,
            is_auto_mapped=False,
            requires_review=True
        )

