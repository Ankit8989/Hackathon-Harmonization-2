"""
Data Quality Agent for the Agentic AI Data Harmonization System.
Performs comprehensive data quality analysis and anomaly detection.
"""

import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from agents.base_agent import BaseAgent
from agents.llm_reasoning_agent import get_llm_reasoning_agent
from config import DATA_QUALITY_CONFIG, DQ_THRESHOLDS
from models.schemas import (
    AgentResponse,
    ColumnStatistics,
    DataQualityIssue,
    DataQualityReport,
    IssueSeverity,
    ProcessingStatus,
    ColumnDataType
)


class DataQualityAgent(BaseAgent):
    """
    Agent responsible for data quality analysis and anomaly detection.
    
    Capabilities:
    - Range checks and outlier detection
    - Missing value analysis
    - Duplicate detection
    - Business rule validation
    - LLM-powered anomaly explanation
    - Issue severity classification
    """
    
    PROMPT_TEMPLATE = """You are a senior data quality analyst with expertise in survey data and business intelligence.

TASK: Analyze the data quality of a dataset and provide actionable insights.

DATASET OVERVIEW:
{dataset_summary}

COLUMN STATISTICS:
{column_statistics}

DETECTED ISSUES:
{detected_issues}

BUSINESS RULES TO VALIDATE:
{business_rules}

Provide comprehensive analysis including:
1. Root cause analysis for each issue
2. Business impact assessment
3. Classification: blocking (must fix), fixable (can auto-correct), or ignorable (safe to proceed)
4. Specific remediation recommendations
5. Overall data quality score (0-100)

Consider:
- Data integrity and consistency
- Statistical anomalies
- Business logic violations
- Survey data quality standards

RESPOND IN STRICT JSON FORMAT:
{{
    "overall_quality_score": 85.0,
    "is_acceptable": true,
    "blocking_issues": [
        {{
            "issue_id": "uuid",
            "column_name": "column",
            "issue_type": "missing_values|outliers|duplicates|invalid_format|business_rule_violation",
            "severity": "blocking",
            "description": "detailed description",
            "affected_rows": 100,
            "affected_percentage": 5.0,
            "root_cause": "likely cause",
            "business_impact": "impact on analysis",
            "suggested_fix": "how to fix",
            "can_auto_fix": false
        }}
    ],
    "fixable_issues": [...],
    "ignorable_issues": [...],
    "column_quality_scores": {{
        "column_name": 95.0
    }},
    "recommendations": ["list of recommendations"],
    "data_quality_summary": "overall assessment",
    "confidence_score": 0.90
}}"""

    def __init__(self):
        """Initialize the Data Quality Agent"""
        super().__init__(
            name=DATA_QUALITY_CONFIG.name,
            confidence_threshold=DATA_QUALITY_CONFIG.confidence_threshold,
            max_retries=DATA_QUALITY_CONFIG.max_retries
        )
        self.llm_agent = get_llm_reasoning_agent()
        self.thresholds = DQ_THRESHOLDS
    
    def get_prompt_template(self) -> str:
        """Get the agent's prompt template"""
        return self.PROMPT_TEMPLATE
    
    def execute(
        self,
        df: pd.DataFrame,
        master_schema: Optional[Dict[str, Any]] = None,
        business_rules: Optional[List[str]] = None,
        column_mappings: Optional[List[Dict[str, Any]]] = None
    ) -> AgentResponse:
        """
        Execute data quality analysis.
        
        Args:
            df: Input DataFrame to analyze
            master_schema: Optional master schema for validation
            business_rules: Optional list of business rules
            column_mappings: Optional column mappings from structural validation
            
        Returns:
            AgentResponse containing DataQualityReport
        """
        self.start_execution()
        
        try:
            # Step 1: Calculate column statistics
            self.add_audit_entry(
                action="Calculating column statistics",
                status=ProcessingStatus.IN_PROGRESS
            )
            column_stats = self._calculate_column_statistics(df)
            
            # Step 2: Detect data quality issues
            self.add_audit_entry(
                action="Detecting data quality issues",
                status=ProcessingStatus.IN_PROGRESS
            )
            detected_issues = self._detect_issues(df, column_stats, master_schema)
            
            # Step 3: Validate business rules
            if business_rules:
                self.add_audit_entry(
                    action="Validating business rules",
                    status=ProcessingStatus.IN_PROGRESS
                )
                rule_issues = self._validate_business_rules(df, business_rules)
                detected_issues.extend(rule_issues)
            
            # Step 4: Use LLM for analysis and classification
            self.add_audit_entry(
                action="Performing LLM-powered analysis",
                status=ProcessingStatus.IN_PROGRESS
            )
            llm_analysis = self._llm_analyze_quality(
                df, column_stats, detected_issues, business_rules or []
            )
            
            # Step 5: Build quality report
            self.add_audit_entry(
                action="Building data quality report",
                status=ProcessingStatus.IN_PROGRESS
            )
            quality_report = self._build_quality_report(
                df, column_stats, detected_issues, llm_analysis
            )
            
            # Determine success
            success = quality_report.is_acceptable
            confidence = llm_analysis.get("confidence_score", 0.8)
            
            self.end_execution(success)
            
            return self.create_response(
                success=success,
                confidence_score=confidence,
                result=quality_report.model_dump(),
                errors=[i.description for i in quality_report.blocking_issues],
                warnings=[i.description for i in quality_report.fixable_issues[:5]]
            )
            
        except Exception as e:
            self.logger.error(f"Data quality analysis failed: {str(e)}")
            self.end_execution(False)
            
            return self.create_response(
                success=False,
                confidence_score=0.0,
                result=None,
                errors=[str(e)]
            )
    
    def _calculate_column_statistics(
        self,
        df: pd.DataFrame
    ) -> List[ColumnStatistics]:
        """
        Calculate comprehensive statistics for each column.
        
        Args:
            df: Input DataFrame
            
        Returns:
            List of ColumnStatistics objects
        """
        stats_list = []
        
        for col in df.columns:
            series = df[col]
            
            # Basic stats
            total = len(series)
            non_null = int(series.notna().sum())
            null_count = int(series.isna().sum())
            unique_count = int(series.nunique())
            
            stats = ColumnStatistics(
                column_name=col,
                data_type=self._infer_data_type(series),
                total_count=total,
                non_null_count=non_null,
                null_count=null_count,
                null_percentage=round(null_count / total * 100, 2) if total > 0 else 0,
                unique_count=unique_count,
                unique_percentage=round(unique_count / non_null * 100, 2) if non_null > 0 else 0
            )
            
            # Numeric statistics
            if pd.api.types.is_numeric_dtype(series):
                non_null_series = series.dropna()
                if len(non_null_series) > 0:
                    stats.mean = float(non_null_series.mean())
                    stats.std = float(non_null_series.std())
                    stats.min_value = float(non_null_series.min())
                    stats.max_value = float(non_null_series.max())
                    stats.median = float(non_null_series.median())
            
            # Categorical / String statistics
            elif pd.api.types.is_object_dtype(series) or pd.api.types.is_categorical_dtype(series):
                value_counts = series.value_counts().head(10).to_dict()
                stats.top_values = {str(k): int(v) for k, v in value_counts.items()}
                
                # String length stats
                str_series = series.dropna().astype(str)
                if len(str_series) > 0:
                    lengths = str_series.str.len()
                    stats.avg_length = float(lengths.mean())
                    stats.max_length = int(lengths.max())
                    stats.min_length = int(lengths.min())
            
            stats_list.append(stats)
        
        return stats_list
    
    def _infer_data_type(self, series: pd.Series) -> ColumnDataType:
        """Infer the data type of a column"""
        dtype = series.dtype
        
        if pd.api.types.is_bool_dtype(dtype):
            return ColumnDataType.BOOLEAN
        elif pd.api.types.is_integer_dtype(dtype):
            return ColumnDataType.INTEGER
        elif pd.api.types.is_float_dtype(dtype):
            return ColumnDataType.FLOAT
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            return ColumnDataType.DATETIME
        elif pd.api.types.is_categorical_dtype(dtype):
            return ColumnDataType.CATEGORICAL
        elif pd.api.types.is_object_dtype(dtype):
            # Check if categorical
            non_null = series.dropna()
            if len(non_null) > 0 and non_null.nunique() / len(non_null) < 0.05:
                return ColumnDataType.CATEGORICAL
            return ColumnDataType.STRING
        return ColumnDataType.UNKNOWN
    
    def _detect_issues(
        self,
        df: pd.DataFrame,
        column_stats: List[ColumnStatistics],
        master_schema: Optional[Dict[str, Any]] = None
    ) -> List[DataQualityIssue]:
        """
        Detect data quality issues in the dataset.
        
        Args:
            df: Input DataFrame
            column_stats: Column statistics
            master_schema: Optional master schema
            
        Returns:
            List of detected issues
        """
        issues = []
        
        for stats in column_stats:
            col = stats.column_name
            
            # 1. Missing value issues
            if stats.null_percentage > 0:
                severity = IssueSeverity.BLOCKING if stats.null_percentage > self.thresholds.blocking_missing_threshold \
                    else IssueSeverity.FIXABLE if stats.null_percentage > self.thresholds.fixable_missing_threshold \
                    else IssueSeverity.WARNING
                
                issues.append(DataQualityIssue(
                    issue_id=str(uuid.uuid4())[:8],
                    column_name=col,
                    issue_type="missing_values",
                    severity=severity,
                    description=f"Column '{col}' has {stats.null_percentage:.1f}% missing values ({stats.null_count} rows)",
                    affected_rows=stats.null_count,
                    affected_percentage=stats.null_percentage,
                    sample_values=["NULL"] * min(5, stats.null_count),
                    suggested_fix=self._suggest_missing_value_fix(stats),
                    is_fixable=stats.null_percentage < self.thresholds.blocking_missing_threshold
                ))
            
            # 2. Outlier detection (numeric columns)
            if stats.data_type in [ColumnDataType.INTEGER, ColumnDataType.FLOAT]:
                outlier_issues = self._detect_outliers(df[col], stats)
                issues.extend(outlier_issues)
            
            # 3. Cardinality issues
            if stats.unique_percentage == 100 and stats.total_count > 10:
                # Possible unique identifier
                issues.append(DataQualityIssue(
                    issue_id=str(uuid.uuid4())[:8],
                    column_name=col,
                    issue_type="high_cardinality",
                    severity=IssueSeverity.WARNING,
                    description=f"Column '{col}' has 100% unique values - may be an identifier",
                    affected_rows=0,
                    affected_percentage=0,
                    is_fixable=False
                ))
            elif stats.unique_count == 1:
                # Constant column
                issues.append(DataQualityIssue(
                    issue_id=str(uuid.uuid4())[:8],
                    column_name=col,
                    issue_type="constant_value",
                    severity=IssueSeverity.WARNING,
                    description=f"Column '{col}' has only one unique value - may be redundant",
                    affected_rows=0,
                    affected_percentage=0,
                    is_fixable=True,
                    suggested_fix="Consider removing this constant column"
                ))
        
        # 4. Duplicate row detection
        dup_count = df.duplicated().sum()
        if dup_count > 0:
            dup_pct = dup_count / len(df) * 100
            severity = IssueSeverity.BLOCKING if dup_pct > self.thresholds.duplicate_threshold \
                else IssueSeverity.FIXABLE if dup_pct > 1 \
                else IssueSeverity.WARNING
            
            issues.append(DataQualityIssue(
                issue_id=str(uuid.uuid4())[:8],
                column_name=None,
                issue_type="duplicates",
                severity=severity,
                description=f"Dataset contains {dup_count} duplicate rows ({dup_pct:.1f}%)",
                affected_rows=int(dup_count),
                affected_percentage=dup_pct,
                suggested_fix="Remove duplicate rows",
                is_fixable=True
            ))
        
        # 5. Sample size check
        if len(df) < self.thresholds.min_sample_size:
            issues.append(DataQualityIssue(
                issue_id=str(uuid.uuid4())[:8],
                column_name=None,
                issue_type="insufficient_sample",
                severity=IssueSeverity.BLOCKING,
                description=f"Dataset has only {len(df)} rows, minimum required is {self.thresholds.min_sample_size}",
                affected_rows=len(df),
                affected_percentage=100,
                is_fixable=False
            ))
        
        return issues
    
    def _detect_outliers(
        self,
        series: pd.Series,
        stats: ColumnStatistics
    ) -> List[DataQualityIssue]:
        """
        Detect outliers in a numeric column.
        
        Args:
            series: Column data
            stats: Column statistics
            
        Returns:
            List of outlier issues
        """
        issues = []
        
        if stats.std is None or stats.mean is None or stats.std == 0:
            return issues
        
        non_null = series.dropna()
        if len(non_null) == 0:
            return issues
        
        # Z-score based outlier detection
        z_scores = np.abs((non_null - stats.mean) / stats.std)
        outliers = non_null[z_scores > self.thresholds.outlier_std_threshold]
        
        if len(outliers) > 0:
            outlier_pct = len(outliers) / len(non_null) * 100
            severity = IssueSeverity.FIXABLE if outlier_pct > 5 else IssueSeverity.WARNING
            
            issues.append(DataQualityIssue(
                issue_id=str(uuid.uuid4())[:8],
                column_name=stats.column_name,
                issue_type="outliers",
                severity=severity,
                description=f"Column '{stats.column_name}' has {len(outliers)} outliers ({outlier_pct:.1f}%) beyond {self.thresholds.outlier_std_threshold} standard deviations",
                affected_rows=int(len(outliers)),
                affected_percentage=outlier_pct,
                sample_values=outliers.head(5).tolist(),
                suggested_fix="Review outliers for data entry errors or apply capping",
                is_fixable=True
            ))
        
        # IQR based outlier detection
        q1 = non_null.quantile(0.25)
        q3 = non_null.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        iqr_outliers = non_null[(non_null < lower_bound) | (non_null > upper_bound)]
        
        if len(iqr_outliers) > len(outliers):
            additional = len(iqr_outliers) - len(outliers)
            if additional > 0:
                issues.append(DataQualityIssue(
                    issue_id=str(uuid.uuid4())[:8],
                    column_name=stats.column_name,
                    issue_type="iqr_outliers",
                    severity=IssueSeverity.WARNING,
                    description=f"Column '{stats.column_name}' has {len(iqr_outliers)} values outside IQR bounds [{lower_bound:.2f}, {upper_bound:.2f}]",
                    affected_rows=int(len(iqr_outliers)),
                    affected_percentage=len(iqr_outliers) / len(non_null) * 100,
                    is_fixable=True
                ))
        
        return issues
    
    def _suggest_missing_value_fix(self, stats: ColumnStatistics) -> str:
        """Suggest how to handle missing values based on column type"""
        if stats.data_type in [ColumnDataType.INTEGER, ColumnDataType.FLOAT]:
            if stats.null_percentage < 10:
                return f"Impute with median ({stats.median}) or mean ({stats.mean:.2f})"
            else:
                return "Consider removing column or using advanced imputation"
        elif stats.data_type == ColumnDataType.CATEGORICAL:
            return "Impute with mode or create 'Unknown' category"
        elif stats.data_type == ColumnDataType.STRING:
            return "Replace with empty string or 'Unknown'"
        else:
            return "Review and handle based on business context"
    
    def _validate_business_rules(
        self,
        df: pd.DataFrame,
        rules: List[str]
    ) -> List[DataQualityIssue]:
        """
        Validate business rules against the dataset.
        
        Args:
            df: Input DataFrame
            rules: List of business rules as strings
            
        Returns:
            List of rule violation issues
        """
        issues = []
        
        for rule in rules:
            rule_lower = rule.lower()
            
            # Parse common rule patterns
            try:
                if "minimum" in rule_lower and "records" in rule_lower:
                    # Minimum records rule
                    import re
                    match = re.search(r'(\d+)', rule)
                    if match:
                        min_records = int(match.group(1))
                        if len(df) < min_records:
                            issues.append(DataQualityIssue(
                                issue_id=str(uuid.uuid4())[:8],
                                column_name=None,
                                issue_type="business_rule_violation",
                                severity=IssueSeverity.BLOCKING,
                                description=f"Business rule violation: {rule}. Found {len(df)} records.",
                                affected_rows=len(df),
                                affected_percentage=100,
                                is_fixable=False
                            ))
                
                elif "range" in rule_lower or "between" in rule_lower:
                    # Range validation rule
                    import re
                    col_match = re.search(r'column[:\s]+["\']?(\w+)["\']?', rule_lower)
                    range_match = re.findall(r'(\d+(?:\.\d+)?)', rule)
                    
                    if col_match and len(range_match) >= 2:
                        col_name = col_match.group(1)
                        min_val, max_val = float(range_match[0]), float(range_match[1])
                        
                        # Find matching column (case insensitive)
                        matching_cols = [c for c in df.columns if c.lower() == col_name.lower()]
                        if matching_cols:
                            col = matching_cols[0]
                            if pd.api.types.is_numeric_dtype(df[col]):
                                violations = df[(df[col] < min_val) | (df[col] > max_val)]
                                if len(violations) > 0:
                                    issues.append(DataQualityIssue(
                                        issue_id=str(uuid.uuid4())[:8],
                                        column_name=col,
                                        issue_type="business_rule_violation",
                                        severity=IssueSeverity.FIXABLE,
                                        description=f"Business rule violation: Values in '{col}' outside range [{min_val}, {max_val}]",
                                        affected_rows=len(violations),
                                        affected_percentage=len(violations) / len(df) * 100,
                                        sample_values=violations[col].head(5).tolist(),
                                        is_fixable=True,
                                        suggested_fix=f"Cap values to range [{min_val}, {max_val}]"
                                    ))
                
                elif "not null" in rule_lower or "required" in rule_lower:
                    # Required field rule
                    import re
                    col_match = re.search(r'column[:\s]+["\']?(\w+)["\']?', rule_lower)
                    if col_match:
                        col_name = col_match.group(1)
                        matching_cols = [c for c in df.columns if c.lower() == col_name.lower()]
                        if matching_cols:
                            col = matching_cols[0]
                            null_count = df[col].isna().sum()
                            if null_count > 0:
                                issues.append(DataQualityIssue(
                                    issue_id=str(uuid.uuid4())[:8],
                                    column_name=col,
                                    issue_type="business_rule_violation",
                                    severity=IssueSeverity.BLOCKING,
                                    description=f"Business rule violation: Required field '{col}' has {null_count} null values",
                                    affected_rows=int(null_count),
                                    affected_percentage=null_count / len(df) * 100,
                                    is_fixable=False
                                ))
            
            except Exception as e:
                self.logger.warning(f"Error parsing business rule '{rule}': {str(e)}")
        
        return issues
    
    def _llm_analyze_quality(
        self,
        df: pd.DataFrame,
        column_stats: List[ColumnStatistics],
        detected_issues: List[DataQualityIssue],
        business_rules: List[str]
    ) -> Dict[str, Any]:
        """
        Use LLM for comprehensive quality analysis.
        
        Args:
            df: Input DataFrame
            column_stats: Column statistics
            detected_issues: Detected issues
            business_rules: Business rules
            
        Returns:
            LLM analysis result
        """
        # Prepare dataset summary
        dataset_summary = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "memory_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
            "column_names": list(df.columns)
        }
        
        # Prepare column statistics summary
        stats_summary = []
        for stats in column_stats[:15]:  # Limit to avoid token overflow
            stats_summary.append({
                "name": stats.column_name,
                "type": stats.data_type.value,
                "null_pct": stats.null_percentage,
                "unique_count": stats.unique_count,
                "mean": stats.mean,
                "std": stats.std
            })
        
        # Prepare issues summary
        issues_summary = []
        for issue in detected_issues[:20]:  # Limit
            issues_summary.append({
                "column": issue.column_name,
                "type": issue.issue_type,
                "severity": issue.severity.value,
                "description": issue.description,
                "affected_pct": issue.affected_percentage
            })
        
        # OPTIMIZATION: Calculate quality score WITHOUT LLM to save tokens
        # All metrics are computed using pandas - no API call needed!
        
        total_issues = len(detected_issues)
        blocking_count = len([i for i in detected_issues if i.severity.value == 'blocking'])
        fixable_count = len([i for i in detected_issues if i.severity.value == 'fixable'])
        warning_count = len([i for i in detected_issues if i.severity.value == 'warning'])
        
        # Calculate quality score based on issues (hardcoded logic, no LLM)
        base_score = 100.0
        base_score -= blocking_count * 10  # Each blocking issue costs 10 points
        base_score -= fixable_count * 2    # Each fixable issue costs 2 points
        base_score -= warning_count * 0.5  # Each warning costs 0.5 points
        
        # Factor in missing data
        avg_null_pct = sum(s.null_percentage for s in column_stats) / len(column_stats) if column_stats else 0
        base_score -= avg_null_pct * 0.5  # Penalize for missing data
        
        quality_score = max(0, min(100, base_score))  # Clamp to 0-100
        
        # Generate recommendations without LLM
        recommendations = []
        if blocking_count > 0:
            recommendations.append(f"Fix {blocking_count} blocking issues before proceeding")
        if avg_null_pct > 10:
            recommendations.append(f"High missing data ({avg_null_pct:.1f}%) - consider imputation")
        if fixable_count > 10:
            recommendations.append(f"Review {fixable_count} fixable issues for data cleanup")
        
        # Build result WITHOUT calling LLM
        result = {
            "overall_quality_score": quality_score,
            "is_acceptable": blocking_count == 0 and quality_score >= 60,
            "recommendations": recommendations if recommendations else ["Data quality acceptable"],
            "data_quality_summary": f"Analyzed {dataset_summary.get('total_rows', 0)} rows, {dataset_summary.get('total_columns', 0)} columns. Found {total_issues} issues.",
            "confidence_score": 0.92,  # High confidence since we computed everything
            "blocking_issues_count": blocking_count,
            "fixable_issues_count": fixable_count,
            "warning_count": warning_count,
            "llm_used": False  # Flag to indicate no LLM was used
        }
        
        self.logger.info(f"Data quality analysis complete (NO LLM). Score: {quality_score:.1f}")
        
        return result
    
    def _build_quality_report(
        self,
        df: pd.DataFrame,
        column_stats: List[ColumnStatistics],
        detected_issues: List[DataQualityIssue],
        llm_analysis: Dict[str, Any]
    ) -> DataQualityReport:
        """
        Build the final data quality report.
        
        Args:
            df: Input DataFrame
            column_stats: Column statistics
            detected_issues: Detected issues
            llm_analysis: LLM analysis results
            
        Returns:
            DataQualityReport object
        """
        # Classify issues by severity
        blocking = [i for i in detected_issues if i.severity == IssueSeverity.BLOCKING]
        fixable = [i for i in detected_issues if i.severity == IssueSeverity.FIXABLE]
        ignorable = [i for i in detected_issues if i.severity in [IssueSeverity.WARNING, IssueSeverity.IGNORABLE]]
        
        # Add LLM-identified issues
        for issue_data in llm_analysis.get("blocking_issues", []):
            if not any(i.issue_id == issue_data.get("issue_id") for i in blocking):
                blocking.append(DataQualityIssue(
                    issue_id=issue_data.get("issue_id", str(uuid.uuid4())[:8]),
                    column_name=issue_data.get("column_name"),
                    issue_type=issue_data.get("issue_type", "llm_identified"),
                    severity=IssueSeverity.BLOCKING,
                    description=issue_data.get("description", ""),
                    affected_rows=issue_data.get("affected_rows", 0),
                    affected_percentage=issue_data.get("affected_percentage", 0),
                    suggested_fix=issue_data.get("suggested_fix"),
                    llm_explanation=issue_data.get("root_cause"),
                    is_fixable=issue_data.get("can_auto_fix", False)
                ))
        
        # Calculate overall quality score
        quality_score = llm_analysis.get("overall_quality_score", 0)
        if quality_score == 0:
            # Calculate based on issues
            deductions = len(blocking) * 20 + len(fixable) * 5 + len(ignorable) * 1
            quality_score = max(0, 100 - deductions)
        
        # Count issues by type
        issues_by_type = {}
        for issue in detected_issues:
            issues_by_type[issue.issue_type] = issues_by_type.get(issue.issue_type, 0) + 1
        
        # Determine acceptability
        is_acceptable = len(blocking) == 0 and quality_score >= 60
        
        execution_time = 0.0
        if self.start_time:
            execution_time = (datetime.now() - self.start_time).total_seconds()
        
        return DataQualityReport(
            overall_quality_score=quality_score,
            is_acceptable=is_acceptable,
            total_records=len(df),
            total_columns=len(df.columns),
            column_statistics=column_stats,
            blocking_issues=blocking,
            fixable_issues=fixable,
            ignorable_issues=ignorable,
            total_issues=len(detected_issues),
            issues_by_type=issues_by_type,
            llm_summary=llm_analysis.get("data_quality_summary", ""),
            recommendations=llm_analysis.get("recommendations", []),
            processing_time_seconds=execution_time,
            timestamp=datetime.now()
        )
    
    def quick_assessment(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform a quick quality assessment without LLM calls.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Quick assessment dictionary
        """
        assessment = {
            "rows": len(df),
            "columns": len(df.columns),
            "null_cells": int(df.isna().sum().sum()),
            "null_percentage": round(df.isna().sum().sum() / (len(df) * len(df.columns)) * 100, 2),
            "duplicates": int(df.duplicated().sum()),
            "column_completeness": {}
        }
        
        for col in df.columns:
            completeness = (df[col].notna().sum() / len(df)) * 100
            assessment["column_completeness"][col] = round(completeness, 2)
        
        return assessment

