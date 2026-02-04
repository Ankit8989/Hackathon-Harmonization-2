"""
Pydantic schemas and data models for the Agentic AI Data Harmonization System.
Defines all structured outputs used across agents.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, ConfigDict


# =============================================================================
# ENUMERATIONS
# =============================================================================

class ProcessingStatus(str, Enum):
    """Status of processing operations"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    REQUIRES_REVIEW = "requires_review"
    SKIPPED = "skipped"


class IssueSeverity(str, Enum):
    """Severity classification for data issues"""
    BLOCKING = "blocking"  # Must be fixed before proceeding
    FIXABLE = "fixable"    # Can be auto-corrected
    IGNORABLE = "ignorable"  # Can be ignored safely
    WARNING = "warning"    # Should be noted but not critical


class ColumnDataType(str, Enum):
    """Supported column data types"""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    CATEGORICAL = "categorical"
    UNKNOWN = "unknown"


class MappingConfidence(str, Enum):
    """Confidence level for column mappings"""
    HIGH = "high"      # >90% confidence
    MEDIUM = "medium"  # 70-90% confidence
    LOW = "low"        # 50-70% confidence
    UNCERTAIN = "uncertain"  # <50% confidence


# =============================================================================
# COLUMN AND SCHEMA MODELS
# =============================================================================

class ColumnMetadata(BaseModel):
    """Metadata for a single column"""
    model_config = ConfigDict(extra='allow')
    
    name: str
    data_type: ColumnDataType = ColumnDataType.UNKNOWN
    description: Optional[str] = None
    is_required: bool = False
    is_key: bool = False
    allowed_values: Optional[List[Any]] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    pattern: Optional[str] = None  # Regex pattern for validation
    default_value: Optional[Any] = None


class ColumnMapping(BaseModel):
    """Mapping between source and target columns"""
    model_config = ConfigDict(extra='allow')
    
    source_column: str
    target_column: str
    confidence: float = Field(ge=0.0, le=1.0)
    confidence_level: MappingConfidence = MappingConfidence.UNCERTAIN
    transformation: Optional[str] = None  # Description of required transformation
    llm_reasoning: Optional[str] = None  # LLM's explanation for the mapping
    is_auto_mapped: bool = True
    requires_review: bool = False


class SchemaDrift(BaseModel):
    """Schema drift detection result"""
    missing_columns: List[str] = Field(default_factory=list)
    extra_columns: List[str] = Field(default_factory=list)
    type_mismatches: Dict[str, Dict[str, str]] = Field(default_factory=dict)
    ambiguous_columns: List[str] = Field(default_factory=list)
    drift_score: float = Field(ge=0.0, le=1.0, default=0.0)
    drift_severity: IssueSeverity = IssueSeverity.IGNORABLE


class SchemaValidationResult(BaseModel):
    """Complete schema validation result"""
    model_config = ConfigDict(extra='allow')
    
    is_valid: bool = False
    confidence_score: float = Field(ge=0.0, le=1.0, default=0.0)
    column_mappings: List[ColumnMapping] = Field(default_factory=list)
    schema_drift: SchemaDrift = Field(default_factory=SchemaDrift)
    validation_errors: List[str] = Field(default_factory=list)
    validation_warnings: List[str] = Field(default_factory=list)
    llm_analysis: Optional[str] = None
    processing_time_seconds: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.now)


# =============================================================================
# DATA QUALITY MODELS
# =============================================================================

class ColumnStatistics(BaseModel):
    """Statistical summary for a column"""
    column_name: str
    data_type: ColumnDataType = ColumnDataType.UNKNOWN
    total_count: int = 0
    non_null_count: int = 0
    null_count: int = 0
    null_percentage: float = 0.0
    unique_count: int = 0
    unique_percentage: float = 0.0
    
    # Numeric statistics
    mean: Optional[float] = None
    std: Optional[float] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    median: Optional[float] = None
    
    # Categorical statistics
    top_values: Optional[Dict[str, int]] = None
    
    # String statistics
    avg_length: Optional[float] = None
    max_length: Optional[int] = None
    min_length: Optional[int] = None


class DataQualityIssue(BaseModel):
    """Individual data quality issue"""
    model_config = ConfigDict(extra='allow')
    
    issue_id: str
    column_name: Optional[str] = None
    issue_type: str  # missing_values, outliers, duplicates, invalid_format, etc.
    severity: IssueSeverity = IssueSeverity.WARNING
    description: str
    affected_rows: int = 0
    affected_percentage: float = 0.0
    sample_values: List[Any] = Field(default_factory=list)
    suggested_fix: Optional[str] = None
    llm_explanation: Optional[str] = None
    is_fixable: bool = True
    fix_applied: bool = False


class DataQualityReport(BaseModel):
    """Complete data quality assessment report"""
    model_config = ConfigDict(extra='allow')
    
    overall_quality_score: float = Field(ge=0.0, le=100.0, default=0.0)
    is_acceptable: bool = False
    total_records: int = 0
    total_columns: int = 0
    
    # Statistics
    column_statistics: List[ColumnStatistics] = Field(default_factory=list)
    
    # Issues by severity
    blocking_issues: List[DataQualityIssue] = Field(default_factory=list)
    fixable_issues: List[DataQualityIssue] = Field(default_factory=list)
    ignorable_issues: List[DataQualityIssue] = Field(default_factory=list)
    
    # Summary counts
    total_issues: int = 0
    issues_by_type: Dict[str, int] = Field(default_factory=dict)
    
    # LLM analysis
    llm_summary: Optional[str] = None
    recommendations: List[str] = Field(default_factory=list)
    
    processing_time_seconds: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.now)


# =============================================================================
# HARMONIZATION MODELS
# =============================================================================

class TransformationRecord(BaseModel):
    """Record of a single transformation applied"""
    column_name: str
    transformation_type: str  # scale_normalize, value_map, date_format, etc.
    original_value: Any
    transformed_value: Any
    transformation_rule: str
    success: bool = True
    error_message: Optional[str] = None


class ColumnTransformationSummary(BaseModel):
    """Summary of transformations for a column"""
    source_column: str
    target_column: str
    transformation_type: str
    records_transformed: int = 0
    records_failed: int = 0
    sample_transformations: List[TransformationRecord] = Field(default_factory=list)


class HarmonizationResult(BaseModel):
    """Complete harmonization result"""
    model_config = ConfigDict(extra='allow')
    
    success: bool = False
    confidence_score: float = Field(ge=0.0, le=1.0, default=0.0)
    
    # Record counts
    input_records: int = 0
    output_records: int = 0
    records_modified: int = 0
    records_dropped: int = 0
    
    # Transformation details
    column_transformations: List[ColumnTransformationSummary] = Field(default_factory=list)
    applied_mappings: List[ColumnMapping] = Field(default_factory=list)
    
    # Schema changes
    columns_added: List[str] = Field(default_factory=list)
    columns_removed: List[str] = Field(default_factory=list)
    columns_renamed: Dict[str, str] = Field(default_factory=dict)
    
    # Issues encountered
    harmonization_issues: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    
    # LLM decisions
    llm_decisions: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Output paths
    output_file_path: Optional[str] = None
    comparison_file_path: Optional[str] = None
    
    processing_time_seconds: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.now)


# =============================================================================
# AGENT RESPONSE MODELS
# =============================================================================

class AgentResponse(BaseModel):
    """Standard response format for all agents"""
    model_config = ConfigDict(extra='allow')
    
    agent_name: str
    status: ProcessingStatus = ProcessingStatus.PENDING
    success: bool = False
    confidence_score: float = Field(ge=0.0, le=1.0, default=0.0)
    
    # Results (generic container)
    result: Optional[Union[SchemaValidationResult, DataQualityReport, HarmonizationResult, Dict[str, Any]]] = None
    
    # Execution metadata
    execution_time_seconds: float = 0.0
    llm_calls_made: int = 0
    tokens_used: int = 0
    
    # Error handling
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    
    # Audit trail
    actions_taken: List[str] = Field(default_factory=list)
    
    timestamp: datetime = Field(default_factory=datetime.now)


# =============================================================================
# SUPERVISOR AND AUDIT MODELS
# =============================================================================

class AuditEntry(BaseModel):
    """Single audit log entry"""
    timestamp: datetime = Field(default_factory=datetime.now)
    agent_name: str
    action: str
    status: ProcessingStatus
    confidence_score: Optional[float] = None
    details: Optional[str] = None
    input_summary: Optional[Dict[str, Any]] = None
    output_summary: Optional[Dict[str, Any]] = None
    duration_seconds: float = 0.0


class SupervisorDecision(BaseModel):
    """Decision made by the Supervisor Agent"""
    model_config = ConfigDict(extra='allow')
    
    decision_id: str
    decision_type: str  # proceed, retry, abort, request_review
    reasoning: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    affected_agent: Optional[str] = None
    conditions: List[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)


class PipelineResult(BaseModel):
    """Final result of the complete pipeline"""
    model_config = ConfigDict(extra='allow')
    
    pipeline_id: str
    status: ProcessingStatus = ProcessingStatus.PENDING
    success: bool = False
    
    # Input/Output info
    input_file: str
    output_file: Optional[str] = None
    
    # Agent results
    structural_validation: Optional[AgentResponse] = None
    data_quality: Optional[AgentResponse] = None
    harmonization: Optional[AgentResponse] = None
    
    # Supervisor decisions
    supervisor_decisions: List[SupervisorDecision] = Field(default_factory=list)
    
    # Audit trail
    audit_trail: List[AuditEntry] = Field(default_factory=list)
    
    # Reports
    reports_generated: List[str] = Field(default_factory=list)
    
    # Summary
    total_processing_time_seconds: float = 0.0
    total_llm_calls: int = 0
    total_tokens_used: int = 0
    
    # Final quality metrics
    final_quality_score: float = 0.0
    final_confidence_score: float = 0.0
    
    start_time: datetime = Field(default_factory=datetime.now)
    end_time: Optional[datetime] = None


# =============================================================================
# MASTER METADATA MODELS
# =============================================================================

class MasterColumnDefinition(BaseModel):
    """Definition of a column in the master schema"""
    name: str
    canonical_name: str
    data_type: ColumnDataType
    description: str
    is_required: bool = False
    is_key: bool = False
    aliases: List[str] = Field(default_factory=list)
    validation_rules: Dict[str, Any] = Field(default_factory=dict)
    transformation_hints: Optional[str] = None


class MasterMetadata(BaseModel):
    """Complete master metadata schema"""
    model_config = ConfigDict(extra='allow')
    
    schema_name: str
    version: str
    description: str
    columns: List[MasterColumnDefinition] = Field(default_factory=list)
    business_rules: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


# =============================================================================
# COMPARISON MODELS
# =============================================================================

class BeforeAfterComparison(BaseModel):
    """Before/After comparison of data transformations"""
    model_config = ConfigDict(extra='allow')
    
    column_name: str
    records_compared: int = 0
    records_changed: int = 0
    change_percentage: float = 0.0
    
    sample_changes: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Statistical comparison
    before_stats: Optional[ColumnStatistics] = None
    after_stats: Optional[ColumnStatistics] = None


class DataComparisonReport(BaseModel):
    """Complete before/after comparison report"""
    model_config = ConfigDict(extra='allow')
    
    input_file: str
    output_file: str
    
    # Record-level changes
    total_records_before: int = 0
    total_records_after: int = 0
    records_added: int = 0
    records_removed: int = 0
    
    # Column-level changes
    columns_before: List[str] = Field(default_factory=list)
    columns_after: List[str] = Field(default_factory=list)
    column_comparisons: List[BeforeAfterComparison] = Field(default_factory=list)
    
    # Overall summary
    total_cells_changed: int = 0
    change_percentage: float = 0.0
    
    timestamp: datetime = Field(default_factory=datetime.now)


