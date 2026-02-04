"""
Prompt Templates for the Agentic AI Data Harmonization System.
All LLM prompts are centralized here for easy maintenance and versioning.
"""

# =============================================================================
# STRUCTURAL VALIDATION PROMPTS
# =============================================================================

STRUCTURAL_VALIDATION_PROMPT = """You are an expert data engineer specializing in schema validation and data harmonization.

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


COLUMN_MAPPING_PROMPT = """You are a data engineering expert analyzing column mappings between a source dataset and a master schema.

TASK: Analyze the following columns and determine the best mappings.

SOURCE COLUMNS:
{source_columns}

MASTER SCHEMA COLUMNS:
{master_columns}

For each source column, determine:
1. The most likely matching master column (or "UNMAPPED" if no match)
2. Confidence score (0.0 to 1.0)
3. Reasoning for the mapping
4. Any transformation needed

RESPOND IN STRICT JSON FORMAT:
{{
    "mappings": [
        {{
            "source_column": "source_col_name",
            "target_column": "target_col_name or UNMAPPED",
            "confidence": 0.95,
            "reasoning": "explanation",
            "transformation": "description of transformation needed or null"
        }}
    ],
    "unmapped_source": ["list of source columns with no match"],
    "missing_required": ["list of required master columns not found in source"],
    "overall_confidence": 0.85,
    "analysis_summary": "overall analysis"
}}"""


# =============================================================================
# DATA QUALITY PROMPTS
# =============================================================================

DATA_QUALITY_PROMPT = """You are a senior data quality analyst with expertise in survey data and business intelligence.

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


ANOMALY_EXPLANATION_PROMPT = """You are a data quality expert analyzing anomalies in a dataset.

TASK: Analyze the following data anomalies and provide explanations.

COLUMN: {column_name}
DATA TYPE: {data_type}
EXPECTED RANGE: {expected_range}

ANOMALIES DETECTED:
{anomalies}

SAMPLE VALUES:
{sample_values}

For each anomaly type, provide:
1. Likely root cause
2. Business impact assessment
3. Recommended action (fix, flag, or ignore)
4. Confidence in the assessment

RESPOND IN STRICT JSON FORMAT:
{{
    "anomaly_analysis": [
        {{
            "anomaly_type": "type of anomaly",
            "root_cause": "likely cause",
            "business_impact": "impact description",
            "recommended_action": "fix|flag|ignore",
            "fix_suggestion": "how to fix if applicable",
            "confidence": 0.85
        }}
    ],
    "overall_severity": "blocking|fixable|ignorable",
    "summary": "overall analysis summary"
}}"""


# =============================================================================
# HARMONIZATION PROMPTS
# =============================================================================

HARMONIZATION_PROMPT = """You are an expert data engineer specializing in data transformation and harmonization.

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


AMBIGUITY_RESOLUTION_PROMPT = """You are a data harmonization expert resolving ambiguous data mappings.

TASK: Resolve the following ambiguous mapping situation.

SOURCE COLUMN: {source_column}
SAMPLE VALUES: {sample_values}

CANDIDATE TARGETS:
{candidate_targets}

CONTEXT:
{context}

Determine the best target mapping considering:
1. Semantic similarity
2. Data type compatibility
3. Value patterns
4. Business context

RESPOND IN STRICT JSON FORMAT:
{{
    "selected_target": "best_target_column",
    "confidence": 0.85,
    "reasoning": "detailed explanation",
    "alternative_targets": [
        {{
            "target": "alternative_column",
            "confidence": 0.65,
            "why_not_selected": "reason"
        }}
    ],
    "transformation_needed": "description or null",
    "warnings": ["any warnings about this mapping"]
}}"""


# =============================================================================
# SUPERVISOR PROMPTS
# =============================================================================

SUPERVISOR_PROMPT = """You are a senior data pipeline supervisor responsible for orchestrating a data harmonization pipeline.

TASK: Evaluate the current pipeline state and make a decision on how to proceed.

PIPELINE STATE:
{pipeline_state}

AGENT RESULTS:
{agent_results}

DECISION REQUIRED: {decision_type}

Based on the results:
1. Evaluate if the pipeline should proceed, retry, or abort
2. If retrying, specify which agent and what parameters to adjust
3. If proceeding, note any warnings or conditions
4. If aborting, explain the reasoning

Consider:
- Confidence thresholds (minimum {confidence_threshold})
- Blocking issues vs fixable issues
- Data quality acceptability
- Business impact of decisions

RESPOND IN STRICT JSON FORMAT:
{{
    "decision": "proceed|retry|abort|request_review",
    "reasoning": "detailed explanation of the decision",
    "confidence": 0.95,
    "affected_agent": "agent name if retry",
    "retry_parameters": {{}},
    "conditions": ["list of conditions or warnings"],
    "next_steps": ["ordered list of next steps"],
    "risk_assessment": "low|medium|high",
    "recommendations": ["list of recommendations"]
}}"""


# =============================================================================
# SCHEMA INFERENCE PROMPTS
# =============================================================================

SCHEMA_INFERENCE_PROMPT = """You are a data architect inferring schema information from data samples.

TASK: Analyze the following data samples and infer the schema.

COLUMN: {column_name}
SAMPLE VALUES: {sample_values}
VALUE COUNTS: {value_counts}

Determine:
1. Most likely data type
2. Whether it's categorical or continuous
3. Probable constraints (min, max, allowed values)
4. Suggested canonical name
5. Any data quality concerns

RESPOND IN STRICT JSON FORMAT:
{{
    "inferred_type": "string|integer|float|boolean|datetime|categorical",
    "is_categorical": true,
    "suggested_canonical_name": "snake_case_name",
    "constraints": {{
        "min_value": null,
        "max_value": null,
        "allowed_values": ["list"] or null,
        "pattern": "regex pattern or null"
    }},
    "quality_concerns": ["list of concerns"],
    "confidence": 0.90,
    "reasoning": "explanation"
}}"""


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_prompt_template(prompt_name: str) -> str:
    """
    Get a prompt template by name.
    
    Args:
        prompt_name: Name of the prompt template
        
    Returns:
        Prompt template string
    """
    templates = {
        "structural_validation": STRUCTURAL_VALIDATION_PROMPT,
        "column_mapping": COLUMN_MAPPING_PROMPT,
        "data_quality": DATA_QUALITY_PROMPT,
        "anomaly_explanation": ANOMALY_EXPLANATION_PROMPT,
        "harmonization": HARMONIZATION_PROMPT,
        "ambiguity_resolution": AMBIGUITY_RESOLUTION_PROMPT,
        "supervisor": SUPERVISOR_PROMPT,
        "schema_inference": SCHEMA_INFERENCE_PROMPT
    }
    
    return templates.get(prompt_name, "")


def format_prompt(template: str, **kwargs) -> str:
    """
    Format a prompt template with provided values.
    
    Args:
        template: Prompt template string
        **kwargs: Values to substitute
        
    Returns:
        Formatted prompt string
    """
    try:
        return template.format(**kwargs)
    except KeyError as e:
        raise ValueError(f"Missing required template variable: {e}")


