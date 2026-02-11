"""
LLM Reasoning Agent for the Agentic AI Data Harmonization System.
Provides shared LLM-based reasoning capabilities for all agents.
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from agents.base_agent import BaseAgent
from config import AZURE_CONFIG
from models.schemas import AgentResponse, ProcessingStatus


class LLMReasoningAgent(BaseAgent):
    """
    Shared LLM reasoning utility agent.
    
    Provides common LLM-based analysis capabilities:
    - Column mapping inference
    - Anomaly explanation
    - Ambiguity resolution
    - Data quality analysis
    - Schema inference
    """
    
    PROMPT_TEMPLATES = {
        "column_mapping": """You are a data engineering expert analyzing column mappings between a source dataset and a master schema.

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
}}""",

        "anomaly_explanation": """You are a data quality expert analyzing anomalies in a dataset.

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
}}""",

        "ambiguity_resolution": """You are a data harmonization expert resolving ambiguous data mappings.

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
}}""",

        "data_quality_analysis": """You are a data quality analyst reviewing a dataset summary.

TASK: Analyze the following data quality metrics and provide recommendations.

DATASET SUMMARY:
{dataset_summary}

COLUMN STATISTICS:
{column_statistics}

BUSINESS RULES:
{business_rules}

Analyze:
1. Overall data quality assessment
2. Critical issues that block processing
3. Issues that can be auto-fixed
4. Issues that can be safely ignored
5. Recommendations for improvement

RESPOND IN STRICT JSON FORMAT:
{{
    "overall_quality_score": 85.5,
    "is_acceptable": true,
    "critical_issues": [
        {{
            "issue": "description",
            "column": "column_name",
            "severity": "blocking",
            "recommendation": "what to do"
        }}
    ],
    "fixable_issues": [...],
    "ignorable_issues": [...],
    "recommendations": ["list of recommendations"],
    "summary": "overall summary"
}}""",

        "schema_inference": """You are a data architect inferring schema information from data samples.

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
}}""",

        "transformation_suggestion": """You are a data transformation expert.

TASK: Suggest transformations to convert source data to target format.

SOURCE COLUMN: {source_column}
SOURCE TYPE: {source_type}
SOURCE SAMPLES: {source_samples}

TARGET COLUMN: {target_column}
TARGET TYPE: {target_type}
TARGET CONSTRAINTS: {target_constraints}

Provide transformation logic that:
1. Converts data types appropriately
2. Handles edge cases
3. Preserves data integrity
4. Handles null/missing values

RESPOND IN STRICT JSON FORMAT:
{{
    "transformation_type": "scale_normalize|value_map|type_cast|date_format|custom",
    "transformation_logic": "description of transformation",
    "python_expression": "pandas expression to apply",
    "handles_nulls": true,
        "potential_data_loss": false,
        "confidence": 0.90,
        "warnings": ["any warnings"]
    }}""",

        "agentic_code_improvement": """You are an autonomous data engineer inside an agentic data-cleaning loop.

You NEVER see raw data. You only see:
- Per-column statistics (null %, min, max, mean, top values)
- Overall data-quality summary
- Optional error message from the last code you generated
- Optional learned fixes from previous runs

Your job is to:
- Generate SAFE Python code that operates on a pandas DataFrame named df
- Actually CHANGE df to improve missing-value handling, format issues, and obvious data-quality problems
- Avoid file I/O, network calls, or imports (pd and np are already available)

YOU MUST:
- Always modify df in a meaningful way (no-op code is not allowed)
- Look at "top_values" for each column: if you see date-like values with WRONG separators (e.g. "2024 > 06 > 01" using " > " instead of "-"), fix them to standard format. For survey/data dates use hyphen: YYYY-MM-DD. Example: df["survey_date"] = df["survey_date"].astype(str).str.replace(" > ", "-", regex=False) then parse with pd.to_datetime(..., errors="coerce") and format back to "%Y-%m-%d".
- Fix other format issues (typos, inconsistent casing, wrong delimiters) when evident from top_values.
- Prioritize the top 3–5 columns with the highest null percentage for imputation
- For numeric columns with missing values, use mean/median imputation
- For categorical/string columns with missing values, use a constant like "Unknown"
- Keep transformations simple and explainable

LEARNED FROM PREVIOUS RUNS (use to improve this run):
{learned_format_fixes}

Current iteration: {iteration}

DATAFRAME STATS (before this iteration):
{stats_before}

DATA QUALITY SUMMARY (before this iteration):
{dq_before}

LAST ERROR (if any, may be empty):
{last_error}

RULES:
- Only modify the existing DataFrame df
- Do NOT read or write any files
- Do NOT import any new libraries
- Focus on:
  - fixing date/format issues (wrong separators like " > " in dates -> use "-", then standard YYYY-MM-DD)
  - filling or safely handling missing values
  - fixing obviously invalid numeric ranges (e.g. negative ages)
  - lightweight transformations that are reversible

RESPOND IN STRICT JSON (no comments, no extra keys) with this shape:
{{
  "code": "Python code that assumes df, pd, np exist. Either modify df in-place or assign df = <new_df>.",
  "description": "Short summary of what the code does.",
  "expected_effect": "What quality issues this should improve.",
  "confidence": 0.9
}}""",

        "schema_validation_from_knowledge": """You are a data engineer doing schema validation. You have LEARNED from previous runs (knowledge bag) and see CURRENT data stats. Your job is to suggest column mappings (source → master) and validation issues.

LEARNED FROM PREVIOUS RUNS (knowledge bag):
{knowledge_bag_summary}

MASTER SCHEMA (target columns):
{master_schema_summary}

CURRENT DATA STATS (this file; no raw rows):
{current_stats}

TASK:
1. Suggest column mappings: for each column in current data, map to the best matching master column (or UNMAPPED).
2. Use knowledge bag (learned_mappings, schema_diffs, datasets) to prefer mappings we have seen before.
3. List any validation_errors (blocking) and validation_warnings (fixable).
4. Be concise; confidence 0.0-1.0 per mapping.

RESPOND IN STRICT JSON ONLY (no markdown, no explanation):
{{
  "mappings": [
    {{ "source_column": "name_in_current_data", "target_column": "master_column_or_UNMAPPED", "confidence": 0.95, "reasoning": "brief reason" }}
  ],
  "validation_errors": ["blocking issue 1", "..."],
  "validation_warnings": ["fixable warning 1", "..."],
  "analysis_summary": "1-2 sentences"
}}""",

        "suggest_harmonization_options": """You are a data engineer. Given what we LEARNED from previous runs (knowledge bag) and CURRENT data stats, suggest harmonization options. No raw rows.

LEARNED FROM PREVIOUS RUNS:
{knowledge_bag_summary}

CURRENT DATA STATS (columns, null %, types, top_values):
{current_stats}

TASK: Suggest options for harmonization. Prefer learned imputation_overrides when present.
- column_strategies: for each column with missing values, suggest one of: Replace with MEAN, Replace with MEDIAN, Replace with 0, Replace with MODE, Replace with 'Unknown', Replace with 'Other', Keep missing, Remove rows, Drop column.
- standardize_cols: true/false (normalize column names)
- remove_special: true/false (clean special chars in values)
- date_standardize: true/false (normalize date columns to YYYY-MM-DD)
- country_mapping: true/false (standardize country to codes)
- category_mapping: true/false (standardize channel/type values)
- default_numeric: "Replace with MEAN" or "Replace with MEDIAN" or "Replace with 0"
- default_text: "Replace with 'Unknown'" or "Replace with 'Other'"

RESPOND IN STRICT JSON ONLY:
{{
  "column_strategies": {{ "column_name": "Replace with MEAN" or "Replace with 'Unknown'" etc }},
  "standardize_cols": true,
  "remove_special": true,
  "date_standardize": true,
  "country_mapping": true,
  "category_mapping": true,
  "default_numeric": "Replace with MEAN",
  "default_text": "Replace with 'Unknown'"
}}"""
    }
    
    def __init__(self):
        """Initialize the LLM Reasoning Agent"""
        super().__init__(
            name="LLMReasoningAgent",
            confidence_threshold=0.80,
            max_retries=3
        )
    
    def get_prompt_template(self) -> str:
        """Get the default prompt template"""
        return self.PROMPT_TEMPLATES["column_mapping"]
    
    def execute(self, *args, **kwargs) -> AgentResponse:
        """
        Execute is not directly used - this agent provides utility methods.
        """
        raise NotImplementedError(
            "LLMReasoningAgent provides utility methods, not direct execution. "
            "Use specific methods like infer_column_mappings() instead."
        )
    
    def infer_column_mappings(
        self,
        source_columns: List[Dict[str, Any]],
        master_columns: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], float]:
        """
        Use LLM to infer mappings between source and master columns.
        
        Args:
            source_columns: List of source column info dicts
            master_columns: List of master schema column info dicts
            
        Returns:
            Tuple of (mapping results dict, tokens used)
        """
        self.logger.info("Inferring column mappings with LLM")
        
        prompt = self.PROMPT_TEMPLATES["column_mapping"].format(
            source_columns=json.dumps(source_columns, indent=2),
            master_columns=json.dumps(master_columns, indent=2)
        )
        
        messages = [
            {"role": "system", "content": "You are a data engineering expert. Always respond in valid JSON format."},
            {"role": "user", "content": prompt}
        ]
        
        response, tokens = self.call_llm(messages, purpose="column_mapping_inference")
        
        # Parse JSON response
        result = self._parse_json_response(response)
        
        self.add_audit_entry(
            action="Inferred column mappings",
            status=ProcessingStatus.COMPLETED,
            confidence_score=result.get("overall_confidence", 0),
            details=f"Mapped {len(result.get('mappings', []))} columns"
        )
        
        return result, tokens
    
    def explain_anomalies(
        self,
        column_name: str,
        data_type: str,
        expected_range: str,
        anomalies: List[Dict[str, Any]],
        sample_values: List[Any]
    ) -> Tuple[Dict[str, Any], float]:
        """
        Use LLM to explain detected anomalies.
        
        Args:
            column_name: Name of the column with anomalies
            data_type: Data type of the column
            expected_range: Expected value range
            anomalies: List of detected anomalies
            sample_values: Sample values from the column
            
        Returns:
            Tuple of (analysis result dict, tokens used)
        """
        self.logger.info(f"Analyzing anomalies for column: {column_name}")
        
        prompt = self.PROMPT_TEMPLATES["anomaly_explanation"].format(
            column_name=column_name,
            data_type=data_type,
            expected_range=expected_range,
            anomalies=json.dumps(anomalies, indent=2),
            sample_values=json.dumps(sample_values[:20], default=str)
        )
        
        messages = [
            {"role": "system", "content": "You are a data quality expert. Always respond in valid JSON format."},
            {"role": "user", "content": prompt}
        ]
        
        response, tokens = self.call_llm(messages, purpose="anomaly_explanation")
        
        result = self._parse_json_response(response)
        
        self.add_audit_entry(
            action=f"Explained anomalies for {column_name}",
            status=ProcessingStatus.COMPLETED,
            confidence_score=result.get("anomaly_analysis", [{}])[0].get("confidence", 0) if result.get("anomaly_analysis") else 0
        )
        
        return result, tokens
    
    def resolve_ambiguity(
        self,
        source_column: str,
        sample_values: List[Any],
        candidate_targets: List[Dict[str, Any]],
        context: str = ""
    ) -> Tuple[Dict[str, Any], float]:
        """
        Use LLM to resolve ambiguous column mappings.
        
        Args:
            source_column: Source column name
            sample_values: Sample values from the source column
            candidate_targets: List of potential target column info
            context: Additional context for decision making
            
        Returns:
            Tuple of (resolution result dict, tokens used)
        """
        self.logger.info(f"Resolving ambiguity for column: {source_column}")
        
        prompt = self.PROMPT_TEMPLATES["ambiguity_resolution"].format(
            source_column=source_column,
            sample_values=json.dumps(sample_values[:15], default=str),
            candidate_targets=json.dumps(candidate_targets, indent=2),
            context=context or "No additional context provided"
        )
        
        messages = [
            {"role": "system", "content": "You are a data harmonization expert. Always respond in valid JSON format."},
            {"role": "user", "content": prompt}
        ]
        
        response, tokens = self.call_llm(messages, purpose="ambiguity_resolution")
        
        result = self._parse_json_response(response)
        
        self.add_audit_entry(
            action=f"Resolved ambiguity for {source_column}",
            status=ProcessingStatus.COMPLETED,
            confidence_score=result.get("confidence", 0),
            details=f"Selected target: {result.get('selected_target', 'unknown')}"
        )
        
        return result, tokens
    
    def analyze_data_quality(
        self,
        dataset_summary: Dict[str, Any],
        column_statistics: List[Dict[str, Any]],
        business_rules: List[str]
    ) -> Tuple[Dict[str, Any], float]:
        """
        Use LLM to perform comprehensive data quality analysis.
        
        Args:
            dataset_summary: Summary of the dataset
            column_statistics: Statistics for each column
            business_rules: List of business rules to check
            
        Returns:
            Tuple of (analysis result dict, tokens used)
        """
        self.logger.info("Performing LLM-based data quality analysis")
        
        prompt = self.PROMPT_TEMPLATES["data_quality_analysis"].format(
            dataset_summary=json.dumps(dataset_summary, indent=2),
            column_statistics=json.dumps(column_statistics[:20], indent=2),  # Limit to avoid token overflow
            business_rules=json.dumps(business_rules)
        )
        
        messages = [
            {"role": "system", "content": "You are a data quality analyst. Always respond in valid JSON format."},
            {"role": "user", "content": prompt}
        ]
        
        response, tokens = self.call_llm(messages, purpose="data_quality_analysis")
        
        result = self._parse_json_response(response)
        
        self.add_audit_entry(
            action="Completed LLM data quality analysis",
            status=ProcessingStatus.COMPLETED,
            confidence_score=result.get("overall_quality_score", 0) / 100,
            details=f"Quality score: {result.get('overall_quality_score', 0)}"
        )
        
        return result, tokens
    
    def infer_schema(
        self,
        column_name: str,
        sample_values: List[Any],
        value_counts: Dict[str, int]
    ) -> Tuple[Dict[str, Any], float]:
        """
        Use LLM to infer schema for a column.
        
        Args:
            column_name: Name of the column
            sample_values: Sample values from the column
            value_counts: Value frequency counts
            
        Returns:
            Tuple of (inferred schema dict, tokens used)
        """
        self.logger.info(f"Inferring schema for column: {column_name}")
        
        prompt = self.PROMPT_TEMPLATES["schema_inference"].format(
            column_name=column_name,
            sample_values=json.dumps(sample_values[:30], default=str),
            value_counts=json.dumps(dict(list(value_counts.items())[:20]))
        )
        
        messages = [
            {"role": "system", "content": "You are a data architect. Always respond in valid JSON format."},
            {"role": "user", "content": prompt}
        ]
        
        response, tokens = self.call_llm(messages, purpose="schema_inference")
        
        result = self._parse_json_response(response)
        
        self.add_audit_entry(
            action=f"Inferred schema for {column_name}",
            status=ProcessingStatus.COMPLETED,
            confidence_score=result.get("confidence", 0)
        )
        
        return result, tokens
    
    def suggest_transformation(
        self,
        source_column: str,
        source_type: str,
        source_samples: List[Any],
        target_column: str,
        target_type: str,
        target_constraints: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], float]:
        """
        Use LLM to suggest data transformation logic.
        
        Args:
            source_column: Source column name
            source_type: Source data type
            source_samples: Sample source values
            target_column: Target column name
            target_type: Target data type
            target_constraints: Target constraints
            
        Returns:
            Tuple of (transformation suggestion dict, tokens used)
        """
        self.logger.info(f"Suggesting transformation: {source_column} -> {target_column}")
        
        prompt = self.PROMPT_TEMPLATES["transformation_suggestion"].format(
            source_column=source_column,
            source_type=source_type,
            source_samples=json.dumps(source_samples[:20], default=str),
            target_column=target_column,
            target_type=target_type,
            target_constraints=json.dumps(target_constraints)
        )
        
        messages = [
            {"role": "system", "content": "You are a data transformation expert. Always respond in valid JSON format."},
            {"role": "user", "content": prompt}
        ]
        
        response, tokens = self.call_llm(messages, purpose="transformation_suggestion")
        
        result = self._parse_json_response(response)
        
        self.add_audit_entry(
            action=f"Suggested transformation for {source_column}",
            status=ProcessingStatus.COMPLETED,
            confidence_score=result.get("confidence", 0)
        )
        
        return result, tokens

    def generate_agentic_code(
        self,
        stats_before: Dict[str, Any],
        dq_before: Dict[str, Any],
        last_error: Optional[str] = None,
        iteration: int = 1,
        learned_format_fixes: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[Dict[str, Any], float]:
        """
        Ask the LLM to generate concrete Python code that operates on df
        to improve data quality, based on summary statistics, previous
        error (if any), and learned format fixes from past runs.
        """
        self.logger.info(f"Generating agentic code for iteration {iteration}")

        learned_text: str
        if learned_format_fixes:
            lines = [
                f"- Column '{e.get('column', '?')}': {e.get('problem', '')} -> fix: {e.get('fix', '')}"
                for e in learned_format_fixes[-15:]
            ]
            learned_text = "\n".join(lines) if lines else "None yet."
        else:
            learned_text = "None yet."

        prompt = self.PROMPT_TEMPLATES["agentic_code_improvement"].format(
            iteration=iteration,
            learned_format_fixes=learned_text,
            stats_before=json.dumps(stats_before, indent=2),
            dq_before=json.dumps(dq_before, indent=2),
            last_error=json.dumps(last_error or "", ensure_ascii=False),
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a senior data engineer. "
                    "Always respond with STRICT, VALID JSON exactly matching the requested schema. "
                    "Never include comments or markdown code fences."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        response, tokens = self.call_llm(messages, purpose="agentic_code_generation")
        result = self._parse_json_response(response)

        self.add_audit_entry(
            action=f"Generated agentic code (iteration {iteration})",
            status=ProcessingStatus.COMPLETED,
            confidence_score=result.get("confidence", 0),
            details=result.get("description", "")[:300],
        )

        return result, tokens

    def schema_validation_from_knowledge(
        self,
        knowledge_bag: Dict[str, Any],
        current_stats: Dict[str, Any],
        master_schema: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], int]:
        """
        AI-driven schema validation using knowledge bag (learned from past runs) and current data stats.
        Returns (result with mappings, validation_errors, validation_warnings), tokens_used.
        """
        self.logger.info("Running AI-driven schema validation from knowledge bag and current stats")

        # Compact summary of knowledge bag (no huge payloads)
        kb = knowledge_bag or {}
        datasets = kb.get("datasets") or {}
        learned = kb.get("learned_mappings") or []
        schema_diffs = kb.get("schema_diffs") or []
        knowledge_bag_summary = {
            "datasets_seen": list(datasets.keys()),
            "dataset_columns": {k: v.get("columns", [])[:15] for k, v in list(datasets.items())[:5]},
            "learned_mappings_recent": [
                {"source": m.get("source_column"), "target": m.get("target_column"), "confidence": m.get("confidence")}
                for m in learned[-30:]
            ],
            "schema_diffs_recent": [{"dataset": d.get("dataset"), "missing_in_source": d.get("missing_in_source", [])[:5], "extra_in_source": d.get("extra_in_source", [])[:5]} for d in schema_diffs[-10:]],
        }

        # Master schema: list of column names + types
        master_cols = master_schema.get("columns") or []
        master_schema_summary = [
            {"name": c.get("name"), "data_type": c.get("data_type"), "aliases": (c.get("aliases") or [])[:5]}
            for c in (master_cols if isinstance(master_cols, list) else list(master_cols))[:50]
        ]
        if not master_schema_summary and isinstance(master_schema.get("columns"), dict):
            master_schema_summary = [{"name": k, "data_type": "unknown"} for k in list(master_schema["columns"].keys())[:50]]

        prompt = self.PROMPT_TEMPLATES["schema_validation_from_knowledge"].format(
            knowledge_bag_summary=json.dumps(knowledge_bag_summary, indent=2),
            master_schema_summary=json.dumps(master_schema_summary, indent=2),
            current_stats=json.dumps(current_stats, indent=2),
        )
        messages = [
            {"role": "system", "content": "You are a data engineer. Respond with valid JSON only. No markdown, no explanation."},
            {"role": "user", "content": prompt},
        ]
        response, tokens = self.call_llm(messages, purpose="schema_validation_from_knowledge")
        result = self._parse_json_response(response)

        # Normalize to shape app expects: column_mappings with source_column, target_column, confidence, reasoning
        mappings = result.get("mappings") or []
        column_mappings = []
        for m in mappings:
            column_mappings.append({
                "source_column": m.get("source_column") or m.get("source"),
                "target_column": m.get("target_column") or m.get("target") or "UNMAPPED",
                "confidence": float(m.get("confidence", 0.8)),
                "reasoning": m.get("reasoning") or "",
            })
        out = {
            "column_mappings": column_mappings,
            "validation_errors": result.get("validation_errors") or [],
            "validation_warnings": result.get("validation_warnings") or [],
            "analysis_summary": result.get("analysis_summary") or "",
        }
        self.add_audit_entry(
            action="Schema validation from knowledge",
            status=ProcessingStatus.COMPLETED,
            confidence_score=0.85,
            details=f"{len(column_mappings)} mappings",
        )
        return out, tokens

    def suggest_harmonization_options(
        self,
        knowledge_bag: Dict[str, Any],
        current_stats: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], int]:
        """
        AI-driven harmonization options from knowledge bag + current data stats.
        Returns (options dict to merge into harmonization_options, tokens_used).
        """
        self.logger.info("Suggesting harmonization options from knowledge bag and current stats")
        kb = knowledge_bag or {}
        imputation = kb.get("imputation_overrides") or []
        learned = kb.get("learned_mappings") or []
        knowledge_bag_summary = {
            "imputation_overrides_recent": [{"column": e.get("column"), "strategy": e.get("strategy")} for e in imputation[-20:]],
            "learned_mappings_count": len(learned),
        }
        prompt = self.PROMPT_TEMPLATES["suggest_harmonization_options"].format(
            knowledge_bag_summary=json.dumps(knowledge_bag_summary, indent=2),
            current_stats=json.dumps(current_stats, indent=2),
        )
        messages = [
            {"role": "system", "content": "You are a data engineer. Respond with valid JSON only. No markdown."},
            {"role": "user", "content": prompt},
        ]
        response, tokens = self.call_llm(messages, purpose="suggest_harmonization_options")
        result = self._parse_json_response(response)
        options = {
            "column_strategies": result.get("column_strategies") or {},
            "standardize_cols": result.get("standardize_cols", True),
            "remove_special": result.get("remove_special", True),
            "date_standardize": result.get("date_standardize", True),
            "country_mapping": result.get("country_mapping", True),
            "category_mapping": result.get("category_mapping", True),
            "default_numeric": result.get("default_numeric") or "Replace with MEAN",
            "default_text": result.get("default_text") or "Replace with 'Unknown'",
        }
        self.add_audit_entry(
            action="Suggested harmonization options",
            status=ProcessingStatus.COMPLETED,
            confidence_score=0.85,
        )
        return options, tokens
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """
        Parse JSON from LLM response, handling common issues.
        
        Args:
            response: Raw LLM response string
            
        Returns:
            Parsed dictionary
        """
        try:
            # Try direct parsing first
            return json.loads(response)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code blocks
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
            if json_match:
                try:
                    return json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    pass
            
            # Try to find JSON object in the response
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass
            
            self.logger.warning("Failed to parse LLM response as JSON, returning raw response")
            return {"raw_response": response, "parse_error": True}
    
    def custom_reasoning(
        self,
        prompt: str,
        system_message: str = "You are an AI assistant helping with data analysis."
    ) -> Tuple[str, int]:
        """
        Perform custom LLM reasoning with a specific prompt.
        
        Args:
            prompt: The user prompt
            system_message: The system message
            
        Returns:
            Tuple of (response content, tokens used)
        """
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]
        
        return self.call_llm(messages, purpose="custom_reasoning")


# Singleton instance for shared usage
_llm_agent_instance: Optional[LLMReasoningAgent] = None


def get_llm_reasoning_agent() -> LLMReasoningAgent:
    """
    Get the singleton LLM Reasoning Agent instance.
    
    Returns:
        LLMReasoningAgent instance
    """
    global _llm_agent_instance
    if _llm_agent_instance is None:
        _llm_agent_instance = LLMReasoningAgent()
    return _llm_agent_instance


