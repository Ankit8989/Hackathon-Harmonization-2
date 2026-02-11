"""
Agentic data-cleaning loop.

The LLM:
- Sees only aggregate statistics (no raw rows)
- Generates Python code to operate on `df`
- Code is executed locally on the DataFrame
- Results are re-validated with DataQualityAgent
- Loop repeats for a small number of iterations
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from agents.data_quality_agent import DataQualityAgent
from agents.llm_reasoning_agent import get_llm_reasoning_agent
from config import METADATA_DIR
from utils.knowledge_bag import (
    add_learned_format_fix,
    load_knowledge_bag,
    save_knowledge_bag,
)


def _summarize_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """Build compact, aggregate-only statistics for the LLM."""
    summary: Dict[str, Any] = {
        "total_rows": int(len(df)),
        "total_columns": int(len(df.columns)),
        "columns": {},
    }

    for col in df.columns:
        series = df[col]
        # In rare cases (e.g. MultiIndex columns), df[col] can be a DataFrame.
        # Fall back to the first sub-column so downstream logic always sees a Series.
        if isinstance(series, pd.DataFrame):
            if series.shape[1] == 0:
                continue
            series = series.iloc[:, 0]
        col_info: Dict[str, Any] = {
            "dtype": str(series.dtype),
            "non_null_count": int(series.notna().sum()),
            "null_count": int(series.isna().sum()),
            "null_percentage": float((series.isna().mean() * 100.0) if len(series) > 0 else 0.0),
        }

        if pd.api.types.is_numeric_dtype(series):
            numeric = pd.to_numeric(series, errors="coerce")
            if numeric.notna().any():
                col_info["min"] = float(numeric.min())
                col_info["max"] = float(numeric.max())
                col_info["mean"] = float(numeric.mean())

        # Top value frequencies (stringified)
        value_counts = series.value_counts(dropna=True).head(5).to_dict()
        col_info["top_values"] = {str(k): int(v) for k, v in value_counts.items()}

        summary["columns"][str(col)] = col_info

    return summary


def _build_dq_summary(dq_result: Dict[str, Any]) -> Dict[str, Any]:
    """Extract a small, stable summary from a DataQualityReport-like dict."""
    if dq_result is None:
        return {
            "overall_quality_score": 0.0,
            "is_acceptable": False,
            "total_issues": 0,
        }

    score = float(dq_result.get("overall_quality_score", 0.0) or 0.0)
    is_ok = bool(dq_result.get("is_acceptable", False))

    blocking = len(dq_result.get("blocking_issues", []) or [])
    fixable = len(dq_result.get("fixable_issues", []) or [])
    ignorable = len(dq_result.get("ignorable_issues", []) or [])

    return {
        "overall_quality_score": score,
        "is_acceptable": is_ok,
        "blocking_issues": int(blocking),
        "fixable_issues": int(fixable),
        "ignorable_issues": int(ignorable),
        "total_issues": int(blocking + fixable + ignorable),
    }


def _looks_like_format_fix(description: str, code: str) -> bool:
    """Heuristic: did this iteration fix a date/format issue (e.g. separator ' > ' -> '-')?"""
    d = (description or "").lower()
    c = (code or "").lower()
    if "date" in d or "format" in d or "separator" in d:
        return True
    if "replace" in c and (" > " in (code or "") or "'>'" in (code or "") or '">"' in (code or "")):
        return True
    if "strftime" in c or "to_datetime" in c:
        return True
    return False


def _column_mentioned_in_code(code: str) -> Optional[str]:
    """Extract first column name from code like df['col'] or df[\"col\"]."""
    if not code:
        return None
    m = re.search(r"df\s*\[\s*['\"]([^'\"]+)['\"]\s*\]", code)
    return m.group(1) if m else None


def _run_data_quality(
    df: pd.DataFrame,
    master_schema: Optional[Dict[str, Any]] = None,
    business_rules: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run DataQualityAgent and return a plain dict result."""
    dq_agent = DataQualityAgent()
    resp = dq_agent.execute(
        df=df,
        master_schema=master_schema or {},
        business_rules=business_rules or [],
        column_mappings=None,
    )

    # resp.result may be a Pydantic model; normalize to dict
    if resp.result is None:
        return {}
    if hasattr(resp.result, "model_dump"):
        return resp.result.model_dump()
    if isinstance(resp.result, dict):
        return resp.result
    return {}


def run_agentic_loop(
    df: pd.DataFrame,
    master_schema: Optional[Dict[str, Any]] = None,
    business_rules: Optional[List[str]] = None,
    max_iterations: int = 3,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]], Dict[str, Any]]:
    """
    Run an agentic improvement loop on a DataFrame.

    Returns:
        cleaned_df: The final DataFrame after all iterations
        history:    List of per-iteration logs (code, scores, errors)
        final_dq:   Final data-quality report dict
    """
    if df is None or df.empty:
        return df, [], {}

    llm_agent = get_llm_reasoning_agent()
    knowledge_bag_path: Path = METADATA_DIR / "knowledge_bag.yaml"
    knowledge_bag = load_knowledge_bag(knowledge_bag_path)
    learned_format_fixes: List[Dict[str, Any]] = knowledge_bag.get("learned_format_fixes") or []

    # Initial quality evaluation
    dq_dict = _run_data_quality(df, master_schema, business_rules)
    dq_summary = _build_dq_summary(dq_dict)
    prev_score = dq_summary.get("overall_quality_score", 0.0)

    history: List[Dict[str, Any]] = []
    last_error: Optional[str] = None

    working_df = df.copy()
    # Track original missing values so we can decide whether to apply
    # a deterministic fallback clean-up if the LLM never manages to improve.
    original_missing = int(df.isna().sum().sum())

    for iteration in range(1, max_iterations + 1):
        stats_before = _summarize_dataframe(working_df)
        dq_before_summary = _build_dq_summary(dq_dict)

        # Ask LLM for code. If the model is at capacity (429) or any other
        # API-level error occurs, we capture it and stop the loop gracefully
        # instead of crashing the whole app.
        plan: Dict[str, Any] = {}
        llm_call_error: Optional[str] = None
        try:
            plan, _ = llm_agent.generate_agentic_code(
                stats_before=stats_before,
                dq_before=dq_before_summary,
                last_error=last_error,
                iteration=iteration,
                learned_format_fixes=learned_format_fixes,
            )
        except Exception as exc:  # pragma: no cover - defensive: includes 429s
            llm_call_error = str(exc)

        code = plan.get("code", "")
        description = plan.get("description", "")
        expected_effect = plan.get("expected_effect", "")
        confidence = plan.get("confidence", 0.0)

        exec_error: Optional[str] = None

        if llm_call_error is not None:
            # We couldn't even get code (e.g. 429 / NoCapacity). Preserve the
            # original dataframe and record the error for the UI.
            exec_error = f"LLM call failed (agentic loop stopped): {llm_call_error}"
        elif not code or not isinstance(code, str):
            exec_error = "LLM did not return valid code."
        else:
            # Execute code in a restricted namespace
            local_ns: Dict[str, Any] = {
                "df": working_df,
                "pd": pd,
                "np": np,
            }
            try:
                exec(code, {}, local_ns)
                # Expect df to be modified in-place or reassigned
                new_df = local_ns.get("df", working_df)
                if not isinstance(new_df, pd.DataFrame):
                    exec_error = "Generated code did not leave a valid DataFrame in variable 'df'."
                else:
                    working_df = new_df
            except Exception as exc:  # pragma: no cover - defensive runtime protection
                exec_error = str(exc)

        # Re-evaluate quality after execution attempt
        dq_dict = _run_data_quality(working_df, master_schema, business_rules)
        dq_after_summary = _build_dq_summary(dq_dict)
        new_score = dq_after_summary.get("overall_quality_score", prev_score)
        improvement = new_score - prev_score

        iteration_log: Dict[str, Any] = {
            "iteration": iteration,
            "code": code,
            "description": description,
            "expected_effect": expected_effect,
            "confidence": confidence,
            "exec_error": exec_error,
            "quality_before": dq_before_summary,
            "quality_after": dq_after_summary,
            "improvement": improvement,
        }
        history.append(iteration_log)

        # If AI fixed a format/date issue successfully, store it for future runs
        if exec_error is None and _looks_like_format_fix(description, code):
            col = _column_mentioned_in_code(code) or "unknown"
            add_learned_format_fix(
                knowledge_bag,
                column=col,
                problem_description="Date/format with wrong separator (e.g. ' > ' instead of '-')",
                fix_description=description or "Normalize to standard format (e.g. YYYY-MM-DD)",
                sample_before=None,
                sample_after=None,
            )
            save_knowledge_bag(knowledge_bag_path, knowledge_bag)
            learned_format_fixes = knowledge_bag.get("learned_format_fixes") or []

        # Update loop state
        last_error = exec_error
        prev_score = new_score

        # Stopping conditions:
        # - LLM call failed (e.g. 429 / capacity) → stop immediately
        # - repeated failures to execute
        # - negligible improvement once score is reasonably high
        if llm_call_error is not None:
            break
        if exec_error is not None and iteration >= 2:
            break
        if improvement < 0.5 and new_score >= 80:
            break

    # ------------------------------------------------------------------
    # Fallback: if the LLM could not run or did not improve missingness,
    # apply a simple rule-based imputation so the "cleaned" data is
    # visibly different and safer to use downstream.
    # ------------------------------------------------------------------
    final_missing = int(working_df.isna().sum().sum())
    if original_missing > 0 and final_missing >= original_missing:
        fallback_log: Dict[str, Any] = {
            "iteration": len(history) + 1,
            "code": "# Fallback rule-based cleaning applied: "
                    "numeric -> mean, non-numeric -> 'Unknown'",
            "description": "Fallback imputation because LLM could not improve data.",
            "expected_effect": "Reduce missing values using simple rules.",
            "confidence": 1.0,
            "exec_error": None,
            "quality_before": _build_dq_summary(dq_dict),
        }

        # Numeric: fill NaNs with column mean; Non-numeric: fill NaNs with 'Unknown'
        numeric_cols = working_df.select_dtypes(include=["number"]).columns
        for col in numeric_cols:
            series = working_df[col]
            # Handle rare case where df[col] is a DataFrame (e.g. MultiIndex); use first sub-column
            if isinstance(series, pd.DataFrame):
                if series.shape[1] == 0:
                    continue
                series = series.iloc[:, 0]
            if series.isna().any():
                mean_val = series.mean()
                working_df[col] = series.fillna(mean_val)

        non_numeric_cols = working_df.columns.difference(numeric_cols)
        for col in non_numeric_cols:
            series = working_df[col]
            if isinstance(series, pd.DataFrame):
                if series.shape[1] == 0:
                    continue
                series = series.iloc[:, 0]
            if series.isna().any():
                working_df[col] = series.fillna("Unknown")

        # Recompute quality after fallback
        dq_dict = _run_data_quality(working_df, master_schema, business_rules)
        dq_after_summary = _build_dq_summary(dq_dict)
        fallback_log["quality_after"] = dq_after_summary
        fallback_log["improvement"] = (
            dq_after_summary.get("overall_quality_score", 0.0)
            - fallback_log["quality_before"].get("overall_quality_score", 0.0)
        )
        history.append(fallback_log)

    # Ensure column names are unique so downstream display (pyarrow/Streamlit)
    # does not fail on duplicates created by AI-generated code or mappings.
    if working_df is not None and not working_df.empty:
        cols = list(working_df.columns)
        seen: Dict[str, int] = {}
        new_cols = []
        for c in cols:
            if c in seen:
                seen[c] += 1
                new_cols.append(f"{c}_dup{seen[c]}")
            else:
                seen[c] = 0
                new_cols.append(c)
        working_df.columns = new_cols

    return working_df, history, dq_dict

