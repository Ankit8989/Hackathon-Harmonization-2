import pandas as pd
import numpy as np
from pathlib import Path
import yaml


def main() -> None:
    base = Path("data/input")
    base.mkdir(parents=True, exist_ok=True)
    metadata_dir = Path("metadata")
    metadata_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(7)

    rows = 200
    hist = pd.DataFrame({
        "resp_id": range(1000, 1000 + rows),
        "interview_date": pd.date_range("2023-01-01", periods=rows, freq="D").strftime("%Y-%m-%d"),
        "market": np.random.choice(
            ["US", "UK", "DE", "FR", "ES", "IT", "NL", "BE", "JP", "AU", "CA", "BR", "IN", "CN"],
            rows
        ),
        "age_group": np.random.choice(["18-24", "25-34", "35-44", "45-54", "55+"], rows),
        "gender": np.random.choice(["Male", "Female", "Other"], rows),
        "csat": np.random.randint(1, 6, rows),
        "nps": np.random.randint(0, 11, rows),
        "purchase_likelihood": np.random.choice(["VU", "U", "N", "L", "VL"], rows),
        "channel": np.random.choice(["ONL", "RET", "MOB", "SOC"], rows)
    })

    rows = 180
    curr = pd.DataFrame({
        "respondent_id": range(2000, 2000 + rows),
        "survey_date": pd.date_range("2024-06-01", periods=rows, freq="D").strftime("%Y-%m-%d"),
        "country": np.random.choice(
            ["USA", "UK", "Germany", "France", "Spain", "Italy", "Netherlands", "Belgium",
             "Japan", "Australia", "Canada", "Brazil", "India", "China"],
            rows
        ),
        "age": np.random.choice(
            ["18 to 24", "25 to 34", "35 to 44", "45 to 54", "55 plus"],
            rows
        ),
        "gender": np.random.choice(["M", "F", "Other"], rows),
        "satisfaction_score": np.random.randint(0, 101, rows),
        "nps_score": np.random.randint(0, 11, rows),
        "purchase_intent": np.random.choice(
            ["Very Unlikely", "Unlikely", "Neutral", "Likely", "Very Likely"], rows
        ),
        "channel": np.random.choice(["Online", "Retail", "Mobile", "Social"], rows)
    })

    rows = 40
    incr = pd.DataFrame({
        "respondent_id": range(3000, 3000 + rows),
        "survey_date": pd.date_range("2024-10-01", periods=rows, freq="D").strftime("%Y-%m-%d"),
        "country": np.random.choice(["USA", "UK", "Germany", "France", "Japan"], rows),
        "age": np.random.choice(["18-24", "25-34", "35-44", "45-54", "55+"], rows),
        "gender": np.random.choice(["Male", "Female", "Other"], rows),
        "satisfaction_score": np.random.randint(1, 11, rows),
        "nps_score": np.random.randint(0, 11, rows),
        "purchase_intent": np.random.choice(
            ["Very Unlikely", "Unlikely", "Neutral", "Likely", "Very Likely"], rows
        ),
        "channel": np.random.choice(["Online", "Retail", "Mobile", "Social"], rows)
    })

    hist.to_csv(base / "sample_historical.csv", index=False)
    curr.to_csv(base / "sample_current.csv", index=False)
    incr.to_csv(base / "sample_incremental.csv", index=False)

    meta = {
        "schema_name": "sample_master_metadata",
        "version": "1.0.0",
        "description": "Sample master metadata for multi-source harmonization",
        "columns": [
            {"name": "respondent_id", "canonical_name": "respondent_id", "data_type": "string",
             "aliases": ["resp_id", "id", "record_id"]},
            {"name": "survey_date", "canonical_name": "survey_date", "data_type": "datetime",
             "aliases": ["interview_date", "date"]},
            {"name": "country", "canonical_name": "country", "data_type": "categorical",
             "aliases": ["market"]},
            {"name": "age", "canonical_name": "age", "data_type": "categorical",
             "aliases": ["age_group"]},
            {"name": "gender", "canonical_name": "gender", "data_type": "categorical",
             "aliases": []},
            {"name": "satisfaction_score", "canonical_name": "satisfaction_score", "data_type": "float",
             "validation_rules": {"min_value": 0, "max_value": 100}, "aliases": ["csat"]},
            {"name": "nps_score", "canonical_name": "nps_score", "data_type": "integer",
             "validation_rules": {"min_value": 0, "max_value": 10}, "aliases": ["nps"]},
            {"name": "purchase_intent", "canonical_name": "purchase_intent", "data_type": "categorical",
             "aliases": ["purchase_likelihood"]},
            {"name": "channel", "canonical_name": "channel", "data_type": "categorical",
             "aliases": []}
        ]
    }
    with open(metadata_dir / "sample_master_metadata.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(meta, f, sort_keys=False)

    mapping_table = {
        "channel": {
            "ONL": "Online",
            "RET": "Retail",
            "MOB": "Mobile",
            "SOC": "Social"
        },
        "purchase_intent": {
            "VU": "Very Unlikely",
            "U": "Unlikely",
            "N": "Neutral",
            "L": "Likely",
            "VL": "Very Likely"
        }
    }
    with open(metadata_dir / "sample_mapping_table.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(mapping_table, f, sort_keys=False)

    validation_rules = {
        "rules": [
            {"min_records": 30},
            {"column": "satisfaction_score", "min": 0, "max": 100},
            {"column": "nps_score", "min": 0, "max": 10},
            {"column": "age", "allowed_values": [
                "18-24", "25-34", "35-44", "45-54", "55+",
                "18 to 24", "25 to 34", "35 to 44", "45 to 54", "55 plus"
            ]},
            {"column": "country", "allowed_values": [
                "USA", "UK", "Germany", "France", "Spain", "Italy", "Netherlands", "Belgium",
                "Japan", "Australia", "Canada", "Brazil", "India", "China"
            ]}
        ]
    }
    with open(metadata_dir / "sample_validation_rules.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(validation_rules, f, sort_keys=False)

    baseline = pd.DataFrame({
        "dataset": ["baseline"] * 3,
        "column": ["satisfaction_score", "nps_score", "purchase_intent"],
        "mean": [75.2, 6.4, None],
        "median": [78, 7, None],
        "mode": [80, 8, "Likely"],
        "min": [0, 0, None],
        "max": [100, 10, None],
        "std": [12.5, 2.1, None],
        "outlier_count": [2, 1, None],
        "missing_pct": [1.2, 0.8, 0.5],
        "unique_count": [101, 11, 5]
    })
    baseline.to_csv(base / "sample_descriptive_stats.csv", index=False)

    print("Sample files created:")
    print(" data/input/sample_historical.csv")
    print(" data/input/sample_current.csv")
    print(" data/input/sample_incremental.csv")
    print(" data/input/sample_descriptive_stats.csv")
    print(" metadata/sample_master_metadata.yaml")
    print(" metadata/sample_mapping_table.yaml")
    print(" metadata/sample_validation_rules.yaml")


if __name__ == "__main__":
    main()
