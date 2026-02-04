import json
import re
from pathlib import Path

import pandas as pd


def to_snake(value: str) -> str:
    text = re.sub(r"[^0-9a-zA-Z]+", "_", str(value)).strip("_")
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
    return text.lower()


def main() -> None:
    input_path = Path("data/output/harmonized_master.csv")
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    mapping = {col: to_snake(col) for col in df.columns}

    master_schema = {
        "schema_name": "master_from_input",
        "version": "1.0.0",
        "columns": [
            {
                "name": mapping[col],
                "canonical_name": mapping[col],
                "data_type": str(df[col].dtype),
                "aliases": [col]
            }
            for col in df.columns
        ]
    }

    output = {
        "standardized_headers": mapping,
        "master_schema": master_schema
    }

    Path("metadata").mkdir(parents=True, exist_ok=True)
    out_path = Path("metadata/master_columns_from_input.json")
    out_path.write_text(json.dumps(output, indent=2))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
