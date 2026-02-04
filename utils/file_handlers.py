"""
File handling utilities for the Agentic AI Data Harmonization System.
Supports CSV, Excel, SPSS, and JSON file formats.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import numpy as np
import yaml

from utils.logger import get_logger

logger = get_logger("FileHandler")


class FileHandler:
    """
    Unified file handler for multiple data formats.
    Supports reading and writing CSV, Excel, SPSS, and JSON files.
    """
    
    SUPPORTED_EXTENSIONS = {
        '.csv': 'csv',
        '.xlsx': 'excel',
        '.xls': 'excel',
        '.sav': 'spss',
        '.json': 'json'
    }
    
    def __init__(self):
        self._spss_available = self._check_spss_support()
    
    def _check_spss_support(self) -> bool:
        """Check if pyreadstat is available for SPSS support"""
        try:
            import pyreadstat
            return True
        except ImportError:
            logger.warning("pyreadstat not installed. SPSS file support disabled.")
            return False
    
    def read_file(
        self,
        file_path: Union[str, Path],
        **kwargs
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Read a data file and return DataFrame with metadata.
        
        Args:
            file_path: Path to the file
            **kwargs: Additional arguments passed to the reader
            
        Returns:
            Tuple of (DataFrame, metadata dict)
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        extension = file_path.suffix.lower()
        
        if extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file format: {extension}")
        
        file_type = self.SUPPORTED_EXTENSIONS[extension]
        logger.info(f"Reading {file_type.upper()} file: {file_path.name}")
        
        metadata = {
            "file_path": str(file_path),
            "file_name": file_path.name,
            "file_type": file_type,
            "file_size_bytes": file_path.stat().st_size
        }
        
        try:
            if file_type == 'csv':
                df, meta = self._read_csv(file_path, **kwargs)
            elif file_type == 'excel':
                df, meta = self._read_excel(file_path, **kwargs)
            elif file_type == 'spss':
                df, meta = self._read_spss(file_path, **kwargs)
            elif file_type == 'json':
                df, meta = self._read_json(file_path, **kwargs)
            else:
                raise ValueError(f"Unknown file type: {file_type}")
            
            metadata.update(meta)
            metadata["rows"] = len(df)
            metadata["columns"] = list(df.columns)
            metadata["column_count"] = len(df.columns)
            
            logger.info(f"Successfully loaded {len(df)} rows, {len(df.columns)} columns")
            
            return df, metadata
            
        except Exception as e:
            logger.error(f"Error reading file: {str(e)}")
            raise
    
    def _read_csv(
        self,
        file_path: Path,
        encoding: str = 'utf-8',
        **kwargs
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Read CSV file"""
        # Try different encodings if default fails
        encodings_to_try = [encoding, 'latin-1', 'cp1252', 'iso-8859-1']
        
        for enc in encodings_to_try:
            try:
                df = pd.read_csv(file_path, encoding=enc, **kwargs)
                return df, {"encoding": enc}
            except pd.errors.EmptyDataError:
                logger.warning(f"CSV file is empty: {file_path}")
                return pd.DataFrame(), {"encoding": enc, "empty_file": True}
            except UnicodeDecodeError:
                continue
        
        raise ValueError(f"Could not decode CSV file with any supported encoding")
    
    def _read_excel(
        self,
        file_path: Path,
        sheet_name: Union[str, int] = 0,
        **kwargs
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Read Excel file"""
        # Get sheet names
        excel_file = pd.ExcelFile(file_path)
        sheet_names = excel_file.sheet_names
        
        # Read the specified sheet
        df = pd.read_excel(file_path, sheet_name=sheet_name, **kwargs)
        
        metadata = {
            "sheet_names": sheet_names,
            "active_sheet": sheet_names[sheet_name] if isinstance(sheet_name, int) else sheet_name
        }
        
        return df, metadata
    
    def _read_spss(
        self,
        file_path: Path,
        **kwargs
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Read SPSS (.sav) file"""
        if not self._spss_available:
            raise ImportError("pyreadstat is required to read SPSS files")
        
        import pyreadstat
        
        df, meta = pyreadstat.read_sav(str(file_path), **kwargs)
        
        metadata = {
            "variable_labels": meta.column_names_to_labels,
            "value_labels": meta.variable_value_labels,
            "column_labels": meta.column_labels,
            "original_variable_types": meta.original_variable_types
        }
        
        return df, metadata
    
    def _read_json(
        self,
        file_path: Path,
        **kwargs
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Read JSON file"""
        df = pd.read_json(file_path, **kwargs)
        return df, {}
    
    def write_file(
        self,
        df: pd.DataFrame,
        file_path: Union[str, Path],
        file_type: Optional[str] = None,
        **kwargs
    ) -> Path:
        """
        Write DataFrame to file.
        
        Args:
            df: DataFrame to write
            file_path: Output file path
            file_type: Optional file type override
            **kwargs: Additional arguments passed to the writer
            
        Returns:
            Path to written file
        """
        file_path = Path(file_path)
        
        if file_type is None:
            file_type = self.SUPPORTED_EXTENSIONS.get(file_path.suffix.lower())
        
        if file_type is None:
            file_type = 'csv'  # Default to CSV
            file_path = file_path.with_suffix('.csv')
        
        # Ensure directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Writing {file_type.upper()} file: {file_path.name}")
        
        try:
            if file_type == 'csv':
                df.to_csv(file_path, index=False, **kwargs)
            elif file_type == 'excel':
                df.to_excel(file_path, index=False, **kwargs)
            elif file_type == 'json':
                df.to_json(file_path, orient='records', indent=2, **kwargs)
            else:
                raise ValueError(f"Cannot write to file type: {file_type}")
            
            logger.info(f"Successfully wrote {len(df)} rows to {file_path.name}")
            return file_path
            
        except Exception as e:
            logger.error(f"Error writing file: {str(e)}")
            raise
    
    def write_json(
        self,
        data: Union[Dict, List],
        file_path: Union[str, Path],
        indent: int = 2
    ) -> Path:
        """
        Write JSON data to file.
        
        Args:
            data: Dictionary or list to write
            file_path: Output file path
            indent: JSON indentation level
            
        Returns:
            Path to written file
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, default=str)
        
        logger.info(f"Wrote JSON file: {file_path.name}")
        return file_path
    
    def read_yaml(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Read YAML file.
        
        Args:
            file_path: Path to YAML file
            
        Returns:
            Dictionary containing YAML data
        """
        file_path = Path(file_path)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        logger.info(f"Loaded YAML file: {file_path.name}")
        return data
    
    def write_yaml(
        self,
        data: Dict[str, Any],
        file_path: Union[str, Path]
    ) -> Path:
        """
        Write data to YAML file.
        
        Args:
            data: Dictionary to write
            file_path: Output file path
            
        Returns:
            Path to written file
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Wrote YAML file: {file_path.name}")
        return file_path


class MetadataHandler:
    """Handler for master metadata schemas"""
    
    def __init__(self, metadata_dir: Union[str, Path]):
        self.metadata_dir = Path(metadata_dir)
        self.file_handler = FileHandler()
    
    def load_master_schema(
        self,
        schema_name: str = "master_schema"
    ) -> Dict[str, Any]:
        """
        Load master schema definition.
        
        Args:
            schema_name: Name of the schema file (without extension)
            
        Returns:
            Dictionary containing schema definition
        """
        # Try YAML first, then JSON
        yaml_path = self.metadata_dir / f"{schema_name}.yaml"
        json_path = self.metadata_dir / f"{schema_name}.json"
        
        if yaml_path.exists():
            return self.file_handler.read_yaml(yaml_path)
        elif json_path.exists():
            with open(json_path, 'r') as f:
                return json.load(f)
        else:
            raise FileNotFoundError(
                f"Master schema not found: {schema_name}.yaml or {schema_name}.json"
            )
    
    def load_mapping_tables(
        self,
        mapping_name: str = "mapping_tables"
    ) -> Dict[str, Any]:
        """
        Load value mapping tables.
        
        Args:
            mapping_name: Name of the mapping file (without extension)
            
        Returns:
            Dictionary containing mapping tables
        """
        yaml_path = self.metadata_dir / f"{mapping_name}.yaml"
        json_path = self.metadata_dir / f"{mapping_name}.json"
        
        if yaml_path.exists():
            return self.file_handler.read_yaml(yaml_path)
        elif json_path.exists():
            with open(json_path, 'r') as f:
                return json.load(f)
        else:
            raise FileNotFoundError(
                f"Mapping tables not found: {mapping_name}.yaml or {mapping_name}.json"
            )
    
    def save_learned_mappings(
        self,
        mappings: Dict[str, Any],
        mapping_name: str = "learned_mappings"
    ) -> Path:
        """
        Save auto-learned mappings.
        
        Args:
            mappings: Dictionary of learned mappings
            mapping_name: Name for the output file
            
        Returns:
            Path to saved file
        """
        output_path = self.metadata_dir / f"{mapping_name}.yaml"
        return self.file_handler.write_yaml(mappings, output_path)


def get_dataframe_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate a summary of a DataFrame for LLM context.
    
    Args:
        df: DataFrame to summarize
        
    Returns:
        Dictionary containing DataFrame summary
    """
    summary = {
        "shape": {"rows": len(df), "columns": len(df.columns)},
        "columns": [],
        "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024
    }
    
    for col in df.columns:
        col_info = {
            "name": col,
            "dtype": str(df[col].dtype),
            "non_null_count": int(df[col].notna().sum()),
            "null_count": int(df[col].isna().sum()),
            "null_percentage": round(df[col].isna().sum() / len(df) * 100, 2),
            "unique_values": int(df[col].nunique())
        }
        
        # Add sample values
        non_null_values = df[col].dropna()
        if len(non_null_values) > 0:
            col_info["sample_values"] = non_null_values.head(5).tolist()
        
        # Add numeric statistics
        if pd.api.types.is_numeric_dtype(df[col]):
            col_info["min"] = float(df[col].min()) if not pd.isna(df[col].min()) else None
            col_info["max"] = float(df[col].max()) if not pd.isna(df[col].max()) else None
            col_info["mean"] = float(df[col].mean()) if not pd.isna(df[col].mean()) else None
        
        summary["columns"].append(col_info)
    
    return summary


def compare_dataframes(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame
) -> Dict[str, Any]:
    """
    Compare two DataFrames to identify changes.
    
    Args:
        df_before: Original DataFrame
        df_after: Modified DataFrame
        
    Returns:
        Dictionary containing comparison results
    """
    comparison = {
        "rows_before": len(df_before),
        "rows_after": len(df_after),
        "columns_before": list(df_before.columns),
        "columns_after": list(df_after.columns),
        "columns_added": [],
        "columns_removed": [],
        "columns_common": [],
        "column_changes": []
    }
    
    cols_before = set(df_before.columns)
    cols_after = set(df_after.columns)
    
    comparison["columns_added"] = list(cols_after - cols_before)
    comparison["columns_removed"] = list(cols_before - cols_after)
    comparison["columns_common"] = list(cols_before & cols_after)
    
    # Compare common columns
    for col in comparison["columns_common"]:
        if col in df_before.columns and col in df_after.columns:
            try:
                # Check if values changed
                before_vals = df_before[col].fillna("__NULL__")
                after_vals = df_after[col].fillna("__NULL__")
                
                if len(before_vals) == len(after_vals):
                    changes = (before_vals != after_vals).sum()
                    if changes > 0:
                        comparison["column_changes"].append({
                            "column": col,
                            "values_changed": int(changes),
                            "change_percentage": round(changes / len(before_vals) * 100, 2)
                        })
            except Exception:
                pass  # Skip columns that can't be compared
    
    return comparison


