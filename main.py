#!/usr/bin/env python3
"""
Agentic AI Data Harmonization System
=====================================

Main entry point for the autonomous data harmonization pipeline.
Orchestrates intelligent data ingestion, validation, quality assessment,
and harmonization using LLM-powered agents.

Usage:
    python main.py                           # Run with defaults
    python main.py --input data/input/raw.csv
    python main.py --input data.xlsx --schema custom_schema
    python main.py --help                    # Show all options

Author: AI-Generated for Hackathon
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

from config import (
    INPUT_DIR,
    OUTPUT_DIR,
    REPORTS_DIR,
    METADATA_DIR,
    AZURE_CONFIG
)
from agents.supervisor_agent import SupervisorOrchestratorAgent
from utils.logger import setup_logging, get_logger, console
from utils.file_handlers import FileHandler
from utils.multi_source_harmonizer import MultiSourceHarmonizer


def validate_environment():
    """Validate that required environment variables are set"""
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    if not api_key:
        console.print(
            "[red bold]ERROR: AZURE_OPENAI_API_KEY environment variable is not set![/red bold]"
        )
        console.print("\nPlease set your Azure OpenAI API key:")
        console.print("  [cyan]export AZURE_OPENAI_API_KEY='your-api-key-here'[/cyan]  (Linux/Mac)")
        console.print("  [cyan]set AZURE_OPENAI_API_KEY=your-api-key-here[/cyan]  (Windows CMD)")
        console.print("  [cyan]$env:AZURE_OPENAI_API_KEY='your-api-key-here'[/cyan]  (PowerShell)")
        console.print("\nOr create a .env file with:")
        console.print("  [cyan]AZURE_OPENAI_API_KEY=your-api-key-here[/cyan]")
        return False
    return True


def find_input_file(input_path: str = None) -> Path:
    """
    Find or validate input file.
    
    Args:
        input_path: Optional explicit input path
        
    Returns:
        Path to input file
    """
    if input_path:
        path = Path(input_path)
        if path.exists():
            return path
        # Check in input directory
        input_dir_path = INPUT_DIR / path.name
        if input_dir_path.exists():
            return input_dir_path
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Look for files in input directory
    supported_extensions = ['.csv', '.xlsx', '.xls', '.sav', '.json']
    
    for ext in supported_extensions:
        files = list(INPUT_DIR.glob(f'*{ext}'))
        if files:
            # Return the most recently modified file
            return max(files, key=lambda f: f.stat().st_mtime)
    
    # Check project root for common files
    for ext in supported_extensions:
        files = list(PROJECT_ROOT.glob(f'*{ext}'))
        if files:
            return max(files, key=lambda f: f.stat().st_mtime)
    
    raise FileNotFoundError(
        f"No input files found. Please place a data file in {INPUT_DIR} "
        f"or specify with --input"
    )


def create_sample_data():
    """Create sample input data for demonstration"""
    import pandas as pd
    import numpy as np
    
    console.print("[yellow]Creating sample data for demonstration...[/yellow]")
    
    np.random.seed(42)
    n_rows = 500
    
    # Generate sample survey data
    data = {
        'respondent_id': range(1001, 1001 + n_rows),
        'survey_date': pd.date_range('2024-01-01', periods=n_rows, freq='h').strftime('%Y-%m-%d'),
        'region': np.random.choice(['North', 'South', 'East', 'West', 'Central'], n_rows),
        'country': np.random.choice(['USA', 'UK', 'Germany', 'France', 'Japan', 'Australia'], n_rows),
        'brand': np.random.choice(['Brand_A', 'Brand_B', 'Brand_C', 'Brand_D'], n_rows),
        'product_category': np.random.choice(['Electronics', 'Apparel', 'Food', 'Beverages'], n_rows),
        'satisfaction_score': np.random.randint(1, 11, n_rows),  # 1-10 scale
        'nps_score': np.random.randint(0, 11, n_rows),  # 0-10 scale
        'purchase_intent': np.random.choice(['Very Unlikely', 'Unlikely', 'Neutral', 'Likely', 'Very Likely'], n_rows),
        'age_group': np.random.choice(['18-24', '25-34', '35-44', '45-54', '55+'], n_rows),
        'gender': np.random.choice(['Male', 'Female', 'Other', 'Prefer not to say'], n_rows),
        'income_bracket': np.random.choice(['<25K', '25K-50K', '50K-75K', '75K-100K', '>100K'], n_rows),
        'channel': np.random.choice(['Online', 'Retail', 'Mobile', 'Social'], n_rows),
        'wave': np.random.choice(['W1_2024', 'W2_2024', 'W3_2024', 'W4_2024'], n_rows),
        'weight': np.round(np.random.uniform(0.5, 2.0, n_rows), 3),
        'comments': np.random.choice([
            'Great product!', 'Could be better', 'Excellent service',
            'Not satisfied', 'Will buy again', '', np.nan
        ], n_rows)
    }
    
    df = pd.DataFrame(data)
    
    # Introduce some data quality issues for demonstration
    # Add some missing values
    missing_idx = np.random.choice(n_rows, size=int(n_rows * 0.05), replace=False)
    df.loc[missing_idx, 'satisfaction_score'] = np.nan
    
    # Add some outliers
    outlier_idx = np.random.choice(n_rows, size=5, replace=False)
    df.loc[outlier_idx, 'satisfaction_score'] = np.random.choice([15, -5, 100], 5)
    
    # Add some duplicates
    dup_rows = df.sample(10)
    df = pd.concat([df, dup_rows], ignore_index=True)
    
    # Save to input directory
    output_path = INPUT_DIR / 'raw.csv'
    df.to_csv(output_path, index=False)
    
    console.print(f"[green]Sample data created: {output_path}[/green]")
    console.print(f"  Rows: {len(df)}, Columns: {len(df.columns)}")
    
    return output_path


def create_sample_metadata():
    """Create sample master schema and mapping tables"""
    import yaml
    
    console.print("[yellow]Creating sample metadata...[/yellow]")
    
    # Master Schema
    master_schema = {
        "schema_name": "survey_data_canonical",
        "version": "1.0.0",
        "description": "Canonical schema for harmonized survey data",
        "columns": [
            {
                "name": "respondent_id",
                "canonical_name": "respondent_id",
                "data_type": "integer",
                "description": "Unique respondent identifier",
                "is_required": True,
                "is_key": True,
                "aliases": ["resp_id", "id", "participant_id"]
            },
            {
                "name": "survey_date",
                "canonical_name": "survey_date",
                "data_type": "datetime",
                "description": "Date of survey completion",
                "is_required": True,
                "aliases": ["date", "completion_date", "interview_date"]
            },
            {
                "name": "region",
                "canonical_name": "region",
                "data_type": "categorical",
                "description": "Geographic region",
                "is_required": True,
                "aliases": ["geo_region", "area", "territory"]
            },
            {
                "name": "country",
                "canonical_name": "country",
                "data_type": "categorical",
                "description": "Country code or name",
                "is_required": True,
                "aliases": ["country_code", "nation", "market"]
            },
            {
                "name": "brand",
                "canonical_name": "brand",
                "data_type": "categorical",
                "description": "Brand name",
                "is_required": True,
                "aliases": ["brand_name", "product_brand"]
            },
            {
                "name": "product_category",
                "canonical_name": "product_category",
                "data_type": "categorical",
                "description": "Product category",
                "is_required": False,
                "aliases": ["category", "prod_cat", "segment"]
            },
            {
                "name": "satisfaction_score",
                "canonical_name": "satisfaction_score",
                "data_type": "float",
                "description": "Customer satisfaction score (normalized 0-100)",
                "is_required": True,
                "validation_rules": {
                    "min_value": 0,
                    "max_value": 100
                },
                "aliases": ["csat", "satisfaction", "sat_score"]
            },
            {
                "name": "nps_score",
                "canonical_name": "nps_score",
                "data_type": "integer",
                "description": "Net Promoter Score (0-10)",
                "is_required": True,
                "validation_rules": {
                    "min_value": 0,
                    "max_value": 10
                },
                "aliases": ["nps", "promoter_score"]
            },
            {
                "name": "purchase_intent",
                "canonical_name": "purchase_intent",
                "data_type": "categorical",
                "description": "Purchase intention",
                "is_required": False,
                "aliases": ["intent", "buying_intent", "purchase_likelihood"]
            },
            {
                "name": "demographic_age",
                "canonical_name": "demographic_age",
                "data_type": "categorical",
                "description": "Age group",
                "is_required": False,
                "aliases": ["age_group", "age", "age_bracket"]
            },
            {
                "name": "demographic_gender",
                "canonical_name": "demographic_gender",
                "data_type": "categorical",
                "description": "Gender",
                "is_required": False,
                "aliases": ["gender", "sex"]
            },
            {
                "name": "channel",
                "canonical_name": "channel",
                "data_type": "categorical",
                "description": "Sales/Marketing channel",
                "is_required": False,
                "aliases": ["sales_channel", "marketing_channel"]
            },
            {
                "name": "wave",
                "canonical_name": "wave",
                "data_type": "string",
                "description": "Survey wave identifier",
                "is_required": True,
                "aliases": ["survey_wave", "wave_id", "period"]
            },
            {
                "name": "weight",
                "canonical_name": "weight",
                "data_type": "float",
                "description": "Statistical weight",
                "is_required": False,
                "default_value": 1.0,
                "aliases": ["stat_weight", "sample_weight"]
            }
        ],
        "business_rules": [
            "Minimum 30 records required for valid analysis",
            "satisfaction_score must be in range 0-100",
            "nps_score must be in range 0-10",
            "respondent_id must be unique"
        ]
    }
    
    schema_path = METADATA_DIR / 'master_schema.yaml'
    with open(schema_path, 'w') as f:
        yaml.dump(master_schema, f, default_flow_style=False, sort_keys=False)
    
    console.print(f"[green]Master schema created: {schema_path}[/green]")
    
    # Mapping Tables
    mapping_tables = {
        "region": {
            "N": "North",
            "S": "South",
            "E": "East",
            "W": "West",
            "C": "Central",
            "NORTH": "North",
            "SOUTH": "South",
            "EAST": "East",
            "WEST": "West",
            "CENTRAL": "Central"
        },
        "purchase_intent": {
            "1": "Very Unlikely",
            "2": "Unlikely",
            "3": "Neutral",
            "4": "Likely",
            "5": "Very Likely",
            "VU": "Very Unlikely",
            "U": "Unlikely",
            "N": "Neutral",
            "L": "Likely",
            "VL": "Very Likely"
        },
        "channel": {
            "ONL": "Online",
            "RET": "Retail",
            "MOB": "Mobile",
            "SOC": "Social",
            "online": "Online",
            "retail": "Retail",
            "mobile": "Mobile",
            "social": "Social"
        }
    }
    
    mapping_path = METADATA_DIR / 'mapping_tables.yaml'
    with open(mapping_path, 'w') as f:
        yaml.dump(mapping_tables, f, default_flow_style=False, sort_keys=False)
    
    console.print(f"[green]Mapping tables created: {mapping_path}[/green]")
    
    return schema_path, mapping_path


def run_pipeline(args):
    """
    Run the harmonization pipeline.
    
    Args:
        args: Command line arguments
    """
    logger = get_logger("Main")
    
    # Display banner
    console.print("\n" + "=" * 70, style="bold magenta")
    console.print("🤖 AGENTIC AI DATA HARMONIZATION SYSTEM", style="bold white")
    console.print("   Autonomous Data Pipeline powered by Azure OpenAI", style="dim")
    console.print("=" * 70 + "\n", style="bold magenta")
    
    # Validate environment
    if not validate_environment():
        sys.exit(1)
    
    # Create sample data if requested or no input found
    if args.create_sample:
        create_sample_data()
        create_sample_metadata()
        if not args.input:
            console.print("[yellow]Run again without --create-sample to process the data[/yellow]")
            return
    
    # Multi-source pipeline execution
    if args.historical or args.current or args.incremental or args.master_metadata:
        console.print("\n[cyan]Running multi-source harmonization pipeline...[/cyan]\n")
        harmonizer = MultiSourceHarmonizer()
        result = harmonizer.run(
            master_metadata_file=args.master_metadata,
            historical_file=args.historical,
            current_file=args.current,
            incremental_file=args.incremental,
            validation_rules_file=args.validation_rules,
            mapping_table_file=args.mapping_table,
            descriptive_stats_file=args.descriptive_stats,
            output_file=args.output,
            full_reprocess=args.full_reprocess
        )
        console.print("[green]Multi-source harmonization completed.[/green]")
        console.print(f"📄 Output File: {result.get('output_file')}")
        console.print(f"🧭 Mapping Table: {result.get('mapping_table_file')}")
        console.print(f"📊 Calibration Report: {result.get('calibration_report')}")
        console.print(f"📈 Descriptive Stats: {result.get('descriptive_statistics')}")
        console.print(f"🧾 Validation Flags: {result.get('validation_flags')}")
        return True

    # Find input file (single-source pipeline)
    try:
        input_file = find_input_file(args.input)
        console.print(f"[cyan]Input file: {input_file}[/cyan]")
    except FileNotFoundError as e:
        console.print(f"[red]{str(e)}[/red]")
        console.print("\n[yellow]Creating sample data for demonstration...[/yellow]")
        input_file = create_sample_data()
        create_sample_metadata()
    
    # Ensure metadata exists
    schema_path = METADATA_DIR / f"{args.schema}.yaml"
    if not schema_path.exists() and not (METADATA_DIR / f"{args.schema}.json").exists():
        console.print(f"[yellow]Schema '{args.schema}' not found, creating defaults...[/yellow]")
        create_sample_metadata()
    
    # Initialize supervisor agent
    console.print("\n[cyan]Initializing Agentic AI Pipeline...[/cyan]\n")
    supervisor = SupervisorOrchestratorAgent()
    
    # Parse business rules
    business_rules = None
    if args.rules:
        business_rules = [r.strip() for r in args.rules.split(';')]
    
    # Determine output path
    output_file = args.output
    if not output_file:
        output_file = str(OUTPUT_DIR / "harmonized.csv")
    
    # Execute pipeline
    try:
        result = supervisor.execute(
            input_file=str(input_file),
            master_schema_name=args.schema,
            mapping_tables_name=args.mappings,
            business_rules=business_rules,
            output_file=output_file
        )
        
        # Display summary
        console.print("\n" + "=" * 70, style="bold cyan")
        console.print("📊 PIPELINE EXECUTION SUMMARY", style="bold white")
        console.print("=" * 70, style="bold cyan")
        
        if result.success:
            console.print("\n✅ [green bold]Pipeline completed successfully![/green bold]\n")
        else:
            console.print("\n⚠️  [yellow bold]Pipeline completed with warnings[/yellow bold]\n")
        
        if result.result:
            pipeline_data = result.result
            console.print(f"📁 Input File:  {pipeline_data.get('input_file', 'N/A')}")
            console.print(f"📄 Output File: {pipeline_data.get('output_file', 'N/A')}")
            console.print(f"📊 Quality Score: {pipeline_data.get('final_quality_score', 0):.1f}%")
            console.print(f"🎯 Confidence: {pipeline_data.get('final_confidence_score', 0):.1%}")
            console.print(f"⏱️  Duration: {pipeline_data.get('total_processing_time_seconds', 0):.2f}s")
            console.print(f"🤖 LLM Calls: {pipeline_data.get('total_llm_calls', 0)}")
            console.print(f"🔤 Tokens Used: {pipeline_data.get('total_tokens_used', 0):,}")
            
            console.print("\n📋 Generated Reports:")
            for report in pipeline_data.get('reports_generated', []):
                console.print(f"   • {report}")
        
        if result.errors:
            console.print("\n[red]Errors:[/red]")
            for error in result.errors:
                console.print(f"   ❌ {error}")
        
        if result.warnings:
            console.print("\n[yellow]Warnings:[/yellow]")
            for warning in result.warnings[:5]:
                console.print(f"   ⚠️  {warning}")
        
        console.print("\n" + "=" * 70 + "\n", style="bold cyan")
        
        return result.success
        
    except Exception as e:
        console.print(f"\n[red bold]Pipeline failed: {str(e)}[/red bold]")
        logger.error(f"Pipeline execution failed: {str(e)}")
        return False


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Agentic AI Data Harmonization System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                              # Auto-detect input, use defaults
  python main.py --input data/raw.csv         # Specify input file
  python main.py --input data.xlsx --output harmonized.csv
  python main.py --schema custom_schema       # Use custom schema
  python main.py --create-sample              # Create sample data first
  python main.py --rules "min 100 records; satisfaction range 1-10"

For more information, visit the project documentation.
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        help='Path to input data file (CSV, Excel, SPSS, JSON)'
    )

    parser.add_argument(
        '--historical',
        type=str,
        help='Historical data file (Vendor A, old structure)'
    )

    parser.add_argument(
        '--current',
        type=str,
        help='Current data file (Vendor B, new structure)'
    )

    parser.add_argument(
        '--incremental',
        type=str,
        help='Incremental data file (new respondents)'
    )

    parser.add_argument(
        '--master-metadata',
        type=str,
        help='Master metadata file (variable dictionary, scales, mappings)'
    )

    parser.add_argument(
        '--validation-rules',
        type=str,
        help='User validation rules file (JSON/YAML/TXT)'
    )

    parser.add_argument(
        '--mapping-table',
        type=str,
        help='Mapping table file (self-updating)'
    )

    parser.add_argument(
        '--descriptive-stats',
        type=str,
        help='Baseline descriptive statistics file (<= 1000 rows)'
    )

    parser.add_argument(
        '--full-reprocess',
        action='store_true',
        help='Reprocess history even when incremental file is provided'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Path for output harmonized file'
    )
    
    parser.add_argument(
        '--schema', '-s',
        type=str,
        default='master_schema',
        help='Name of master schema file (without extension)'
    )
    
    parser.add_argument(
        '--mappings', '-m',
        type=str,
        default='mapping_tables',
        help='Name of mapping tables file (without extension)'
    )
    
    parser.add_argument(
        '--rules', '-r',
        type=str,
        help='Business rules separated by semicolons'
    )
    
    parser.add_argument(
        '--create-sample',
        action='store_true',
        help='Create sample data and metadata files'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Run pipeline
    success = run_pipeline(args)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()


