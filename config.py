"""
Configuration module for the Agentic AI Data Harmonization System.
Contains all settings, paths, and Azure OpenAI configuration.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

# Base paths
BASE_DIR = Path(__file__).parent.absolute()
DATA_DIR = BASE_DIR / "data"
INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "output"
REPORTS_DIR = BASE_DIR / "reports"
METADATA_DIR = BASE_DIR / "metadata"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
for directory in [INPUT_DIR, OUTPUT_DIR, REPORTS_DIR, METADATA_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# =============================================================================
# AZURE OPENAI CONFIGURATION
# =============================================================================

# Set to False to ignore env overrides (useful when env has stale values).
USE_ENV_OVERRIDES = True  # Read API key from .env (keep False only if using a safe default)
AZURE_OPENAI_ENDPOINT_DEFAULT = "https://ankit-mkozog2d-eastus2.cognitiveservices.azure.com/"
AZURE_OPENAI_DEPLOYMENT_DEFAULT = "gpt-5.2-chat"
AZURE_OPENAI_API_VERSION_DEFAULT = "2024-12-01-preview"
# Set via .env or environment variable AZURE_OPENAI_API_KEY (never commit real key)
AZURE_OPENAI_API_KEY_DEFAULT = ""


def _get_env_or_default(name: str, default: str) -> str:
    if USE_ENV_OVERRIDES:
        value = os.getenv(name)
        if value:
            return value
    return default

@dataclass
class AzureOpenAIConfig:
    """Azure OpenAI API Configuration"""
    endpoint: str = field(
        default_factory=lambda: _get_env_or_default(
            "AZURE_OPENAI_ENDPOINT",
            AZURE_OPENAI_ENDPOINT_DEFAULT
        )
    )
    deployment: str = field(
        default_factory=lambda: _get_env_or_default(
            "AZURE_OPENAI_DEPLOYMENT",
            AZURE_OPENAI_DEPLOYMENT_DEFAULT
        )
    )
    api_version: str = field(
        default_factory=lambda: _get_env_or_default(
            "AZURE_OPENAI_API_VERSION",
            AZURE_OPENAI_API_VERSION_DEFAULT
        )
    )
    api_key: str = field(
        default_factory=lambda: _get_env_or_default(
            "AZURE_OPENAI_API_KEY",
            AZURE_OPENAI_API_KEY_DEFAULT
        )
    )
    max_completion_tokens: int = 16384
    temperature: float = 1.0  # gpt-5.2-chat only supports default temperature
    
    def validate(self) -> bool:
        """Validate that API key is set"""
        if not self.api_key:
            raise ValueError("AZURE_OPENAI_API_KEY environment variable is not set")
        return True

# Global Azure OpenAI config instance
AZURE_CONFIG = AzureOpenAIConfig()

# =============================================================================
# AGENT CONFIGURATION
# =============================================================================

@dataclass
class AgentConfig:
    """Configuration for individual agents"""
    name: str
    confidence_threshold: float = 0.85
    max_retries: int = 3
    timeout_seconds: int = 120
    log_level: str = "INFO"

# Agent-specific configurations
STRUCTURAL_VALIDATION_CONFIG = AgentConfig(
    name="StructuralValidationAgent",
    confidence_threshold=0.80,
    max_retries=3
)

DATA_QUALITY_CONFIG = AgentConfig(
    name="DataQualityAgent",
    confidence_threshold=0.85,
    max_retries=3
)

HARMONIZATION_CONFIG = AgentConfig(
    name="HarmonizationAgent",
    confidence_threshold=0.90,
    max_retries=3
)

SUPERVISOR_CONFIG = AgentConfig(
    name="SupervisorOrchestratorAgent",
    confidence_threshold=0.95,
    max_retries=2
)

# =============================================================================
# DATA QUALITY THRESHOLDS
# =============================================================================

@dataclass
class DataQualityThresholds:
    """Thresholds for data quality checks"""
    max_missing_percentage: float = 20.0  # Maximum allowed missing %
    min_sample_size: int = 30  # Minimum required records
    outlier_std_threshold: float = 3.0  # Standard deviations for outlier detection
    duplicate_threshold: float = 5.0  # Maximum allowed duplicate %
    
    # Issue severity classifications
    blocking_missing_threshold: float = 50.0  # Above this = blocking issue
    fixable_missing_threshold: float = 20.0  # Above this = fixable issue
    
DQ_THRESHOLDS = DataQualityThresholds()

# =============================================================================
# HARMONIZATION CONFIGURATION
# =============================================================================

@dataclass
class HarmonizationConfig:
    """Configuration for data harmonization"""
    # Scale normalization settings
    default_output_scale_min: float = 0.0
    default_output_scale_max: float = 100.0
    
    # Date format standardization
    target_date_format: str = "%Y-%m-%d"
    
    # String standardization
    standardize_case: str = "upper"  # upper, lower, title
    strip_whitespace: bool = True
    
    # Canonical column naming
    use_snake_case: bool = True

HARMONIZATION_SETTINGS = HarmonizationConfig()

# =============================================================================
# SUPPORTED FILE FORMATS
# =============================================================================

SUPPORTED_FORMATS = {
    ".csv": "CSV",
    ".xlsx": "Excel",
    ".xls": "Excel (Legacy)",
    ".sav": "SPSS",
    ".json": "JSON"
}

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        },
        "detailed": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "standard",
            "stream": "ext://sys.stdout"
        },
        "file": {
            "class": "logging.FileHandler",
            "level": "DEBUG",
            "formatter": "detailed",
            "filename": str(LOGS_DIR / "harmonization.log"),
            "mode": "a"
        }
    },
    "loggers": {
        "": {
            "level": "DEBUG",
            "handlers": ["console", "file"],
            "propagate": True
        }
    }
}

# =============================================================================
# VECTOR MEMORY CONFIGURATION (Stretch Goal)
# =============================================================================

@dataclass
class VectorMemoryConfig:
    """Configuration for FAISS vector memory"""
    enabled: bool = True
    embedding_model: str = "all-MiniLM-L6-v2"
    index_path: str = str(BASE_DIR / "vector_store" / "schema_history.faiss")
    dimension: int = 384
    similarity_threshold: float = 0.75

VECTOR_MEMORY_CONFIG = VectorMemoryConfig()

# =============================================================================
# REPORT CONFIGURATION
# =============================================================================

@dataclass
class ReportConfig:
    """Configuration for report generation"""
    include_visualizations: bool = True
    include_sample_data: bool = True
    sample_size: int = 10
    html_template: str = "default"

REPORT_CONFIG = ReportConfig()

