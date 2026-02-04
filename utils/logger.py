"""
Logging utilities for the Agentic AI Data Harmonization System.
Provides centralized logging with rich formatting and file output.
"""

import logging
import logging.config
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

# Custom theme for rich console
custom_theme = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "red bold",
    "success": "green bold",
    "agent": "magenta bold",
    "confidence": "blue"
})

console = Console(theme=custom_theme)


class AgentLogFormatter(logging.Formatter):
    """Custom formatter for agent logs with confidence scores"""
    
    FORMATS = {
        logging.DEBUG: "%(asctime)s | DEBUG | %(name)s | %(message)s",
        logging.INFO: "%(asctime)s | INFO  | %(name)s | %(message)s",
        logging.WARNING: "%(asctime)s | WARN  | %(name)s | %(message)s",
        logging.ERROR: "%(asctime)s | ERROR | %(name)s | %(message)s",
        logging.CRITICAL: "%(asctime)s | CRIT  | %(name)s | %(message)s"
    }
    
    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno, self.FORMATS[logging.INFO])
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)


class AgentLogger:
    """Enhanced logger for agents with confidence score tracking"""
    
    def __init__(self, name: str, log_file: Optional[Path] = None):
        self.name = name
        self.logger = logging.getLogger(name)
        self.log_file = log_file
        self._setup_handlers()
        
    def _setup_handlers(self):
        """Setup console and file handlers"""
        self.logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Rich console handler
        rich_handler = RichHandler(
            console=console,
            show_time=True,
            show_path=False,
            rich_tracebacks=True
        )
        rich_handler.setLevel(logging.INFO)
        self.logger.addHandler(rich_handler)
        
        # File handler if specified
        if self.log_file:
            file_handler = logging.FileHandler(self.log_file, mode='a')
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(AgentLogFormatter())
            self.logger.addHandler(file_handler)
    
    def info(self, message: str, confidence: Optional[float] = None, **kwargs):
        """Log info message with optional confidence score"""
        if confidence is not None:
            message = f"{message} [confidence: {confidence:.2%}]"
        self.logger.info(message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self.logger.debug(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self.logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message"""
        self.logger.error(message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message"""
        self.logger.critical(message, **kwargs)
    
    def agent_action(self, action: str, status: str = "started", confidence: Optional[float] = None):
        """Log agent action with status"""
        # Use ASCII symbols to avoid Windows encoding issues
        symbol_map = {
            "started": ">>",
            "completed": "[OK]",
            "failed": "[X]",
            "retrying": "[~]",
            "analyzing": "[?]",
            "processing": "[*]",
            "in_progress": "[>]"
        }
        symbol = symbol_map.get(status.lower(), "[>]")
        
        message = f"{symbol} [{self.name}] {action} - {status.upper()}"
        if confidence is not None:
            message += f" [confidence: {confidence:.2%}]"
        
        self.logger.info(message)
    
    def llm_call(self, purpose: str, tokens: int = 0):
        """Log LLM API call"""
        self.logger.info(f"[LLM] Call: {purpose} (tokens: {tokens})")
    
    def separator(self, title: str = ""):
        """Print a visual separator"""
        if title:
            console.print(f"\n{'='*20} {title} {'='*20}\n", style="bold cyan")
        else:
            console.print("=" * 60, style="dim")


def get_logger(name: str, log_file: Optional[Path] = None) -> AgentLogger:
    """
    Get or create an AgentLogger instance.
    
    Args:
        name: Name of the logger (typically agent name)
        log_file: Optional path to log file
        
    Returns:
        AgentLogger instance
    """
    return AgentLogger(name, log_file)


def setup_logging(log_dir: Optional[Path] = None) -> None:
    """
    Setup global logging configuration.
    
    Args:
        log_dir: Directory to store log files
    """
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            RichHandler(console=console, show_time=True, show_path=False)
        ]
    )


def log_pipeline_start(input_file: str):
    """Log the start of a pipeline run"""
    console.print("\n" + "=" * 60, style="bold cyan")
    console.print("AGENTIC AI DATA HARMONIZATION PIPELINE", style="bold magenta")
    console.print("=" * 60, style="bold cyan")
    console.print(f"Input File: {input_file}", style="info")
    console.print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", style="info")
    console.print("=" * 60 + "\n", style="bold cyan")


def log_pipeline_end(success: bool, duration: float, output_file: Optional[str] = None):
    """Log the end of a pipeline run"""
    console.print("\n" + "=" * 60, style="bold cyan")
    
    if success:
        console.print("[OK] PIPELINE COMPLETED SUCCESSFULLY", style="success")
    else:
        console.print("[X] PIPELINE COMPLETED WITH ERRORS", style="error")
    
    console.print(f"Total Duration: {duration:.2f} seconds", style="info")
    
    if output_file:
        console.print(f"Output File: {output_file}", style="info")
    
    console.print("=" * 60 + "\n", style="bold cyan")


def log_agent_summary(agent_name: str, status: str, confidence: float, issues: int = 0):
    """Log agent execution summary"""
    status_symbol = "[OK]" if status == "completed" else "[X]" if status == "failed" else "[!]"
    confidence_color = "green" if confidence >= 0.9 else "yellow" if confidence >= 0.7 else "red"
    
    console.print(f"\n{status_symbol} {agent_name}", style="bold")
    console.print(f"   Status: {status}", style="info")
    console.print(f"   Confidence: {confidence:.2%}", style=confidence_color)
    
    if issues > 0:
        console.print(f"   Issues Found: {issues}", style="warning")

