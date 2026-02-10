"""
Base Agent class for the Agentic AI Data Harmonization System.
Provides common functionality for all specialized agents.
"""

import os
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import AzureOpenAI

from config import AZURE_CONFIG, LOGS_DIR
from models.schemas import (
    AgentResponse,
    AuditEntry,
    ProcessingStatus
)
from utils.logger import get_logger
from utils.token_tracker import record_token_usage


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the harmonization system.
    
    Provides:
    - Azure OpenAI client initialization
    - Common logging functionality
    - Audit trail management
    - Response formatting
    """
    
    def __init__(
        self,
        name: str,
        confidence_threshold: float = 0.85,
        max_retries: int = 3
    ):
        """
        Initialize the base agent.
        
        Args:
            name: Agent name for logging and identification
            confidence_threshold: Minimum confidence score for auto-approval
            max_retries: Maximum number of retry attempts for operations
        """
        self.name = name
        self.confidence_threshold = confidence_threshold
        self.max_retries = max_retries
        
        # Initialize logging
        self.logger = get_logger(name, LOGS_DIR / f"{name.lower()}.log")
        
        # Initialize Azure OpenAI client
        self.client = self._init_azure_client()
        
        # Tracking metrics
        self.llm_calls_made = 0
        self.tokens_used = 0
        self.actions_taken: List[str] = []
        self.audit_entries: List[AuditEntry] = []
        
        # Execution timing
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        
        self.logger.info(f"Agent initialized: {name}")
    
    def _init_azure_client(self) -> AzureOpenAI:
        """
        Initialize Azure OpenAI client using configuration.
        
        Returns:
            Configured AzureOpenAI client
        """
        AZURE_CONFIG.validate()
        
        return AzureOpenAI(
            api_version=AZURE_CONFIG.api_version,
            azure_endpoint=AZURE_CONFIG.endpoint,
            api_key=AZURE_CONFIG.api_key
        )
    
    def call_llm(
        self,
        messages: List[Dict[str, str]],
        purpose: str = "general",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Tuple[str, int]:
        """
        Make a call to Azure OpenAI LLM.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            purpose: Description of the call purpose for logging
            temperature: Optional temperature override
            max_tokens: Optional max tokens override
            
        Returns:
            Tuple of (response content, tokens used)
        """
        self.logger.llm_call(purpose)
        
        try:
            # Note: gpt-5.2-chat doesn't support custom temperature, so we don't pass it
            response = self.client.chat.completions.create(
                model=AZURE_CONFIG.deployment,
                messages=messages,
                max_completion_tokens=max_tokens or AZURE_CONFIG.max_completion_tokens
            )
            
            content = response.choices[0].message.content
            tokens = response.usage.total_tokens if response.usage else 0
            
            self.llm_calls_made += 1
            self.tokens_used += tokens

            # Record usage globally so the app can show session stats
            record_token_usage(self.name, purpose, tokens)
            
            self.logger.debug(f"LLM response received: {tokens} tokens")
            
            return content, tokens
            
        except Exception as e:
            self.logger.error(f"LLM call failed: {str(e)}")
            raise
    
    def add_audit_entry(
        self,
        action: str,
        status: ProcessingStatus,
        confidence_score: Optional[float] = None,
        details: Optional[str] = None,
        input_summary: Optional[Dict[str, Any]] = None,
        output_summary: Optional[Dict[str, Any]] = None,
        duration: float = 0.0
    ) -> None:
        """
        Add an entry to the audit trail.
        
        Args:
            action: Description of the action taken
            status: Status of the action
            confidence_score: Optional confidence score
            details: Optional additional details
            input_summary: Optional input data summary
            output_summary: Optional output data summary
            duration: Duration of the action in seconds
        """
        entry = AuditEntry(
            timestamp=datetime.now(),
            agent_name=self.name,
            action=action,
            status=status,
            confidence_score=confidence_score,
            details=details,
            input_summary=input_summary,
            output_summary=output_summary,
            duration_seconds=duration
        )
        
        self.audit_entries.append(entry)
        self.actions_taken.append(f"{action} ({status.value})")
        
        self.logger.agent_action(action, status.value, confidence_score)
    
    def create_response(
        self,
        success: bool,
        confidence_score: float,
        result: Any,
        errors: Optional[List[str]] = None,
        warnings: Optional[List[str]] = None
    ) -> AgentResponse:
        """
        Create a standardized agent response.
        
        Args:
            success: Whether the operation was successful
            confidence_score: Confidence score for the result
            result: The actual result data
            errors: Optional list of errors
            warnings: Optional list of warnings
            
        Returns:
            AgentResponse object
        """
        execution_time = 0.0
        if self.start_time:
            self.end_time = datetime.now()
            execution_time = (self.end_time - self.start_time).total_seconds()
        
        status = ProcessingStatus.COMPLETED if success else ProcessingStatus.FAILED
        if success and confidence_score < self.confidence_threshold:
            status = ProcessingStatus.REQUIRES_REVIEW
        
        return AgentResponse(
            agent_name=self.name,
            status=status,
            success=success,
            confidence_score=confidence_score,
            result=result,
            execution_time_seconds=execution_time,
            llm_calls_made=self.llm_calls_made,
            tokens_used=self.tokens_used,
            errors=errors or [],
            warnings=warnings or [],
            actions_taken=self.actions_taken,
            timestamp=datetime.now()
        )
    
    def start_execution(self) -> None:
        """Mark the start of agent execution"""
        self.start_time = datetime.now()
        self.llm_calls_made = 0
        self.tokens_used = 0
        self.actions_taken = []
        self.audit_entries = []
        
        self.logger.separator(f"{self.name} Execution")
        self.add_audit_entry(
            action="Agent execution started",
            status=ProcessingStatus.IN_PROGRESS
        )
    
    def end_execution(self, success: bool) -> None:
        """Mark the end of agent execution"""
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds() if self.start_time else 0
        
        status = ProcessingStatus.COMPLETED if success else ProcessingStatus.FAILED
        self.add_audit_entry(
            action="Agent execution completed",
            status=status,
            duration=duration
        )
        
        self.logger.info(
            f"Execution completed: success={success}, "
            f"duration={duration:.2f}s, "
            f"llm_calls={self.llm_calls_made}, "
            f"tokens={self.tokens_used}"
        )
    
    def retry_with_backoff(
        self,
        func,
        *args,
        max_retries: Optional[int] = None,
        **kwargs
    ) -> Any:
        """
        Execute a function with exponential backoff retry.
        
        Args:
            func: Function to execute
            *args: Positional arguments for the function
            max_retries: Optional retry count override
            **kwargs: Keyword arguments for the function
            
        Returns:
            Function result
        """
        retries = max_retries or self.max_retries
        last_exception = None
        
        for attempt in range(retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                wait_time = 2 ** attempt  # Exponential backoff
                
                self.logger.warning(
                    f"Attempt {attempt + 1}/{retries} failed: {str(e)}. "
                    f"Retrying in {wait_time}s..."
                )
                
                self.add_audit_entry(
                    action=f"Retry attempt {attempt + 1}",
                    status=ProcessingStatus.IN_PROGRESS,
                    details=str(e)
                )
                
                time.sleep(wait_time)
        
        self.logger.error(f"All {retries} retry attempts failed")
        raise last_exception
    
    @abstractmethod
    def execute(self, *args, **kwargs) -> AgentResponse:
        """
        Execute the agent's main task.
        Must be implemented by each specialized agent.
        
        Returns:
            AgentResponse with results
        """
        pass
    
    @abstractmethod
    def get_prompt_template(self) -> str:
        """
        Get the agent's prompt template.
        Must be implemented by each specialized agent.
        
        Returns:
            Prompt template string
        """
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"

