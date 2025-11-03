from typing import List, Dict, Optional, Any, Annotated
from pydantic import BaseModel, Field
import time

class ColumnInfo(BaseModel):
    dtype: str
    sample_values: List[str]

class AgentStep(BaseModel):
    """Track individual agent steps for reasoning chain"""
    step_id: str
    action: str
    thought: str
    observation: str
    reflection: Optional[str] = None
    timestamp: float = Field(default_factory=time.time)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class AgentPlan(BaseModel):
    """Agent planning capabilities"""
    goal: str
    steps: List[str]
    current_step: int = 0
    context: Dict[str, Any] = Field(default_factory=dict)
    constraints: List[str] = Field(default_factory=list)

class AgentState(BaseModel):
    # Core data fields
    data_schema: Dict[str, Any]
    gx_rules: Optional[List[Any]] = None
    raw_suggestions: Optional[str] = None
    formatted_rules: Optional[List[Any]] = None
    normalized_suggestions: Optional[Dict[str, Any]] = None
    rule_suggestions: Optional[Annotated[List[Dict[str, Any]], "last"]] = None
    enhanced_prompt: Optional[str] = None  # RAG-enhanced prompt
    
    # Planning and context
    plan: Optional[AgentPlan] = None
    context: Dict[str, Any] = Field(default_factory=dict)
    thoughts: List[str] = Field(default_factory=list)
    observations: List[str] = Field(default_factory=list)
    reflections: List[str] = Field(default_factory=list)
    step_history: List[AgentStep] = Field(default_factory=list)
    quality_metrics: Dict[str, Any] = Field(default_factory=dict)
    confidence_scores: Dict[str, float] = Field(default_factory=dict)
    
    # Minimal state tracking for performance
    errors: List[str] = Field(default_factory=list)
    retry_count: int = 0
    max_retries: int = 2  # Reduced retries
    
    # Essential metrics only
    execution_start_time: float = Field(default_factory=time.time)
    execution_metrics: Dict[str, Any] = Field(default_factory=dict)
