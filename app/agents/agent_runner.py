from typing import Dict, List, Any, Optional, Literal, Union, Tuple
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph
from typing import Annotated
from app.tools.rule_tools import (
    fetch_gx_rules,
    suggest_column_rules,
    format_gx_rules,
    normalize_rule_suggestions,
    convert_to_rule_ms_format
)
from app.validation.llm_validator import validate_llm_response
from app.validation.metrics import record_validation_metric
from app.validation.middleware import AgentValidationContext
from app.core.config import settings
import logging
import time
import json

logger = logging.getLogger(__name__)

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
    
    # ReAct Pattern components
    thoughts: List[str] = Field(default_factory=list)
    actions: List[str] = Field(default_factory=list)
    observations: List[str] = Field(default_factory=list)
    reflections: List[str] = Field(default_factory=list)
    
    # Agent reasoning and planning
    current_step: Optional[str] = None
    plan: Optional[AgentPlan] = None
    step_history: List[AgentStep] = Field(default_factory=list)
    
    # Error handling and recovery
    errors: List[str] = Field(default_factory=list)
    retry_count: int = 0
    max_retries: int = 3
    
    # Quality and confidence tracking
    confidence_scores: Dict[str, float] = Field(default_factory=dict)
    quality_metrics: Dict[str, Any] = Field(default_factory=dict)
    
    # Context and memory
    context: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    working_memory: List[str] = Field(default_factory=list)
    
    # Execution metadata
    execution_start_time: float = Field(default_factory=time.time)
    execution_metrics: Dict[str, Any] = Field(default_factory=dict)

def create_agent_plan(schema: Dict[str, Any]) -> AgentPlan:
    """Create an execution plan based on the schema complexity"""
    column_count = len(schema)
    
    # Analyze schema complexity to determine plan
    if column_count <= 3:
        plan_type = "simple"
        steps = [
            "analyze_schema_structure",
            "fetch_rule_templates", 
            "generate_targeted_rules",
            "validate_and_format"
        ]
    elif column_count <= 10:
        plan_type = "moderate"
        steps = [
            "analyze_schema_structure",
            "identify_column_patterns",
            "fetch_rule_templates",
            "generate_rules_by_type",
            "cross_validate_rules",
            "optimize_rule_set"
        ]
    else:
        plan_type = "complex"
        steps = [
            "analyze_schema_structure",
            "chunk_columns_by_similarity", 
            "identify_pii_and_sensitive_data",
            "fetch_comprehensive_rules",
            "generate_rules_iteratively",
            "validate_rule_consistency",
            "optimize_performance",
            "final_quality_check"
        ]
    
    return AgentPlan(
        goal=f"Generate comprehensive validation rules for {column_count} columns",
        steps=steps,
        context={
            "plan_type": plan_type,
            "column_count": column_count,
            "estimated_duration": len(steps) * 2  # seconds
        },
        constraints=[
            "maintain_rule_consistency",
            "ensure_performance_optimization",
            "respect_data_privacy_requirements"
        ]
    )

def reason_before_action(state: AgentState, action: str) -> Dict[str, Any]:
    """ReAct: Reasoning before taking action"""
    domain = state.data_schema.get("domain", "unknown")
    column_count = len([k for k in state.data_schema.keys() if k != "domain"])
    
    if action == "fetch_rules":
        thought = f"I need to fetch appropriate rule templates for the '{domain}' domain. "
        thought += f"This schema has {column_count} columns. I should look for rules that are "
        thought += "commonly used for this domain type and can be adapted to these specific columns."
        
    elif action == "suggest":
        available_rules = len(state.gx_rules) if state.gx_rules else 0
        thought = f"I have {available_rules} rule templates available. Now I need to analyze each "
        thought += f"column in the {domain} schema and suggest the most appropriate validation rules. "
        thought += "I'll consider data types, potential PII, and domain-specific patterns."
        
    elif action == "format":
        thought = "I've received raw rule suggestions from the LLM. Now I need to format them "
        thought += "into a standardized structure that can be processed by the normalization step. "
        thought += "I'll ensure all required fields are present and properly formatted."
        
    elif action == "normalize":
        thought = "The formatted rules need to be normalized into the expected data structure. "
        thought += "I'll validate the rule syntax and ensure they conform to Great Expectations format. "
        thought += "Any malformed rules will be flagged for fallback processing."
        
    else:
        thought = f"Preparing to execute {action} step in the rule generation workflow."
    
    return {
        "thoughts": state.thoughts + [thought],
        "current_step": action,
        "step_history": state.step_history + [
            AgentStep(
                step_id=f"{action}_{len(state.step_history)}",
                action=action,
                thought=thought,
                observation="",  # Will be filled after action
                timestamp=time.time()
            )
        ]
    }

def observe_after_action(state: AgentState, action: str, result: Any) -> Dict[str, Any]:
    """ReAct: Observe and record results after action"""
    if action == "fetch_rules":
        rule_count = len(result) if isinstance(result, list) else 0
        observation = f"Successfully retrieved {rule_count} rule templates from the rule service. "
        if rule_count == 0:
            observation += "No rules were found - will need to use default templates."
        else:
            rule_types = set()
            for rule in result[:3]:  # Sample first few rules
                if isinstance(rule, dict) and "rule_name" in rule:
                    rule_types.add(rule["rule_name"].split("_")[2] if "_" in rule["rule_name"] else "generic")
            observation += f"Available rule types include: {', '.join(rule_types)}"
            
    elif action == "suggest":
        if isinstance(result, str) and result:
            lines = result.count('\n') + 1
            observation = f"LLM generated {lines} lines of rule suggestions. "
            observation += "The response appears to contain structured rule recommendations."
        else:
            observation = "LLM suggestion step completed but result format needs validation."
            
    elif action == "format":
        if isinstance(result, list):
            observation = f"Successfully formatted {len(result)} rule suggestions. "
            observation += "Rules are now in structured format for normalization."
        else:
            observation = "Formatting step completed but result structure needs review."
            
    elif action == "normalize":
        if isinstance(result, dict):
            valid_rules = sum(1 for v in result.values() if isinstance(v, dict) and v.get("expectations"))
            observation = f"Normalization completed with {valid_rules} valid rule structures. "
            if valid_rules == 0:
                observation += "No valid expectations found - fallback processing required."
        else:
            observation = "Normalization step completed but needs validation."
    else:
        observation = f"Completed {action} step successfully."
    
    # Update the most recent step with observation
    updated_history = state.step_history.copy()
    if updated_history:
        updated_history[-1].observation = observation
    
    return {
        "observations": state.observations + [observation],
        "step_history": updated_history
    }

def reflect_on_progress(state: AgentState) -> Dict[str, Any]:
    """ReAct: Reflect on overall progress and adjust strategy if needed"""
    completed_steps = len(state.step_history)
    total_planned = len(state.plan.steps) if state.plan else 4
    progress = completed_steps / total_planned if total_planned > 0 else 0
    
    reflection = f"Progress check: {completed_steps}/{total_planned} steps completed ({progress:.1%}). "
    
    # Analyze quality of recent steps
    recent_errors = [error for error in state.errors if error]
    if recent_errors:
        reflection += f"Encountered {len(recent_errors)} issues that may need attention. "
        
    # Check if we're on track with the plan
    if state.plan and completed_steps > 0:
        current_step_name = state.plan.steps[min(completed_steps - 1, len(state.plan.steps) - 1)]
        reflection += f"Currently executing '{current_step_name}' phase. "
        
        if progress < 0.5:
            reflection += "Early stage - focusing on data gathering and analysis. "
        elif progress < 0.8:
            reflection += "Mid-stage - actively generating and validating rules. "
        else:
            reflection += "Final stage - optimizing and preparing output. "
    
    # Assess need for strategy adjustment
    if len(recent_errors) > 2:
        reflection += "High error rate detected - may need to adjust approach or use fallback strategies."
    elif progress > 0.8 and not state.rule_suggestions:
        reflection += "Near completion but no final output yet - need to ensure proper conversion."
    
    return {
        "reflections": state.reflections + [reflection],
        "quality_metrics": {
            **state.quality_metrics,
            "progress": progress,
            "error_rate": len(recent_errors) / max(completed_steps, 1),
            "last_reflection_time": time.time()
        }
    }

def build_graph():
    """Build enhanced agentic workflow with ReAct pattern, planning, and reflection"""
    workflow = StateGraph(AgentState)

    # Planning phase
    workflow.add_node("create_plan", lambda s: {
        "plan": create_agent_plan(s.data_schema),
        "context": {
            **s.context,
            "planning_timestamp": time.time(),
            "domain": s.data_schema.get("domain", "unknown")
        }
    })

    # ReAct Pattern: Reason -> Act -> Observe
    workflow.add_node("reason_fetch", lambda s: reason_before_action(s, "fetch_rules"))
    
    workflow.add_node("fetch_rules", lambda s: {
        "gx_rules": fetch_gx_rules.invoke({"query": ""})
    })
    
    workflow.add_node("observe_fetch", lambda s: observe_after_action(s, "fetch_rules", s.gx_rules))

    workflow.add_node("reason_suggest", lambda s: reason_before_action(s, "suggest"))
    
    workflow.add_node("suggest", lambda s: {
        "raw_suggestions": suggest_column_rules.invoke({
            "data_schema": s.data_schema,
            "gx_rules": s.gx_rules
        })
    })
    
    workflow.add_node("observe_suggest", lambda s: observe_after_action(s, "suggest", s.raw_suggestions))

    workflow.add_node("reason_format", lambda s: reason_before_action(s, "format"))
    
    workflow.add_node("format", lambda s: {
        "formatted_rules": format_gx_rules.invoke(s.raw_suggestions)
    })
    
    workflow.add_node("observe_format", lambda s: observe_after_action(s, "format", s.formatted_rules))

    workflow.add_node("reason_normalize", lambda s: reason_before_action(s, "normalize"))
    
    workflow.add_node("normalize", lambda s: {
        "normalized_suggestions": normalize_rule_suggestions.invoke({"rule_input": {"raw": s.formatted_rules}})
    })
    
    workflow.add_node("observe_normalize", lambda s: observe_after_action(s, "normalize", s.normalized_suggestions))

    # Reflection and quality check
    workflow.add_node("reflect", lambda s: reflect_on_progress(s))

    # Error recovery and fallback
    workflow.add_node("fallback", lambda s: {
        "normalized_suggestions": {},
        "errors": s.errors + ["Normalization failed - using fallback processing"],
        "reflections": s.reflections + ["Applied fallback strategy due to processing issues"]
    })

    # Final conversion with quality validation
    workflow.add_node("convert", lambda s: {
        "rule_suggestions": convert_to_rule_ms_format.invoke({"rule_input": {"suggestions": s.normalized_suggestions}}),
        "execution_metrics": {
            **s.execution_metrics,
            "total_execution_time": time.time() - s.execution_start_time,
            "steps_completed": len(s.step_history),
            "final_rule_count": len(s.rule_suggestions) if s.rule_suggestions else 0
        }
    })

    # Build the workflow
    workflow.set_entry_point("create_plan")
    
    # Sequential ReAct flow
    workflow.add_edge("create_plan", "reason_fetch")
    workflow.add_edge("reason_fetch", "fetch_rules")
    workflow.add_edge("fetch_rules", "observe_fetch")
    workflow.add_edge("observe_fetch", "reason_suggest")
    workflow.add_edge("reason_suggest", "suggest")
    workflow.add_edge("suggest", "observe_suggest")
    workflow.add_edge("observe_suggest", "reason_format")
    workflow.add_edge("reason_format", "format")
    workflow.add_edge("format", "observe_format")
    workflow.add_edge("observe_format", "reason_normalize")
    workflow.add_edge("reason_normalize", "normalize")
    workflow.add_edge("normalize", "observe_normalize")
    workflow.add_edge("observe_normalize", "reflect")

    # Conditional routing based on normalization success
    workflow.add_conditional_edges(
        "reflect",
        lambda s: (
            "convert" if (
                isinstance(s.normalized_suggestions, dict) and 
                any(isinstance(v, dict) and v.get("expectations") for v in s.normalized_suggestions.values())
            ) else "fallback"
        ),
        {
            "convert": "convert",
            "fallback": "fallback"
        }
    )

    workflow.add_edge("fallback", "convert")

    return workflow.compile()

def _setup_validation_context(domain: str, schema: dict) -> Optional[AgentValidationContext]:
    """Set up validation context if enabled"""
    if not settings.llm_validation_enabled:
        logger.info(" LLM validation disabled")
        return None
        
    validation_config = settings.get_llm_validation_config()
    user_id = f"agent_{domain}_{hash(str(schema)) % 10000}"
    context = AgentValidationContext(user_id, validation_config)
    logger.info(" LLM validation enabled for agent workflow")
    return context

def _extract_results(result: Union[AgentState, dict]) -> Tuple[List[Any], List[str], List[str], List[str], Dict[str, Any], List[AgentStep]]:
    """Extract and normalize results from either AgentState or dict"""
    if isinstance(result, dict):
        logger.warning("LangGraph returned dict instead of AgentState")
        return (
            result.get("rule_suggestions", []),
            result.get("thoughts", []),
            result.get("observations", []),
            result.get("reflections", []),
            result.get("execution_metrics", {}),
            result.get("step_history", [])
        )
    
    return (
        result.rule_suggestions or [],
        result.thoughts,
        result.observations,
        result.reflections,
        result.execution_metrics,
        result.step_history
    )

def _log_agent_progress(thoughts: List[str], observations: List[str], reflections: List[str], 
                       rule_suggestions: List[Any], validation_metrics: Optional[Dict] = None):
    """Log agent progress and insights"""
    logger.info(f" Agent completed {len(thoughts)} reasoning steps")
    logger.info(f" Agent made {len(observations)} observations")
    logger.info(f" Agent performed {len(reflections)} reflection cycles")
    logger.info(f" Generated {len(rule_suggestions)} rule suggestions")
    
    if validation_metrics:
        logger.info(f" Validation metrics: {validation_metrics}")
    
    # Log sample of thoughts and reflections
    if thoughts:
        logger.info(" Key agent thoughts:")
        for i, thought in enumerate(thoughts[:3], 1):
            logger.info(f"   {i}. {thought[:100]}...")
            
    if reflections:
        logger.info(" Agent reflections:")
        for i, reflection in enumerate(reflections[-2:], 1):
            logger.info(f"   {i}. {reflection[:100]}...")

def _create_validation_response(domain: str, rule_suggestions: List[Any], thoughts: List[str],
                              observations: List[str], reflections: List[str], execution_metrics: Dict,
                              step_history: List[AgentStep]) -> Dict[str, Any]:
    """Create validation response with agent context"""
    return {
        "domain": domain,
        "rules": rule_suggestions,
        "explanation": "Enhanced AI agent generated rules with reasoning and reflection",
        "agent_metadata": {
            "reasoning_steps": len(thoughts),
            "observations": len(observations),
            "reflections": len(reflections),
            "execution_time": execution_metrics.get("total_execution_time", 0),
            "steps_completed": len(step_history)
        }
    }

def _handle_agent_error(e: Exception, domain: str, validation_start_time: float):
    """Handle agent execution error and record metrics"""
    logger.error(f" Enhanced agent execution failed for domain {domain}: {e}")
    validation_time_ms = (time.time() - validation_start_time) * 1000
    
    try:
        from app.validation.llm_validator import ValidationResult, ValidationIssue, ValidationSeverity
        
        failed_result = ValidationResult(
            is_valid=False,
            confidence_score=0.0,
            issues=[ValidationIssue(
                field="enhanced_agent_execution",
                severity=ValidationSeverity.CRITICAL,
                message=f"Enhanced agent workflow failed: {str(e)}",
                suggestion="Check logs, validate configuration, and retry with simpler approach"
            )],
            corrected_data=None,
            metadata={
                "error": str(e),
                "agent_type": "enhanced_react",
                "failure_stage": "execution"
            }
        )
        
        record_validation_metric(
            domain=domain,
            response_type="rule",
            validation_result=failed_result,
            validation_time_ms=validation_time_ms,
            metadata={"failure_type": "agent_execution_error"}
        )
    except Exception as metrics_error:
        logger.warning(f"Failed to record failure metrics: {metrics_error}")

def run_agent(schema: dict) -> List[Dict[str, Any]]:
    """
    Run the enhanced agentic rule suggestion workflow with ReAct pattern,
    planning, reflection, validation, and comprehensive monitoring
    
    Args:
        schema: Domain schema dictionary
        
    Returns:
        List of validated rule suggestions with enhanced metadata
    """
    validation_start_time = time.time()
    domain = schema.get("domain", "unknown")
    
    try:
        # Setup validation
        validation_context = _setup_validation_context(domain, schema)
        
        with validation_context if validation_context else None as validator:
            logger.info(f"🤖 Starting enhanced agentic workflow for domain: {domain}")
            
            # Initialize and run
            initial_state = AgentState(data_schema=schema)
            if validator:
                initial_state.metadata["validation_context"] = validator
            
            graph = build_graph()
            result = graph.invoke(initial_state)

            # Process results
            rule_suggestions, thoughts, observations, reflections, execution_metrics, step_history = _extract_results(result)
            
            # Log progress
            _log_agent_progress(
                thoughts, observations, reflections, rule_suggestions,
                validator.get_metrics() if validator else None
            )

            # Validate if there are suggestions
            if rule_suggestions:
                validation_response = _create_validation_response(
                    domain, rule_suggestions, thoughts, observations,
                    reflections, execution_metrics, step_history
                )
                
                validation_result = validate_llm_response(
                    response=validation_response,
                    response_type="rule",
                    strict_mode=False,
                    auto_correct=True
                )
                
                # Record metrics
                try:
                    record_validation_metric(
                        domain=domain,
                        response_type="rule",
                        validation_result=validation_result,
                        validation_time_ms=(time.time() - validation_start_time) * 1000,
                        metadata={
                            "agent_type": "enhanced_react",
                            "reasoning_steps": len(thoughts),
                            "reflection_cycles": len(reflections),
                            "total_execution_time": execution_metrics.get("total_execution_time", 0)
                        }
                    )
                except Exception as e:
                    logger.warning(f"Failed to record enhanced validation metrics: {e}")
                
                # Log validation results
                if validation_result.issues:
                    logger.warning(f" Validation found {len(validation_result.issues)} issues:")
                    for issue in validation_result.issues:
                        logger.warning(f"   {issue.severity.value.upper()}: {issue.field} - {issue.message}")
                
                logger.info(f" Validation confidence: {validation_result.confidence_score:.2f}")
                logger.info(f" Total execution time: {execution_metrics.get('total_execution_time', 0):.2f}s")
                
                # Use corrected rules if available
                if validation_result.corrected_data and validation_result.corrected_data.get("rules"):
                    logger.info(" Using auto-corrected rule suggestions")
                    rule_suggestions = validation_result.corrected_data["rules"]
            
            return rule_suggestions
            
    except Exception as e:
        _handle_agent_error(e, domain, validation_start_time)
        return []
        
        return []

def generate_agent_report(state: AgentState) -> str:
    """Generate a comprehensive report of the agent's reasoning process"""
    report = ["# Enhanced Agent Execution Report\n"]
    
    # Executive Summary
    domain = state.data_schema.get("domain", "unknown")
    column_count = len([k for k in state.data_schema.keys() if k != "domain"])
    rule_count = len(state.rule_suggestions) if state.rule_suggestions else 0
    
    report.append(f"**Domain**: {domain}")
    report.append(f"**Columns Analyzed**: {column_count}")
    report.append(f"**Rules Generated**: {rule_count}")
    report.append(f"**Execution Time**: {state.execution_metrics.get('total_execution_time', 0):.2f}s\n")
    
    # Planning Phase
    if state.plan:
        report.append("##  Agent Planning")
        report.append(f"**Goal**: {state.plan.goal}")
        report.append(f"**Plan Type**: {state.plan.context.get('plan_type', 'unknown')}")
        report.append("**Planned Steps**:")
        for i, step in enumerate(state.plan.steps, 1):
            report.append(f"  {i}. {step}")
        report.append("")
    
    # Reasoning Chain
    if state.thoughts:
        report.append("##  Reasoning Chain")
        for i, thought in enumerate(state.thoughts, 1):
            report.append(f"**Step {i}**: {thought}")
        report.append("")
    
    # Observations
    if state.observations:
        report.append("##  Observations")
        for i, observation in enumerate(state.observations, 1):
            report.append(f"**Observation {i}**: {observation}")
        report.append("")
    
    # Reflections
    if state.reflections:
        report.append("##  Reflections")
        for i, reflection in enumerate(state.reflections, 1):
            report.append(f"**Reflection {i}**: {reflection}")
        report.append("")
    
    # Quality Metrics
    if state.quality_metrics:
        report.append("##  Quality Metrics")
        for metric, value in state.quality_metrics.items():
            if isinstance(value, float):
                report.append(f"- **{metric}**: {value:.3f}")
            else:
                report.append(f"- **{metric}**: {value}")
        report.append("")
    
    # Error Summary
    if state.errors:
        report.append("##  Issues Encountered")
        for i, error in enumerate(state.errors, 1):
            report.append(f"{i}. {error}")
        report.append("")
    
    return "\n".join(report)
