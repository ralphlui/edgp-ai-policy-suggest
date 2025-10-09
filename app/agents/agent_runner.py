from typing import Dict, List, Any, Optional
from pydantic import BaseModel
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
import logging
import time

logger = logging.getLogger(__name__)

class AgentState(BaseModel):
    data_schema: Dict[str, Any]
    gx_rules: Optional[List[Any]] = None
    raw_suggestions: Optional[str] = None
    formatted_rules: Optional[List[Any]] = None
    normalized_suggestions: Optional[Dict[str, Any]] = None
    rule_suggestions: Optional[Annotated[List[Dict[str, Any]], "last"]] = None

def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("fetch_rules", lambda s: {
        "gx_rules": fetch_gx_rules.invoke({"query": ""})
    })

    workflow.add_node("suggest", lambda s: {
        "raw_suggestions": suggest_column_rules.invoke({
            "data_schema": s.data_schema,
            "gx_rules": s.gx_rules
        })
    })

    workflow.add_node("format", lambda s: {
        "formatted_rules": format_gx_rules.invoke(s.raw_suggestions)
    })

    workflow.add_node("normalize", lambda s: {
        "normalized_suggestions": normalize_rule_suggestions.invoke({"rule_input": {"raw": s.formatted_rules}})
    })

    workflow.add_node("fallback", lambda s: {
        "normalized_suggestions": {}
    })

    workflow.add_node("convert", lambda s: {
    "rule_suggestions": convert_to_rule_ms_format.invoke({"rule_input": {"suggestions": s.normalized_suggestions}})
    })


    workflow.set_entry_point("fetch_rules")
    workflow.add_edge("fetch_rules", "suggest")
    workflow.add_edge("suggest", "format")
    workflow.add_edge("format", "normalize")

    workflow.add_conditional_edges(
        "normalize",
        lambda s: any(
            isinstance(v, dict) and v.get("expectations")
            for v in s.normalized_suggestions.values()
        ) if isinstance(s.normalized_suggestions, dict) else False,
        {
            True: "convert",
            False: "fallback"
        }
    )

    workflow.add_edge("fallback", "convert")

    return workflow.compile()

def run_agent(schema: dict) -> List[Dict[str, Any]]:
    """
    Run the rule suggestion agent with enhanced validation and metrics tracking
    
    Args:
        schema: Domain schema dictionary
        
    Returns:
        List of validated rule suggestions
    """
    validation_start_time = time.time()
    domain = schema.get("domain", "unknown")
    
    try:
        graph = build_graph()
        result = graph.invoke(AgentState(data_schema=schema))

        if isinstance(result, dict):
            logger.warning("LangGraph returned dict instead of AgentState")
            rule_suggestions = result.get("rule_suggestions", [])
        else:
            rule_suggestions = result.rule_suggestions or []

        logger.info("Generated rule suggestions: %s", rule_suggestions)
        
        # Validate the generated rules using the new validation system
        if rule_suggestions:
            # Convert to expected validation format
            validation_response = {
                "domain": domain,
                "rules": rule_suggestions,
                "explanation": "Generated rule suggestions for domain validation"
            }
            
            validation_result = validate_llm_response(
                response=validation_response,
                response_type="rule",
                strict_mode=False,  # Use lenient mode for rule validation
                auto_correct=True
            )
            
            # Record validation metrics
            validation_time_ms = (time.time() - validation_start_time) * 1000
            try:
                record_validation_metric(
                    domain=domain,
                    response_type="rule",
                    validation_result=validation_result,
                    validation_time_ms=validation_time_ms
                )
            except Exception as e:
                logger.warning(f"Failed to record rule validation metrics: {e}")
            
            # Log validation results
            if validation_result.issues:
                logger.warning(f"Rule validation found {len(validation_result.issues)} issues:")
                for issue in validation_result.issues:
                    logger.warning(f"  {issue.severity.value.upper()}: {issue.field} - {issue.message}")
            
            logger.info(f"Rule validation confidence score: {validation_result.confidence_score}")
            
            # Use corrected rules if available
            if validation_result.corrected_data and validation_result.corrected_data.get("rules"):
                logger.info("Using auto-corrected rule suggestions")
                rule_suggestions = validation_result.corrected_data["rules"]
        
        return rule_suggestions
        
    except Exception as e:
        logger.error(f"Rule generation failed for domain {domain}: {e}")
        
        # Record failed validation metrics
        validation_time_ms = (time.time() - validation_start_time) * 1000
        try:
            from app.validation.llm_validator import ValidationResult, ValidationIssue, ValidationSeverity
            
            failed_result = ValidationResult(
                is_valid=False,
                confidence_score=0.0,
                issues=[ValidationIssue(
                    field="rule_generation",
                    severity=ValidationSeverity.CRITICAL,
                    message=f"Rule generation failed: {str(e)}",
                    suggestion="Check logs and retry"
                )],
                corrected_data=None,
                metadata={"error": str(e)}
            )
            
            record_validation_metric(
                domain=domain,
                response_type="rule",
                validation_result=failed_result,
                validation_time_ms=validation_time_ms
            )
        except Exception as metrics_error:
            logger.warning(f"Failed to record failure metrics: {metrics_error}")
        
        return []
