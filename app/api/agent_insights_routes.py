"""
Enhanced Agent Insights and Monitoring API

Provides detailed visibility into agent reasoning, planning, and execution
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from typing import Dict, Any, List, Optional
import logging
import time
from app.agents.agent_runner import (
    run_agent, 
    AgentState, 
    build_graph, 
    generate_agent_report,
    create_agent_plan
)

logger = logging.getLogger(__name__)

SCHEMA_REQUIRED_ERROR = "Schema is required"

router = APIRouter(
    prefix="/api/aips/agent",
    tags=["agent-insights"],
    responses={404: {"description": "Not found"}},
)

@router.post("/analyze-with-insights")
async def analyze_schema_with_insights(request_data: Dict[str, Any]):
    """
    Run agent analysis with full visibility into reasoning process
    """
    try:
        schema = request_data.get("schema")
        if not schema:
            raise HTTPException(status_code=400, detail=SCHEMA_REQUIRED_ERROR)
        
        # Initialize enhanced state for tracking
        initial_state = AgentState(data_schema=schema)
        
        # Execute the enhanced agent
        graph = build_graph()
        result = graph.invoke(initial_state)
        
        # Extract comprehensive results
        if isinstance(result, dict):
            state = AgentState(**result)
        else:
            state = result
            
        # Generate detailed response
        response = {
            "rule_suggestions": state.rule_suggestions or [],
            "agent_insights": {
                "reasoning_chain": state.thoughts,
                "observations": state.observations,
                "reflections": state.reflections,
                "execution_plan": {
                    "goal": state.plan.goal if state.plan else None,
                    "steps": state.plan.steps if state.plan else [],
                    "progress": len(state.step_history)
                },
                "quality_metrics": state.quality_metrics,
                "execution_time": state.execution_metrics.get("total_execution_time", 0),
                "confidence_scores": state.confidence_scores
            },
            "step_by_step_trace": [
                {
                    "step_id": step.step_id,
                    "action": step.action,
                    "thought": step.thought,
                    "observation": step.observation,
                    "reflection": step.reflection,
                    "timestamp": step.timestamp,
                    "metadata": step.metadata
                }
                for step in state.step_history
            ],
            "errors": state.errors,
            "domain": schema.get("domain", "unknown"),
            "summary": {
                "total_steps": len(state.step_history),
                "reasoning_cycles": len(state.thoughts),
                "observations_made": len(state.observations),
                "reflection_cycles": len(state.reflections),
                "rules_generated": len(state.rule_suggestions) if state.rule_suggestions else 0,
                "success_rate": 1.0 - (len(state.errors) / max(len(state.step_history), 1))
            }
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Agent insights analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.post("/plan-analysis")
async def create_analysis_plan(request_data: Dict[str, Any]):
    """
    Generate an execution plan for schema analysis without running it
    """
    try:
        schema = request_data.get("schema")
        if not schema:
            raise HTTPException(status_code=400, detail=SCHEMA_REQUIRED_ERROR)
            
        plan = create_agent_plan(schema)
        
        return {
            "plan": {
                "goal": plan.goal,
                "steps": plan.steps,
                "context": plan.context,
                "constraints": plan.constraints,
                "estimated_duration": plan.context.get("estimated_duration", 0)
            },
            "schema_analysis": {
                "column_count": len([k for k in schema.keys() if k != "domain"]),
                "domain": schema.get("domain", "unknown"),
                "complexity": plan.context.get("plan_type", "unknown")
            }
        }
        
    except Exception as e:
        logger.error(f"Plan creation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Plan creation failed: {str(e)}")

@router.post("/trace-execution")
async def trace_agent_execution(request_data: Dict[str, Any]):
    """
    Execute agent with detailed step-by-step tracing
    """
    try:
        schema = request_data.get("schema")
        if not schema:
            raise HTTPException(status_code=400, detail=SCHEMA_REQUIRED_ERROR)
            
        # Execute with full tracing
        graph = build_graph()
        
        # Add execution tracing
        execution_trace = []
        start_time = time.time()
        
        result = graph.invoke(AgentState(data_schema=schema))
        
        if isinstance(result, dict):
            state = AgentState(**result)
        else:
            state = result
            
        total_time = time.time() - start_time
        
        return {
            "execution_summary": {
                "total_time": total_time,
                "steps_executed": len(state.step_history),
                "success": len(state.errors) == 0,
                "rules_generated": len(state.rule_suggestions) if state.rule_suggestions else 0
            },
            "detailed_trace": [
                {
                    "step": i + 1,
                    "step_id": step.step_id,
                    "action": step.action,
                    "thought": step.thought[:200] + "..." if len(step.thought) > 200 else step.thought,
                    "observation": step.observation[:200] + "..." if len(step.observation) > 200 else step.observation,
                    "duration": step.metadata.get("duration", 0),
                    "timestamp": step.timestamp
                }
                for i, step in enumerate(state.step_history)
            ],
            "final_state": {
                "rule_count": len(state.rule_suggestions) if state.rule_suggestions else 0,
                "error_count": len(state.errors),
                "reflection_count": len(state.reflections),
                "quality_score": state.quality_metrics.get("progress", 0)
            }
        }
        
    except Exception as e:
        logger.error(f"Execution tracing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Tracing failed: {str(e)}")

@router.post("/generate-report")
async def generate_execution_report(request_data: Dict[str, Any]):
    """
    Generate a comprehensive markdown report of agent execution
    """
    try:
        schema = request_data.get("schema")
        if not schema:
            raise HTTPException(status_code=400, detail=SCHEMA_REQUIRED_ERROR)
            
        # Execute agent
        graph = build_graph()
        result = graph.invoke(AgentState(data_schema=schema))
        
        if isinstance(result, dict):
            state = AgentState(**result)
        else:
            state = result
            
        # Generate detailed report
        report = generate_agent_report(state)
        
        return PlainTextResponse(
            content=report,
            media_type="text/markdown",
            headers={"Content-Disposition": "attachment; filename=agent_execution_report.md"}
        )
        
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")

@router.get("/capabilities")
async def get_agent_capabilities():
    """
    Get information about agent capabilities and features
    """
    return {
        "agent_type": "Enhanced ReAct Agent",
        "version": "2.0",
        "capabilities": {
            "reasoning": {
                "description": "Multi-step reasoning with thought chains",
                "features": ["plan_creation", "step_by_step_thinking", "context_awareness"]
            },
            "acting": {
                "description": "Structured action execution with validation",
                "features": ["tool_invocation", "error_handling", "result_validation"]
            },
            "observing": {
                "description": "Comprehensive observation and monitoring",
                "features": ["result_analysis", "progress_tracking", "quality_assessment"]
            },
            "reflecting": {
                "description": "Self-assessment and strategy adjustment",
                "features": ["progress_evaluation", "error_analysis", "strategy_adaptation"]
            },
            "planning": {
                "description": "Intelligent execution planning based on complexity",
                "features": ["complexity_assessment", "step_sequencing", "resource_estimation"]
            }
        },
        "supported_domains": ["customer", "finance", "product", "vendor", "generic"],
        "output_formats": ["structured_rules", "execution_report", "step_trace", "quality_metrics"],
        "monitoring": {
            "real_time_insights": True,
            "execution_tracing": True,
            "quality_metrics": True,
            "error_recovery": True
        }
    }

@router.get("/health")
async def agent_health_check():
    """
    Health check for the enhanced agent system
    """
    try:
        # Test basic agent functionality
        test_schema = {"test_column": {"dtype": "string", "sample_values": ["test"]}}
        plan = create_agent_plan(test_schema)
        
        return {
            "status": "healthy",
            "agent_system": "operational",
            "features": {
                "planning": True,
                "reasoning": True,
                "reflection": True,
                "monitoring": True
            },
            "test_results": {
                "plan_creation": plan is not None,
                "plan_steps": len(plan.steps) if plan else 0
            },
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Agent health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.time()
            }
        )