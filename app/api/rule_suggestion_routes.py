from fastapi import APIRouter, Body, Depends, Request
from fastapi.responses import JSONResponse
from app.auth.authentication import verify_any_scope_token, UserInfo
from app.agents.agent_runner import run_agent
from app.agents.schema_suggester import bootstrap_schema_for_domain
from app.vector_db.schema_loader import get_schema_by_domain
import logging
import traceback

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/aips/rule", tags=["rule-suggestions"])


def _calculate_overall_confidence(state) -> float:
    """Calculate overall confidence score based on various factors"""
    factors = []
    
    # Factor 1: Error rate (fewer errors = higher confidence)
    error_rate = len(state.errors) / max(len(state.step_history), 1)
    error_confidence = max(0, 1.0 - error_rate)
    factors.append(error_confidence)
    
    # Factor 2: Execution completeness
    if state.execution_metrics:
        total_time = state.execution_metrics.get("total_execution_time", 0)
        if total_time > 0:
            # Reasonable execution time suggests good processing
            time_confidence = min(1.0, max(0.5, 1.0 - (total_time - 3.0) / 10.0))
            factors.append(time_confidence)
    
    # Factor 3: Rule generation success
    rule_count = len(state.rule_suggestions or [])
    if rule_count > 0:
        # More rules generated typically indicates better analysis
        rule_confidence = min(1.0, 0.6 + (rule_count / 20.0))
        factors.append(rule_confidence)
    
    # Factor 4: Schema complexity handling
    if state.data_schema:
        column_count = len(state.data_schema) - 1  # Subtract 'domain' key
        if column_count > 0:
            # Confidence decreases slightly with complexity but not dramatically
            complexity_confidence = max(0.7, 1.0 - (column_count - 5) / 20.0)
            factors.append(complexity_confidence)
    
    # Return average of all factors, defaulting to 0.5 if no factors
    return sum(factors) / len(factors) if factors else 0.5


def _get_confidence_level(state) -> str:
    """Get human-readable confidence level"""
    overall = _calculate_overall_confidence(state)
    
    if overall >= 0.8:
        return "high"
    elif overall >= 0.6:
        return "medium"
    elif overall >= 0.4:
        return "low"
    else:
        return "very_low"


def _get_confidence_factors(state) -> dict:
    """Get detailed breakdown of confidence factors"""
    rule_count = len(state.rule_suggestions or [])
    error_count = len(state.errors)
    execution_time = state.execution_metrics.get("total_execution_time", 0) if state.execution_metrics else 0
    
    return {
        "rule_generation": {
            "rules_generated": rule_count,
            "score": min(1.0, 0.6 + (rule_count / 20.0)) if rule_count > 0 else 0.0,
            "status": "good" if rule_count >= 5 else "needs_review" if rule_count > 0 else "failed"
        },
        "error_handling": {
            "errors_encountered": error_count,
            "score": max(0, 1.0 - error_count / max(len(state.step_history), 1)),
            "status": "good" if error_count == 0 else "issues" if error_count < 3 else "problematic"
        },
        "execution_performance": {
            "duration_seconds": round(execution_time, 2),
            "score": min(1.0, max(0.5, 1.0 - (execution_time - 3.0) / 10.0)) if execution_time > 0 else 0.5,
            "status": "fast" if execution_time < 3 else "normal" if execution_time < 8 else "slow"
        },
        "reasoning_depth": {
            "thoughts_generated": len(state.thoughts),
            "observations_made": len(state.observations),
            "reflections_completed": len(state.reflections),
            "score": min(1.0, (len(state.thoughts) + len(state.observations) + len(state.reflections)) / 15.0),
            "status": "thorough" if len(state.thoughts) >= 5 else "adequate" if len(state.thoughts) >= 2 else "minimal"
        }
    }


@router.post("/suggest")
async def suggest_rules(
    domain: str = Body(..., embed=True),
    user: UserInfo = Depends(verify_any_scope_token),
    include_insights: bool = Body(True, embed=True)  # Optional parameter for insights (default: True)
):
    try:
        # Log authenticated user information
        logger.info(f" Suggest rules request from user: {user.email} with scopes: {user.scopes}")
        
        # Try to get schema from vector database
        logger.info(f"Attempting to retrieve schema for domain: {domain}")
        
        # First, test vector DB connectivity
        vector_db_status = "unknown"
        connection_error = None
        
        try:
            schema = get_schema_by_domain(domain)
            
            # If schema is empty but no exception, try with forced refresh for newly created domains
            if not schema:
                logger.info(f"Schema not found for domain {domain}, attempting optimized refresh...")
                from app.vector_db.schema_loader import get_store
                store = get_store()
                if store:
                    # Force refresh to pick up recently created domains
                    store.force_refresh_index()
                    # Reduced wait time for better performance
                    import asyncio
                    await asyncio.sleep(0.1)  # Reduced from 0.5 to 0.1 seconds
                    # Try again
                    schema = get_schema_by_domain(domain)
                    if schema:
                        logger.info(f" Schema found for domain {domain} after optimized refresh")
            
            vector_db_status = "connected"
            logger.info(f"Vector DB connection successful. Schema retrieval result for domain {domain}: {schema}")
        except Exception as db_error:
            connection_error = str(db_error)
            schema = None
            # Check if it's a connection/auth issue vs schema not found
            if "AuthorizationException" in connection_error or "RetryError" in connection_error or "ConnectionError" in connection_error:
                vector_db_status = "connection_failed"
                logger.error(f"Vector DB connection failed for domain {domain}: {connection_error}")
            else:
                vector_db_status = "accessible_but_error"
                logger.warning(f"Vector DB accessible but error occurred for domain {domain}: {connection_error}")

        if schema:
            logger.info(f"Successfully retrieved schema for domain {domain}, generating rules")
            
            if include_insights:
                # Use enhanced agent with full insights
                from app.agents.agent_runner import AgentState, build_graph
                initial_state = AgentState(data_schema=schema)
                graph = build_graph()
                result = graph.invoke(initial_state)
                
                if isinstance(result, dict):
                    state = AgentState(**result)
                else:
                    state = result
                    
                return { 
                    "rule_suggestions": state.rule_suggestions or [],
                    "confidence": {
                        "overall": _calculate_overall_confidence(state),
                        "level": _get_confidence_level(state),
                        "factors": _get_confidence_factors(state)
                    },
                    "agent_insights": {
                        "reasoning_steps": len(state.thoughts),
                        "observations": len(state.observations),
                        "reflections": len(state.reflections),
                        "execution_time": state.execution_metrics.get("total_execution_time", 0),
                        "detailed_confidence_scores": state.confidence_scores,  # For debugging
                        "key_thoughts": state.thoughts[-3:] if state.thoughts else [],  # Last 3 thoughts
                        "final_reflection": state.reflections[-1] if state.reflections else None
                    }
                }
            else:
                # Standard agent execution (backwards compatible)
                rule_suggestions = run_agent(schema)
                return { "rule_suggestions": rule_suggestions }

        # Handle different scenarios when no schema is found
        if vector_db_status == "connection_failed":
            # Case 1: Cannot connect to vector DB at all - Don't generate LLM schema
            return JSONResponse({
                "error": "Vector database connection failed",
                "alert": {
                    "type": "error",
            
                    "message": "Cannot connect to the vector database. Please check your connection and try again.",
                    "details": "Vector database is currently unavailable. Unable to retrieve or create schemas."
                },
                "domain": domain,
                "error_type": "connection_failed"
            }, status_code=503)
        
        else:
            # Case 2: Can connect to vector DB but schema doesn't exist for this domain
            # Only now generate LLM-suggested column names (no data types)
            logger.info(f"Vector DB accessible but no schema found. Generating column name suggestions for domain: {domain}")
            suggested_schema = bootstrap_schema_for_domain(domain)
            
            # Create column names only (no data types) when domain not found in vector DB
            suggested_column_names = list(suggested_schema.keys())
            
            return JSONResponse({
                "error": "Domain not found",
                "alert": {
                    "type": "confirmation",
                    "message": f"No schema exists for domain '{domain}'. Would you like to create one using our AI suggestions?",
                    "details": f"We've generated {len(suggested_column_names)} suggested column names for this domain."
                },
                "domain": domain,
                "suggested_columns": suggested_column_names,
                "actions": {
                    "note": "Column names only suggested. Data types will be inferred from actual CSV data.",
                    "create_schema_with_csv": {
                        "description": "Use suggested column names to create schema from CSV",
                        "endpoint": "/api/aips/create/domain",
                        "method": "POST",
                        "payload": {
                            "domain": domain,
                            "columns": suggested_column_names,
                            "return_csv": True
                        }
                    },
                    "create_schema_only": {
                        "description": "Use suggested column names to create schema",
                        "endpoint": "/api/aips/create/domain",
                        "method": "POST",
                        "payload": {
                            "domain": domain,
                            "columns": suggested_column_names,
                            "return_csv": False
                        }
                    }
                },
                "next_steps": {
                    "after_creation": [
                        "Schema will be saved to vector database",
                        "Optional: Download sample CSV data",
                        "Call /api/aips/rule/suggest again to get validation rules"
                    ]
                }
            }, status_code=404)

    except Exception as e:
        # This catches any unexpected errors not handled above
        error_message = str(e)
        logger.error(f"Unexpected error in suggest_rules for domain {domain}: {error_message}")
        traceback.print_exc()
        
        return JSONResponse({
            "error": "Internal server error",
            "alert": {
             
                "message": f"An unexpected error occurred while processing domain '{domain}'",
                "details": "Please try again or contact support if the issue persists."
            },
            "domain": domain
        }, status_code=500)
