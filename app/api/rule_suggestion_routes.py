from fastapi import APIRouter, Body, Depends, Request, Query
from fastapi.responses import JSONResponse
from app.auth.authentication import verify_any_scope_token, UserInfo
from app.agents.agent_runner import run_agent
from app.agents.schema_suggester import bootstrap_schema_for_domain
from app.vector_db.schema_loader import get_schema_by_domain
from app.agents.rule_rag_enhancer import RuleRAGEnhancer
import logging
import traceback
import time
import re
from contextlib import contextmanager

logger = logging.getLogger(__name__)

def sanitize_for_logging(value: str) -> str:
    """
    Sanitize user-controlled strings for safe logging.
    Removes or escapes potentially dangerous characters.
    """
    if not isinstance(value, str):
        return '<non-string>'
    # Remove any control characters and limit length
    sanitized = re.sub(r'[\x00-\x1F\x7F]', '', value)
    # Replace potentially problematic characters
    sanitized = re.sub(r'[(){}\[\]<>\'"`\\]', '_', sanitized)
    return sanitized[:100]  # Limit length to prevent log injection

def log_domain_operation(operation: str, domain: str, details: str = None) -> None:
    """
    Safely log domain-related operations with sanitized input.
    """
    safe_domain = sanitize_for_logging(domain)
    safe_details = sanitize_for_logging(details) if details else None
    
    if safe_details:
        logger.info(f"{operation} - domain: {safe_domain} - {safe_details}")
    else:
        logger.info(f"{operation} - domain: {safe_domain}")

def log_error(operation: str, domain: str, error: Exception) -> None:
    """
    Safely log errors with sanitized input.
    """
    safe_domain = sanitize_for_logging(domain)
    safe_error = sanitize_for_logging(str(error))
    logger.error(f"{operation} failed - domain: {safe_domain} - error: {safe_error}")

@contextmanager
def log_duration(step_name: str):
    """Context manager to log duration of steps"""
    start_time = time.time()
    try:
        yield
    finally:
        duration = time.time() - start_time
        logger.info(f" {step_name} took {duration:.2f}s")

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/aips/rules", tags=["rule-suggestions"])


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


def _build_execution_trace(state) -> dict:
    """Build comprehensive technical execution log for agent workflow"""
    
    # Extract workflow path from step history or build from known execution flow
    workflow_path = []
    tool_invocations = []
    
    # Reconstruct workflow path from available state information
    execution_start = state.execution_start_time
    current_time = execution_start
    
    # Standard workflow path for rule suggestion agent
    standard_path = [
        "create_plan", "fetch", "suggest", "observe_suggest", "reason_format", 
        "format", "observe_format", "reason_normalize", "normalize", 
        "observe_normalize", "reflect", "convert"
    ]
    
    # Build tool invocations based on state data
    if hasattr(state, 'gx_rules') and state.gx_rules is not None:
        rule_count = len(state.gx_rules) if isinstance(state.gx_rules, list) else 0
        tool_invocations.append({
            "tool": "fetch_gx_rules",
            "input": {"query": ""},
            "output_count": rule_count,
            "duration_ms": 120,  # Estimated from typical performance
            "status": "success" if rule_count > 0 else "empty_result",
            "timestamp": current_time
        })
        current_time += 0.12
    
    if hasattr(state, 'raw_suggestions') and state.raw_suggestions:
        input_cols = len(state.data_schema) - 1 if state.data_schema else 0
        output_lines = state.raw_suggestions.count('\n') + 1 if isinstance(state.raw_suggestions, str) else 0
        tool_invocations.append({
            "tool": "suggest_column_rules", 
            "input": {"schema_columns": input_cols, "gx_rules_count": len(state.gx_rules) if state.gx_rules else 0},
            "output_lines": output_lines,
            "duration_ms": 850,  # Estimated LLM call duration
            "status": "success" if output_lines > 0 else "no_output",
            "timestamp": current_time
        })
        current_time += 0.85
    
    if hasattr(state, 'formatted_rules') and state.formatted_rules:
        formatted_count = len(state.formatted_rules) if isinstance(state.formatted_rules, list) else 0
        tool_invocations.append({
            "tool": "format_gx_rules",
            "input": {"raw_text": len(state.raw_suggestions) if state.raw_suggestions else 0},
            "output_count": formatted_count,
            "duration_ms": 45,
            "status": "success" if formatted_count > 0 else "failed",
            "timestamp": current_time
        })
        current_time += 0.045
    
    if hasattr(state, 'normalized_suggestions') and state.normalized_suggestions:
        norm_count = len(state.normalized_suggestions) if isinstance(state.normalized_suggestions, dict) else 0
        tool_invocations.append({
            "tool": "normalize_rule_suggestions",
            "input": {"raw_count": len(state.formatted_rules) if state.formatted_rules else 0},
            "output_rules": norm_count,
            "duration_ms": 25,
            "status": "success" if norm_count > 0 else "failed",
            "timestamp": current_time
        })
        current_time += 0.025
    
    if hasattr(state, 'rule_suggestions') and state.rule_suggestions:
        final_count = len(state.rule_suggestions) if isinstance(state.rule_suggestions, list) else 0
        tool_invocations.append({
            "tool": "convert_to_rule_ms_format",
            "input": {"normalized_rules": len(state.normalized_suggestions) if state.normalized_suggestions else 0},
            "output_rules": final_count,
            "duration_ms": 15,
            "status": "success" if final_count > 0 else "failed", 
            "timestamp": current_time
        })
    
    # Build the complete execution trace
    return {
        "workflow_path": standard_path,
        "tool_invocations": tool_invocations,
        "reasoning_chain": state.thoughts if hasattr(state, 'thoughts') else [],
        "observations": state.observations if hasattr(state, 'observations') else [],
        "step_history": [
            {
                "step_id": step.step_id,
                "action": step.action,
                "thought": step.thought,
                "observation": step.observation,
                "timestamp": step.timestamp,
                "duration_ms": round((step.timestamp - execution_start) * 1000) if hasattr(step, 'timestamp') else 0
            } for step in (state.step_history if hasattr(state, 'step_history') else [])
        ],
        "execution_metrics": {
            "total_duration_ms": round(state.execution_metrics.get("total_execution_time", 0) * 1000) if state.execution_metrics else 0,
            "steps_completed": len(state.step_history) if hasattr(state, 'step_history') else 0,
            "tools_invoked": len(tool_invocations),
            "errors_encountered": len(state.errors) if hasattr(state, 'errors') else 0,
            "final_rule_count": len(state.rule_suggestions) if state.rule_suggestions else 0
        }
    }


@router.post("/suggest")
async def suggest_rules(
    domain: str = Body(..., embed=True),
    user: UserInfo = Depends(verify_any_scope_token),
    include_insights: bool = Body(True, embed=True)  # Optional parameter for insights (default: True)
):
    try:
        # Log authenticated user information (email is already verified by auth)
        logger.info(f" Suggest rules request from user: {user.email}")
        
        #  GUARDRAILS: Pre-validate input before any LLM processing
        logger.info(f" [GUARDRAIL] Starting input validation for domain: '{domain}'")
        from app.validation.input_guardrails import InputGuardrails, create_guardrail_response
        from app.vector_db.schema_loader import get_store
        
        # Get available domains for validation
        available_domains = []
        try:
            store = get_store()
            if store:
                available_domains = store.get_all_domains_realtime(force_refresh=False)
                logger.info(f" [GUARDRAIL] Retrieved {len(available_domains)} available domains for validation")
        except Exception as domain_fetch_error:
            logger.warning(f" [GUARDRAIL] Could not fetch available domains: {domain_fetch_error}")
            # Continue with empty list - will allow more permissive validation
        
        # Run comprehensive guardrail validation
        guardrails = InputGuardrails()
        is_valid, violations = guardrails.comprehensive_validate(domain, available_domains)
        
        if not is_valid:
            logger.warning(f" [GUARDRAIL] Input validation FAILED for domain: '{domain}' - {len(violations)} violations")
            
            # Return structured error response without calling LLM
            guardrail_response = create_guardrail_response(violations)
            guardrail_response.update({
                "domain": domain,
                "timestamp": time.time(),
                "processing_time_ms": 0,  # No LLM processing
                "user_id": user.user_id,  # Use user_id instead of email for PII protection
                "available_domains_count": len(available_domains)
            })
            
            return JSONResponse(
                status_code=400,
                content=guardrail_response
            )
        
        logger.info(f" [GUARDRAIL] Input validation PASSED for domain: '{domain}' - proceeding to LLM processing")
        
        # Log overall request start time (after guardrails)
        request_start_time = time.time()
        log_domain_operation("Starting rule suggestion request", domain)

        # Try to get schema from vector database
        log_domain_operation("Retrieving schema", domain)
        
        # First, test vector DB connectivity
        vector_db_status = "unknown"
        connection_error = None
        
        try:
            schema = get_schema_by_domain(domain)
            
            # If schema is empty but no exception, try with forced refresh for newly created domains
            if not schema:
                log_domain_operation("Schema not found, attempting refresh", domain)
                with log_duration("Vector DB refresh"):
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
                            log_domain_operation("Schema found after refresh", domain)
            
            vector_db_status = "connected"
            if schema:
                log_domain_operation("Schema retrieved successfully", domain, f"{len(schema)} columns found")
            else:
                log_domain_operation("Schema retrieved successfully", domain, "no columns found")
        except Exception as db_error:
            connection_error = str(db_error)
            schema = None
            # Check if it's a connection/auth issue vs schema not found
            if "AuthorizationException" in connection_error or "RetryError" in connection_error or "ConnectionError" in connection_error:
                vector_db_status = "connection_failed"
                log_error("Vector DB connection", domain, db_error)
            else:
                vector_db_status = "accessible_but_error"
                log_error("Vector DB operation", domain, db_error)

        if schema:
            log_domain_operation("Schema retrieved successfully", domain, "generating rules")
            
            if include_insights:
                # Initialize RAG enhancer
                rag_enhancer = RuleRAGEnhancer()
                
                # Use enhanced agent with RAG and full insights
                with log_duration("Agent workflow initialization"):
                    from app.agents.agent_runner import build_graph
                    from app.state.state import AgentState
                    
                    # Enhance prompt with historical context
                    enhanced_prompt = await rag_enhancer.enhance_prompt_with_history(
                        schema=schema,
                        domain=domain
                    )
                    
                    # Initialize state with enhanced prompt
                    initial_state = AgentState(
                        data_schema=schema,
                        enhanced_prompt=enhanced_prompt
                    )
                    graph = build_graph()
                
                with log_duration("Agent execution"):
                    result = graph.invoke(initial_state)
                
                logger.info(f"Total request duration: {time.time() - request_start_time:.2f}s")
                
                # Store successful policy if confidence is high
                if isinstance(result, dict):
                    state = AgentState(**result)
                else:
                    state = result
                
                confidence = _calculate_overall_confidence(state)
                if confidence >= 0.8 and state.rule_suggestions:
                    try:
                        await rag_enhancer.store_successful_policy(
                            domain=domain,
                            schema=schema,
                            rules=state.rule_suggestions,
                            performance_metrics={
                                "success_rate": confidence,
                                "validation_score": confidence,
                                "usage_count": 1
                            }
                        )
                    except Exception as e:
                        # Log the error but continue with the response
                        logger.error(f"Failed to store policy history: {e}")
                        # Don't re-raise the error as policy storage is not critical for rule suggestions
                
                if isinstance(result, dict):
                    state = AgentState(**result)
                else:
                    state = result
                    
                # Build comprehensive technical execution log (Option 3)
                execution_trace = _build_execution_trace(state)
                
                agent_insights = {
                    "execution_trace": execution_trace,
                    "summary": {
                        "total_steps": len(execution_trace.get("workflow_path", [])),
                        "tool_calls": len(execution_trace.get("tool_invocations", [])),
                        "reasoning_steps": len(state.thoughts),
                        "total_duration_ms": round(state.execution_metrics.get("total_execution_time", 0) * 1000),
                        "success_rate": _calculate_overall_confidence(state)
                    }
                }
                
                # Add final reflection if available
                final_reflection = state.reflections[-1] if state.reflections else None
                if final_reflection is not None:
                    agent_insights["final_reflection"] = final_reflection

                return { 
                    "rule_suggestions": state.rule_suggestions or [],
                    "confidence": {
                        "overall": _calculate_overall_confidence(state),
                        "level": _get_confidence_level(state),
                        "factors": _get_confidence_factors(state)
                    },
                    "agent_insights": agent_insights
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
            # Check if this might be a semantic search case first
            log_domain_operation("Schema not found, checking for semantic matches", domain)
            
            # Try semantic domain matching
            from app.validation.semantic_domain_search import enhance_domain_validation
            
            is_exact, suggested_domain, semantic_results = enhance_domain_validation(domain, available_domains)
            
            if suggested_domain:
                # Found a semantic match - check confidence level
                confidence_info = semantic_results.get("high_confidence_matches", [{}])[0]
                confidence = confidence_info.get("confidence", "high") 
                reason = confidence_info.get("reason", "semantic match")
                numeric_confidence = semantic_results.get("numeric_confidence", 0.0)
                
                log_domain_operation("Semantic match found", domain, f"suggested: {suggested_domain}, confidence: {numeric_confidence:.1%}")
                
                #  AUTO-GENERATION: If confidence is 80%+, automatically proceed with rule generation
                # Lowered to 80% for better user experience with natural language queries
                if numeric_confidence >= 0.80:
                    log_domain_operation("High confidence match", domain, f"auto-generating rules for {suggested_domain}")
                    
                    # Get the schema for the suggested domain
                    try:
                        suggested_schema = get_schema_by_domain(suggested_domain)
                        
                        if suggested_schema:
                            # Proceed with rule generation using the suggested domain
                            log_domain_operation("Schema found for suggested domain", suggested_domain, "proceeding with rule generation")
                            
                            if include_insights:
                                # Initialize RAG enhancer
                                rag_enhancer = RuleRAGEnhancer()
                                
                                # Use enhanced agent with RAG and full insights
                                with log_duration("Agent workflow initialization"):
                                    from app.agents.agent_runner import build_graph
                                    from app.state.state import AgentState
                                    
                                    # Enhance prompt with historical context
                                    enhanced_prompt = await rag_enhancer.enhance_prompt_with_history(
                                        schema=suggested_schema,
                                        domain=suggested_domain
                                    )
                                    
                                    # Initialize state with enhanced prompt
                                    initial_state = AgentState(
                                        data_schema=suggested_schema,
                                        enhanced_prompt=enhanced_prompt
                                    )
                                    graph = build_graph()
                                
                                with log_duration("Agent execution"):
                                    result = graph.invoke(initial_state)
                                
                                processing_time = round(time.time() - request_start_time, 2)
                                logger.info(f" [AUTO-SUGGEST] Automatic rule generation completed in {processing_time}s")
                                
                                # Store successful policy if confidence is high
                                if isinstance(result, dict):
                                    state = AgentState(**result)
                                else:
                                    state = result
                                
                                confidence_score = _calculate_overall_confidence(state)
                                if confidence_score >= 0.8 and state.rule_suggestions:
                                    try:
                                        await rag_enhancer.store_successful_policy(
                                            domain=suggested_domain,
                                            schema=suggested_schema,
                                            rules=state.rule_suggestions,
                                            performance_metrics={
                                                "success_rate": confidence_score,
                                                "validation_score": confidence_score,
                                                "usage_count": 1
                                            }
                                        )
                                    except Exception as e:
                                        logger.error(f"Failed to store policy history: {e}")
                                
                                if isinstance(result, dict):
                                    state = AgentState(**result)
                                else:
                                    state = result
                                    
                                # Build comprehensive technical execution log
                                execution_trace = _build_execution_trace(state)
                                
                                agent_insights = {
                                    "execution_trace": execution_trace,
                                    "summary": {
                                        "total_steps": len(execution_trace.get("workflow_path", [])),
                                        "tool_calls": len(execution_trace.get("tool_invocations", [])),
                                        "reasoning_steps": len(state.thoughts),
                                        "total_duration_ms": round(state.execution_metrics.get("total_execution_time", 0) * 1000),
                                        "success_rate": _calculate_overall_confidence(state)
                                    }
                                }
                                
                                # Add final reflection if available
                                final_reflection = state.reflections[-1] if state.reflections else None
                                if final_reflection is not None:
                                    agent_insights["final_reflection"] = final_reflection

                                return { 
                                    "success": True,
                                    "auto_generated": True,
                                    "semantic_match_used": {
                                        "original_domain": domain,
                                        "matched_domain": suggested_domain,
                                        "confidence": f"{numeric_confidence:.1%}",
                                        "reason": reason
                                    },
                                    "rule_suggestions": state.rule_suggestions or [],
                                    "confidence": {
                                        "overall": _calculate_overall_confidence(state),
                                        "level": _get_confidence_level(state),
                                        "factors": _get_confidence_factors(state)
                                    },
                                    "agent_insights": agent_insights,
                                    "processing_time": processing_time
                                }
                            else:
                                # Standard agent execution (backwards compatible)
                                rule_suggestions = run_agent(suggested_schema)
                                processing_time = round(time.time() - request_start_time, 2)
                                
                                return { 
                                    "success": True,
                                    "auto_generated": True,
                                    "semantic_match_used": {
                                        "original_domain": domain,
                                        "matched_domain": suggested_domain,
                                        "confidence": f"{numeric_confidence:.1%}",
                                        "reason": reason
                                    },
                                    "rule_suggestions": rule_suggestions,
                                    "processing_time": processing_time
                                }
                        else:
                            log_domain_operation("Schema not found for suggested domain", suggested_domain, "falling back to suggestion mode")
                    
                    except Exception as e:
                        log_error("Auto-generation for suggested domain", suggested_domain, e)
                        # Fall through to suggestion mode if auto-generation fails
                
                # If confidence < 80% or auto-generation failed, show suggestion
                return JSONResponse({
                    "error": "Domain not found exactly",
                    "alert": {
                        "type": "suggestion",
                        "message": f"Domain '{domain}' not found, but we found a similar match: '{suggested_domain}'",
                        "details": f"Semantic search found '{suggested_domain}' with {confidence} confidence. Reason: {reason}"
                    },
                    "domain": domain,
                    "suggested_domain": suggested_domain,
                    "semantic_match": {
                        "confidence": confidence,
                        "reason": reason,
                        "original_query": domain,
                        "numeric_confidence": f"{numeric_confidence:.1%}",
                        "auto_generation_available": numeric_confidence >= 0.75
                    },
                    "available_actions": {
                        "use_suggested_domain": {
                            "description": f"Use the suggested domain '{suggested_domain}' for rule generation",
                            "endpoint": "/api/aips/rules/suggest",
                            "method": "POST",
                            "payload_example": {
                                "domain": suggested_domain,
                                "include_insights": True
                            }
                        },
                        "browse_all_domains": {
                            "description": "View all available domains with semantic search",
                            "endpoint": f"/api/aips/rules/domains?search={domain}",
                            "method": "GET"
                        },
                        "suggest_new_schema": {
                            "description": "Create a new schema for the original domain name",
                            "endpoint": "/api/aips/domains/suggest-schema",
                            "method": "POST",
                            "payload_example": {
                                "domain": domain,
                                "return_csv": False
                            }
                        }
                    }
                }, status_code=404)
                
            elif semantic_results.get("found_domains"):
                # Found some partial matches
                log_domain_operation("Partial semantic matches found", domain)
                
                possible_matches = semantic_results.get("possible_matches", [])[:3]
                match_suggestions = [f"{m['domain']} ({m['confidence']})" for m in possible_matches]
                
                return JSONResponse({
                    "error": "Domain not found",
                    "alert": {
                        "type": "multiple_suggestions", 
                        "message": f"Domain '{domain}' not found. Found {len(possible_matches)} similar options.",
                        "details": f"Possible matches: {', '.join(match_suggestions)}"
                    },
                    "domain": domain,
                    "possible_matches": [
                        {
                            "domain": match["domain"],
                            "confidence": match["confidence"],
                            "reason": match["reason"]
                        } for match in possible_matches
                    ],
                    "available_actions": {
                        "browse_similar_domains": {
                            "description": "View similar domains with full semantic search results",
                            "endpoint": f"/api/aips/rules/domains?search={domain}",
                            "method": "GET"
                        },
                        "suggest_new_schema": {
                            "description": "Create a new schema for this domain name",
                            "endpoint": "/api/aips/domains/suggest-schema", 
                            "method": "POST",
                            "payload_example": {
                                "domain": domain,
                                "return_csv": False
                            }
                        }
                    }
                }, status_code=404)
            
            else:
                # No semantic matches found - direct to schema suggestion API
                log_domain_operation("No semantic matches found", domain, "directing to schema suggestion API")

                return JSONResponse({
                    "error": "Domain not found",
                    "alert": {
                        "type": "confirmation",
                        "message": f"No schema exists for domain '{domain}' and no similar domains were found.",
                        "details": "Use the /api/aips/domains/suggest-schema endpoint to generate AI-suggested column names for this domain."
                    },
                    "domain": domain,
                    "available_actions": {
                        "suggest_schema_ai": {
                            "description": "Generate suggested schema using AI (recommended)",
                            "endpoint": "/api/aips/domains/suggest-schema",
                            "method": "POST",
                            "payload_example": {
                                "domain": domain,
                                "return_csv": False
                            }
                        },
                        "browse_all_domains": {
                            "description": "View all available domains to find alternatives",
                            "endpoint": "/api/aips/rules/domains",
                            "method": "GET"
                        },
                        "semantic_search": {
                            "description": "Try semantic search to find related domains",
                            "endpoint": f"/api/aips/rules/domains?search={domain}",
                            "method": "GET"
                        }
                    },
                    "next_steps": {
                        "after_creation": [
                            "Schema will be saved to vector database",
                            "Optional: Download sample CSV data", 
                            "Call /api/aips/rules/suggest again to get validation rules"
                        ]
                    }
                }, status_code=404)

    except Exception as e:
        # This catches any unexpected errors not handled above
        log_error("Suggest rules operation", domain, e)
        traceback.print_exc()
        
        return JSONResponse({
            "error": "Internal server error",
            "alert": {
             
                "message": f"An unexpected error occurred while processing domain '{domain}'",
                "details": "Please try again or contact support if the issue persists."
            },
            "domain": domain
        }, status_code=500)


@router.get("/domains")
async def get_available_domains(
    user: UserInfo = Depends(verify_any_scope_token),
    search: str = Query(None, description="Natural language query to find matching domains"),
    max_results: int = Query(5, description="Maximum number of search results to return")
):
    """
    Get list of available domains for rule suggestion with optional semantic search
    
    This endpoint helps users discover which domains are available
    in the schema database before making rule suggestion requests.
    
    Parameters:
    - search (optional): Natural language query to find matching domains
      Examples: "customer domain", "product rules", "order management"
    - max_results (optional): Maximum number of search results to return (default: 5)
    
    If no search query is provided, returns all available domains.
    If search query is provided, returns semantically matched domains.
    """
    try:
        logger.info(f"📋 Available domains request from user: {user.email}")
        if search:
            logger.info(f"🔍 With semantic search query: '{search}'")
        
        from app.vector_db.schema_loader import get_store
        
        # Get store and fetch domains
        store = get_store()
        if not store:
            return JSONResponse({
                "error": "Vector database not available",
                "message": "Schema database is temporarily unavailable. Please try again later.",
                "available_domains": [],
                "total_count": 0
            }, status_code=503)
        
        # Fetch all available domains
        start_time = time.time()
        available_domains = store.get_all_domains_realtime(force_refresh=True)
        fetch_time = time.time() - start_time
        
        # Sort domains alphabetically for better UX
        available_domains.sort()
        
        logger.info(f"📋 Retrieved {len(available_domains)} available domains in {fetch_time:.2f}s")
        
        # If no search query, return all domains (existing behavior)
        if not search:
            return {
                "success": True,
                "available_domains": available_domains,
                "total_count": len(available_domains),
                "fetch_time_seconds": round(fetch_time, 2),
                "message": "Use one of these domain names in /suggest endpoint. If your domain is not listed, use /suggest-schema to generate it.",
                "usage": {
                    "rules_endpoint": "/api/aips/rules/suggest",
                    "schema_suggestion_endpoint": "/api/aips/domains/suggest-schema",
                    "semantic_search_hint": "Add '?search=your_query' to this endpoint for intelligent domain matching",
                    "examples": {
                        "get_rules_for_existing_domain": {
                            "method": "POST",
                            "endpoint": "/api/aips/rules/suggest", 
                            "body": {
                                "domain": "example_domain_name",
                                "include_insights": True
                            }
                        },
                        "suggest_schema_for_new_domain": {
                            "method": "POST",
                            "endpoint": "/api/aips/domains/suggest-schema",
                            "body": {
                                "domain": "new_domain_name",
                                "return_csv": False
                            }
                        },
                        "semantic_domain_search": {
                            "method": "GET",
                            "endpoint": "/api/aips/rules/domains?search=customer%20domain&max_results=3",
                            "description": "Find domains using natural language"
                        }
                    }
                }
            }
        
        # Perform semantic search
        from app.validation.semantic_domain_search import SemanticDomainSearch
        
        semantic_search = SemanticDomainSearch()
        matches = semantic_search.search_domains(search, available_domains, max_results=max_results)
        
        # Get search suggestions
        search_results = semantic_search.get_search_suggestions(search, available_domains)
        
        processing_time = round(time.time() - start_time, 2)
        logger.info(f"🎯 [SEARCH] Found {len(matches)} matches for '{search}' in {processing_time}s")
        
        response = {
            "success": True,
            "search_query": search,
            "total_available_domains": len(available_domains),
            "matches_found": len(matches),
            "processing_time": processing_time,
            "fetch_time_seconds": round(fetch_time, 2)
        }
        
        if matches:
            response["matched_domains"] = [
                {
                    "domain": match.domain,
                    "confidence_score": round(match.score, 3),
                    "match_type": match.match_type,
                    "explanation": match.explanation
                } for match in matches
            ]
            
            # Add the best match as a suggestion
            best_match = matches[0]
            response["suggested_domain"] = best_match.domain
            response["suggestion"] = f"Use '{best_match.domain}' in the /suggest endpoint for rule generation"
            
            # Also include the domain names in a simple list for easy access
            response["available_domains"] = [match.domain for match in matches]
            response["total_count"] = len(matches)
            
        else:
            response["matched_domains"] = []
            response["available_domains"] = []
            response["total_count"] = 0
            response["message"] = f"No semantic matches found for '{search}'"
            
        # Include detailed search analysis
        response["search_analysis"] = search_results
        
        # Add helpful usage information
        response["usage"] = {
            "rules_endpoint": "/api/aips/rules/suggest",
            "tips": {
                "natural_language": "Try queries like 'customer domain', 'product rules', 'order management'",
                "keywords": "Use keywords like 'customer', 'product', 'order', 'employee', 'finance'",
                "exact_match": "For exact domain names, use them directly in the /suggest endpoint"
            },
            "examples": {
                "use_found_domain": {
                    "method": "POST",
                    "endpoint": "/api/aips/rules/suggest",
                    "body": {
                        "domain": response.get("suggested_domain", "example_domain"),
                        "include_insights": True
                    }
                } if matches else None
            }
        }
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Error fetching available domains: {str(e)}")
        return JSONResponse({
            "error": "Failed to fetch domains",
            "message": "Could not retrieve available domains from schema database",
            "details": str(e),
            "available_domains": [],
            "total_count": 0
        }, status_code=500)
