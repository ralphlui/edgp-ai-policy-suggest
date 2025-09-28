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
import logging

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
    graph = build_graph()
    result = graph.invoke(AgentState(data_schema=schema), return_dict=False)

    if isinstance(result, dict):
        logger.warning("LangGraph returned dict instead of AgentState")
        return result.get("rule_suggestions", [])

    logger.info("Final rule suggestions: %s", result.rule_suggestions)
    return result.rule_suggestions or []
