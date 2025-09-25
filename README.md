## Policy Suggestion Agent

This system provides intelligent, column-level validation rule suggestions for master data governance. It is designed to support policy creation workflows by integrating schema extraction, rule retrieval, and LLM-powered reasoning. 

## Overview
When a user initiates policy creation by selecting a domain, the system performs the following steps:

1. Accepts the domain name via API
2. Extracts column schema and sample values from a vector database (OpenSearch)
3. Retrieves available GX rule templates from an external rule microservice
4. Uses a language model agent to suggest appropriate validation rules per column
5. Returns structured rule suggestions with rationale
This process helps prevent poor-quality master data by enforcing domain-specific validation logic.

## Technology Stack
FastAPI: API interface
LangChain: LLM agent and tool orchestration
LangGraph: Agentic workflow orchestration
Pydantic: Typed state and schema validation
Pandas: Schema parsing and sample extraction
Amazon OpenSearch: Vector database for column metadata
GX Rule Microservice: Source of rule templates