from typing import Dict, Any
from langgraph.graph import StateGraph
from langchain_core.runnables import RunnableLambda
from app.vector_db.schema_loader import get_schema_by_domain
from app.agents.schema_suggester import bootstrap_schema_for_domain
from app.agents.agent_runner import run_agent
from app.embedding.embedder import embed_column_names_batched_async
from app.aoss import column_store
from app.graphs.graph_visualizer import LangGraphVisualizer
from app.state import LangGraphState

# Step 1: Receive domain
def receive_domain(state: Dict[str, Any]) -> Dict[str, Any]:
    return {**state, "domain": state["domain"]}

# Step 2: Search vector DB for schema
def search_vector_db(state: Dict[str, Any]) -> Dict[str, Any]:
    schema = get_schema_by_domain(state["domain"])
    return {**state, "schema": schema}

# Step 3: Check schema and fallback to LLM if missing
def check_schema(state: Dict[str, Any]) -> Any:
    if state["schema"]:
        return "use_schema"
    synthetic = bootstrap_schema_for_domain(state["domain"])
    return {**state, "schema": synthetic}, "use_llm"

# Step 4: Run rule suggestion agent
def run_agent_step(state: Dict[str, Any]) -> Dict[str, Any]:
    rules = run_agent(state["schema"])
    return {**state, "rules": rules}

# Step 5: Embed and upsert columns
async def embed_and_upsert(state: Dict[str, Any]) -> Dict[str, Any]:
    column_names = list(state["schema"].keys())
    embeddings = await embed_column_names_batched_async(column_names)

    docs = []
    for i, col_name in enumerate(column_names):
        col_info = state["schema"][col_name]
        docs.append(column_store.doc_class(
            column_id=f"{state['domain']}.{col_name}",
            column_name=col_name,
            embedding=embeddings[i],
            sample_values=col_info["sample_values"],
            metadata={
                "domain": state["domain"],
                "type": col_info["dtype"],
                "pii": False,
                "table": state["domain"],
                "source": "synthetic"
            }
        ))

    column_store.upsert_columns(docs)
    return state

# Step 6: Return CSV (placeholder)
def return_csv(state: Dict[str, Any]) -> Dict[str, Any]:
    # You can plug in your CSV generation logic here
    return {**state, "csv_ready": True}

# Build LangGraph

graph = StateGraph(state_schema=LangGraphState)

graph.add_node("receive_domain", RunnableLambda(receive_domain), config={"tool": "FastAPI"})
graph.add_node("search_vector_db", RunnableLambda(search_vector_db), config={"tool": "OpenSearch"})
graph.add_node("check_schema", RunnableLambda(check_schema), config={"tool": "LLM Fallback"})
graph.add_node("use_schema", RunnableLambda(run_agent_step), config={"tool": "LangChain Agent"})
graph.add_node("use_llm", RunnableLambda(run_agent_step), config={"tool": "LangChain Agent"})
graph.add_node("embed_upsert", RunnableLambda(embed_and_upsert), config={"tool": "OpenAIEmbeddings"})
graph.add_node("return_csv", RunnableLambda(return_csv), config={"tool": "StreamingResponse"})

graph.set_entry_point("receive_domain")
graph.add_edge("receive_domain", "search_vector_db")
graph.add_edge("search_vector_db", "check_schema")
graph.add_conditional_edges("check_schema", lambda x: "use_llm" if not x["schema"] else "use_schema")
graph.add_edge("use_schema", "embed_upsert")
graph.add_edge("use_llm", "embed_upsert")
graph.add_edge("embed_upsert", "return_csv")

policy_chain = graph.compile()

# ASCII visualization
if __name__ == "__main__":
    viz = LangGraphVisualizer(policy_chain)
    print("\n LangGraph Workflow:\n")
    print(viz.draw_ascii())
    print("\n Edge Transitions:\n")
    print(viz.describe_edges())
