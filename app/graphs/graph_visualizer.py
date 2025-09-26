from langgraph.graph import StateGraph
from typing import Union, Dict, Any, List, Optional
import textwrap

class LangGraphVisualizer:
    def __init__(self, compiled_graph):
        self.graph = compiled_graph.get_graph()
        self.nodes = list(self.graph.nodes.keys())
        self.edges = self.graph.edges
        self.compiled_graph = compiled_graph

    def draw_ascii(self) -> str:
        """Create a comprehensive ASCII diagram of the workflow"""
        lines = []
        lines.append("╔══════════════════════════════════════════════════════════════╗")
        lines.append("║                    EDGP Policy Chain Workflow               ║")
        lines.append("╚══════════════════════════════════════════════════════════════╝")
        lines.append("")
        
        # Draw the flow diagram
        lines.extend(self._draw_workflow_diagram())
        
        return "\n".join(lines)

    def _draw_workflow_diagram(self) -> List[str]:
        """Draw the detailed workflow diagram"""
        diagram = [
            "    START",
            "      │",
            "      ▼",
            "  ┌─────────────────┐",
            "  │ receive_domain  │ ◄─── FastAPI Request",
            "  │                 │      (domain: string)",
            "  └─────────────────┘",
            "          │",
            "          ▼",
            "  ┌─────────────────┐",
            "  │ search_vector_db│ ◄─── OpenSearch Query",
            "  │                 │      (semantic search)",
            "  └─────────────────┘",
            "          │",
            "          ▼",
            "  ┌─────────────────┐",
            "  │  check_schema   │ ◄─── Decision Node",
            "  │                 │      (schema exists?)",
            "  └─────────────────┘",
            "          │",
            "     ┌────┴────┐",
            "     ▼         ▼",
            "┌───────────┐ ┌───────────┐",
            "│use_schema │ │  use_llm  │ ◄─── LangChain Agents",
            "│           │ │           │      (rule generation)",
            "└───────────┘ └───────────┘",
            "     │         │",
            "     └────┬────┘",
            "          ▼",
            "  ┌─────────────────┐",
            "  │  embed_upsert   │ ◄─── OpenAI Embeddings",
            "  │                 │      + OpenSearch Upsert",
            "  └─────────────────┘",
            "          │",
            "          ▼",
            "  ┌─────────────────┐",
            "  │   return_csv    │ ◄─── CSV Generation",
            "  │                 │      (streaming response)",
            "  └─────────────────┘",
            "          │",
            "          ▼",
            "       END"
        ]
        return diagram

    def describe_edges(self) -> str:
        """Describe the edge transitions with details"""
        lines = ["Edge Transitions:"]
        lines.append("─" * 50)
        
        edge_descriptions = {
            ("receive_domain", "search_vector_db"): "Domain validated → Vector DB lookup",
            ("search_vector_db", "check_schema"): "Schema retrieved → Decision point",
            ("check_schema", "use_schema"): "Schema exists → Use existing schema",
            ("check_schema", "use_llm"): "No schema found → Generate with LLM",
            ("use_schema", "embed_upsert"): "Rules generated → Embedding & storage",
            ("use_llm", "embed_upsert"): "Rules generated → Embedding & storage",
            ("embed_upsert", "return_csv"): "Data stored → CSV preparation",
        }
        
        for edge in self.edges:
            if isinstance(edge, tuple) and len(edge) == 2:
                source, target = edge
                description = edge_descriptions.get((source, target), f"{source} → {target}")
                lines.append(f"• {description}")
            else:
                lines.append(f"• {edge}")
        
        return "\n".join(lines)

    def draw_state_flow(self) -> str:
        """Draw the state transformation flow"""
        lines = []
        lines.append("╔══════════════════════════════════════════════════════════════╗")
        lines.append("║                    State Transformation Flow                 ║")
        lines.append("╚══════════════════════════════════════════════════════════════╝")
        lines.append("")
        
        state_flow = [
            "Initial State:",
            "┌─────────────────────────────────────────────────────────────┐",
            "│ LangGraphState                                              │",
            "│ ┌─────────────────────────────────────────────────────────┐ │",
            "│ │ domain: str                     │ 'customer'            │ │",
            "│ │ schema: Optional[Dict]          │ None                  │ │",
            "│ │ rules: Optional[List[str]]      │ None                  │ │",
            "│ │ query_embedding: Optional[List] │ None                  │ │",
            "│ │ results: Optional[List[Dict]]   │ None                  │ │",
            "│ │ csv_ready: Optional[bool]       │ False                 │ │",
            "│ └─────────────────────────────────────────────────────────┘ │",
            "└─────────────────────────────────────────────────────────────┘",
            "",
            "After search_vector_db:",
            "┌─────────────────────────────────────────────────────────────┐",
            "│ schema: Dict[str, ColumnInfo]   │ Retrieved from OpenSearch │",
            "└─────────────────────────────────────────────────────────────┘",
            "",
            "After use_schema/use_llm:",
            "┌─────────────────────────────────────────────────────────────┐",
            "│ rules: List[str]                │ Generated policy rules    │",
            "└─────────────────────────────────────────────────────────────┘",
            "",
            "After embed_upsert:",
            "┌─────────────────────────────────────────────────────────────┐",
            "│ query_embedding: List[float]    │ Column embeddings         │",
            "│ results: List[Dict]             │ Upserted documents        │",
            "└─────────────────────────────────────────────────────────────┘",
            "",
            "Final State:",
            "┌─────────────────────────────────────────────────────────────┐",
            "│ csv_ready: bool                 │ True                      │",
            "└─────────────────────────────────────────────────────────────┘"
        ]
        
        lines.extend(state_flow)
        return "\n".join(lines)

    def draw_tools_and_integrations(self) -> str:
        """Draw the tools and integrations used in each step"""
        lines = []
        lines.append("╔══════════════════════════════════════════════════════════════╗")
        lines.append("║                Tools & Integrations Mapping                  ║")
        lines.append("╚══════════════════════════════════════════════════════════════╝")
        lines.append("")
        
        tools_diagram = [
            "┌─────────────────────┬─────────────────────┬─────────────────────┐",
            "│ Node                │ Tool/Integration    │ Purpose             │",
            "├─────────────────────┼─────────────────────┼─────────────────────┤",
            "│ receive_domain      │ FastAPI             │ HTTP request        │",
            "│                     │                     │ handling            │",
            "├─────────────────────┼─────────────────────┼─────────────────────┤",
            "│ search_vector_db    │ OpenSearch          │ Schema retrieval    │",
            "│                     │ AWS AOSS            │ via semantic search │",
            "├─────────────────────┼─────────────────────┼─────────────────────┤",
            "│ check_schema        │ LLM Fallback        │ Decision logic &    │",
            "│                     │ Schema Bootstrap    │ synthetic schema    │",
            "├─────────────────────┼─────────────────────┼─────────────────────┤",
            "│ use_schema          │ LangChain Agent     │ Rule generation     │",
            "│ use_llm             │ OpenAI GPT          │ with existing/new   │",
            "│                     │                     │ schema              │",
            "├─────────────────────┼─────────────────────┼─────────────────────┤",
            "│ embed_upsert        │ OpenAI Embeddings   │ Vector generation   │",
            "│                     │ OpenSearch Bulk     │ and storage         │",
            "├─────────────────────┼─────────────────────┼─────────────────────┤",
            "│ return_csv          │ StreamingResponse   │ CSV file generation │",
            "│                     │ FastAPI Response    │ and delivery        │",
            "└─────────────────────┴─────────────────────┴─────────────────────┘"
        ]
        
        lines.extend(tools_diagram)
        return "\n".join(lines)

    def get_execution_stats(self) -> str:
        """Get execution statistics and performance info"""
        lines = []
        lines.append("╔══════════════════════════════════════════════════════════════╗")
        lines.append("║                    Execution Information                     ║")
        lines.append("╚══════════════════════════════════════════════════════════════╝")
        lines.append("")
        
        stats = [
            f"Total Nodes: {len(self.nodes)}",
            f"Total Edges: {len(self.edges)}",
            "Entry Point: receive_domain",
            "Exit Point: return_csv",
            "",
            "Conditional Logic:",
            "• check_schema → Routes based on schema availability",
            "• Parallel execution possible for use_schema/use_llm",
            "",
            "External Dependencies:",
            "• OpenSearch Serverless (AWS)",
            "• OpenAI API (embeddings & LLM)",
            "• FastAPI (web framework)",
            "• LangChain (agent orchestration)",
            "",
            "State Management:",
            "• Pydantic-based state validation",
            "• Immutable state transitions",
            "• Type-safe operations"
        ]
        
        lines.extend(stats)
        return "\n".join(lines)

    def full_visualization(self) -> str:
        """Generate complete visualization with all components"""
        sections = [
            self.draw_ascii(),
            "",
            self.describe_edges(),
            "",
            self.draw_state_flow(),
            "",
            self.draw_tools_and_integrations(),
            "",
            self.get_execution_stats()
        ]
        
        return "\n".join(sections)

