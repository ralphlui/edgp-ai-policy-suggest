from app.vector_db.utils import filter_by_pii, filter_by_dtype, rank_by_sample_diversity

def filter_and_rank_columns(state: dict) -> dict:
    """
    LangGraph node: filter and rank semantic search results.
    Expects:
        - state["results"]: List[Dict]
        - Optional: state["pii_only"], state["allowed_types"]
    Returns:
        - state["filtered_columns"]: List[Dict]
    """
    results = state.get("results", [])
    pii_only = state.get("pii_only", False)
    allowed_types = set(state.get("allowed_types", ["string", "integer", "date"]))

    filtered = filter_by_pii(results, pii_only)
    filtered = filter_by_dtype(filtered, allowed_types)
    ranked = rank_by_sample_diversity(filtered)

    return {**state, "filtered_columns": ranked}
