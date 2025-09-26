from typing import Dict, Any

def filter_pii_columns(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return only columns marked as PII in metadata.
    """
    return {
        col: info
        for col, info in schema.items()
        if info.get("metadata", {}).get("pii") is True
    }

def filter_by_dtype(schema: Dict[str, Any], allowed_types: set[str]) -> Dict[str, Any]:
    """
    Return only columns with allowed data types.
    """
    return {
        col: info
        for col, info in schema.items()
        if info.get("dtype") in allowed_types
    }

def rank_columns_by_sample_diversity(schema: Dict[str, Any]) -> list[str]:
    """
    Return column names ranked by sample value uniqueness.
    """
    return sorted(
        schema.keys(),
        key=lambda col: len(set(schema[col].get("sample_values", []))),
        reverse=True
    )
