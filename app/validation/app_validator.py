import re

DOMAIN_PATTERN = re.compile(r"^[a-z0-9_-]+$")

def validate_domain_name(domain: str) -> str:
    """
    Validate that the domain name only contains lowercase letters, digits,
    underscores, or hyphens. Raise ValueError if invalid.
    """
    if not DOMAIN_PATTERN.match(domain):
        raise ValueError(
            f"Invalid domain name '{domain}'. "
            "Only lowercase letters, digits, underscores, and hyphens are allowed."
        )
    return domain

def validate_column_schema(schema: dict) -> bool:
    """
    Validate schema structure for downstream agents.
    Expected format: {column_name: {"dtype": str, "sample_values": List[str]}}
    """
    if not isinstance(schema, dict):
        return False

    for col, info in schema.items():
        if not isinstance(info, dict):
            return False
        if "dtype" not in info or "sample_values" not in info:
            return False
        if not isinstance(info["sample_values"], list):
            return False

    return True
