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
