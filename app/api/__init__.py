# Re-export for backward compatibility
from app.api.aoss_routes import (
    create_domain, get_domains, verify_domain_exists,
    list_domains_in_vectordb, get_domain_from_vectordb,
    download_csv_file, regenerate_suggestions, extend_domain, suggest_extensions, get_store,
    check_vectordb_status
)

# Import rule suggestion routes for re-export
try:
    from app.api.rule_suggestion_routes import suggest_rules
except ImportError:
    # Avoid circular import
    pass

# Import validator routes if available for re-export
try:
    from app.api.validator_routes import validation_router
except ImportError:
    # Validation module might not be available
    pass