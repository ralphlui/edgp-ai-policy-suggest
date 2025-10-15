"""
Configuration settings for the EDGP AI Policy Suggestion API.
Uses Pydantic Settings for env management. No network calls at import time.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings
from pydantic import ConfigDict, field_validator, Field
from typing import List, Optional, Dict, Any
import os
import json
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)


def load_environment_config():
    """
    Load env based on APP_ENV.
    Priority: Existing ENV vars -> .env.{APP_ENV} -> .env.development -> .env
    """
    app_env = os.getenv("APP_ENV", "development")
    env_file_path = f".env.{app_env}"

    logger.info(" Initializing environment: %s", app_env)

    if os.path.exists(env_file_path):
        logger.info(" Loading environment from: %s", env_file_path)
        load_dotenv(dotenv_path=env_file_path, override=False)  # Don't override existing env vars
    else:
        logger.warning(" Environment file %s not found", env_file_path)
        fallback = ".env.development"
        if os.path.exists(fallback):
            logger.info(" Falling back to: %s", fallback)
            load_dotenv(dotenv_path=fallback, override=False)  # Don't override existing env vars
            app_env = "development"
        else:
            logger.warning(" No environment files found, using system env only")

    base = ".env"
    if os.path.exists(base):
        logger.info(" Loading base environment from: %s", base)
        load_dotenv(dotenv_path=base, override=False)

    return app_env, env_file_path


# Load env files but DO NOT fetch secrets here
app_env, env_file_path = load_environment_config()

OPENAI_SECRET_NAME = os.getenv("OPENAI_SECRET_NAME", "sit/edgp/secret")
AWS_REGION = os.getenv("AWS_REGION", os.getenv("AWS_REGION", "ap-southeast-1"))

logger.info(" Configuration Status:")
logger.info("   Environment: %s", app_env)
logger.info("   Environment file: %s", env_file_path)
logger.info("   Secret Name: %s", OPENAI_SECRET_NAME)
# Note: OpenAI API Key and JWT Public Key will be loaded from AWS Secrets Manager at runtime
logger.info("   AWS Secrets Manager configured: %s", bool(AWS_REGION))


class Settings(BaseSettings):
    """
    Application settings with env var support and field aliases.
    No network calls/secret fetch inside this class.
    """

    model_config = ConfigDict(
        env_file=[env_file_path, ".env"] if os.path.exists(env_file_path) else [".env"],
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Server
    host: str = Field(default=os.getenv("HOST", "0.0.0.0"), alias="HOST")
    port: int = Field(default=int(os.getenv("PORT", "8092")), alias="PORT")
    environment: str = Field(default=app_env, alias="ENVIRONMENT")

    # API
    api_title: str = Field(default=os.getenv("API_TITLE", "EDGP Policy Suggestion API"), alias="API_TITLE")
    api_version: str = Field(default=os.getenv("API_VERSION", "1.0.0"), alias="API_VERSION")
    api_description: str = Field(
        default=os.getenv("API_DESCRIPTION", "Data Quality Validation API using Great Expectations rules"),
        alias="API_DESCRIPTION",
    )

    # CORS
    allowed_origins: List[str] = Field(
        default_factory=lambda: Settings._parse_allowed_origins_default(),
        alias="ALLOWED_ORIGINS",
    )

    @staticmethod
    def _parse_allowed_origins_default() -> List[str]:
        origins_env = os.getenv("ALLOWED_ORIGINS", '["http://localhost:3000", "http://localhost:8080"]')
        try:
            if origins_env.startswith("[") and origins_env.endswith("]"):
                return json.loads(origins_env)
            return [o.strip() for o in origins_env.split(",")]
        except Exception:
            return ["http://localhost:3000", "http://localhost:8080"]

    # AWS / AOSS
    aws_region: str = Field(default=AWS_REGION, alias="AWS_REGION")
    aoss_host: str = Field(default=os.getenv("AOSS_HOST", ""), alias="AOSS_HOST")

    # Make these optional to support IRSA (no static keys required)
    aws_access_key_id: Optional[str] = Field(default=None, alias="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: Optional[str] = Field(default=None, alias="AWS_SECRET_ACCESS_KEY")

    # OpenSearch
    opensearch_index: str = Field(default=os.getenv("OPENSEARCH_INDEX", "edgp-column-metadata-sit"), alias="OPENSEARCH_INDEX")
    embed_model: str = Field(default=os.getenv("EMBED_MODEL", "text-embedding-3-small"), alias="EMBED_MODEL")
    embed_dim: int = Field(default=int(os.getenv("EMBED_DIM", "1536")), alias="EMBED_DIM")

    # LLM
    schema_llm_model: str = Field(default=os.getenv("SCHEMA_LLM_MODEL", "gpt-4o-mini"), alias="SCHEMA_LLM_MODEL")
    rules_llm_model: str = Field(default=os.getenv("RULES_LLM_MODEL", "gpt-4o-mini"), alias="RULES_LLM_MODEL")
    llm_temperature: float = Field(default=float(os.getenv("LLM_TEMPERATURE", "0.3")), alias="LLM_TEMPERATURE")

    # LLM Validation and Safety Configuration
    llm_validation_enabled: bool = Field(default=os.getenv("LLM_VALIDATION_ENABLED", "true").lower() == "true", alias="LLM_VALIDATION_ENABLED")
    llm_input_max_length: int = Field(default=int(os.getenv("LLM_INPUT_MAX_LENGTH", "10000")), alias="LLM_INPUT_MAX_LENGTH")
    llm_output_max_length: int = Field(default=int(os.getenv("LLM_OUTPUT_MAX_LENGTH", "50000")), alias="LLM_OUTPUT_MAX_LENGTH")
    llm_rate_limit_per_minute: int = Field(default=int(os.getenv("LLM_RATE_LIMIT_PER_MINUTE", "60")), alias="LLM_RATE_LIMIT_PER_MINUTE")
    llm_rate_limit_per_hour: int = Field(default=int(os.getenv("LLM_RATE_LIMIT_PER_HOUR", "1000")), alias="LLM_RATE_LIMIT_PER_HOUR")
    llm_strict_mode: bool = Field(default=os.getenv("LLM_STRICT_MODE", "true").lower() == "true", alias="LLM_STRICT_MODE")
    llm_auto_correct: bool = Field(default=os.getenv("LLM_AUTO_CORRECT", "false").lower() == "true", alias="LLM_AUTO_CORRECT")
    llm_advanced_safety: bool = Field(default=os.getenv("LLM_ADVANCED_SAFETY", "true").lower() == "true", alias="LLM_ADVANCED_SAFETY")
    llm_safety_threshold: float = Field(default=float(os.getenv("LLM_SAFETY_THRESHOLD", "0.7")), alias="LLM_SAFETY_THRESHOLD")
    
    # Backward compatibility properties
    @property
    def llm_max_input_length(self) -> int:
        """Backward compatibility for llm_max_input_length"""
        return self.llm_input_max_length
    
    @property 
    def validation_rate_limit_per_minute(self) -> int:
        """Backward compatibility for validation_rate_limit_per_minute"""
        return self.llm_rate_limit_per_minute
    
    @property
    def validation_rate_limit_per_hour(self) -> int:
        """Backward compatibility for validation_rate_limit_per_hour"""
        return self.llm_rate_limit_per_hour
    
    @property
    def validation_max_input_length(self) -> int:
        """Backward compatibility for validation_max_input_length"""
        return self.llm_input_max_length
    
    def get_llm_validation_config(self) -> Dict[str, Any]:
        """Get LLM validation configuration with policy-aware defaults"""
        return {
            "enabled": self.llm_validation_enabled,
            "strict_mode": False,  # More lenient for business context
            "auto_correct": True,  # Enable auto-correction for better UX
            "rate_limit_per_minute": self.llm_rate_limit_per_minute,
            "rate_limit_per_hour": self.llm_rate_limit_per_hour,
            "max_input_length": self.llm_input_max_length,
            "enable_advanced_safety": True,
            "policy_aware": True,  # Enable policy-aware validation
            "business_context": True,  # Enable business context allowlist
            "schema_validation": True,  # Enable schema-specific validation
        }

    # JWT — comes from AWS Secrets Manager at runtime via aws_secrets_service.get_jwt_public_key()
    jwt_public_key: str = Field(default="", alias="JWT_PUBLIC_KEY")  # Placeholder, real key from AWS
    jwt_algorithm: str = "RS256"

    # Service URLs
    admin_api_url: str = Field(default=os.getenv("ADMIN_URL", ""), alias="ADMIN_URL")
    rule_api_url: str = Field(default=os.getenv("RULE_URL", ""), alias="RULE_URL")

    # Logging
    log_level: str = Field(default=os.getenv("LOG_LEVEL", "info"), alias="LOG_LEVEL")

    @field_validator("allowed_origins", mode="before")
    @classmethod
    def parse_allowed_origins(cls, v) -> List[str]:
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return [origin.strip() for origin in v.split(",")]
        return v


# One global settings instance
settings = Settings()

logger.info(" Settings validation:")
logger.info("   APP_ENV: %s", app_env)
logger.info("   Environment file loaded: %s", env_file_path if os.path.exists(env_file_path) else "None")
logger.info("   Admin API URL: %s", settings.admin_api_url)
logger.info("   Rule API URL: %s", settings.rule_api_url)
logger.info("   Environment: %s", settings.environment)
logger.info("   Log level: %s", settings.log_level)
