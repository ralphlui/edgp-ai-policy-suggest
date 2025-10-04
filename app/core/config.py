"""
Configuration settings for the EDGP AI Policy Suggestion API.
Uses Pydantic Settings for env management. No network calls at import time.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings
from pydantic import ConfigDict, field_validator, Field
from typing import List, Optional
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
