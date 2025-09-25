from __future__ import annotations
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator, model_validator, AnyHttpUrl, SecretStr
from typing import List, Union, Literal, Optional
from pathlib import Path
import os
import json
from typing import List
from pydantic import Field

class Settings(BaseSettings):
    # ...
    domains: List[str] = Field(default=["customer","vendor", "address", "product"])


def resolve_env_file() -> str:
    """
    Decide which .env file to load based on APP_ENV (with ENVIRONMENT as fallback).

    Mapping:
      DEV / DEVELOPMENT / SIT -> .env.development
      PROD / PRODUCTION / PRD -> .env.production
      (no APP_ENV)            -> prefer .env.development, else .env
    """
    app_env = (os.getenv("APP_ENV") or os.getenv("ENVIRONMENT") or "development").strip().lower()

    # normalize
    dev_aliases = {"dev", "development", "sit"}
    prod_aliases = {"prod", "production", "prd"}

    if app_env in dev_aliases:
        name = ".env.development"
    elif app_env in prod_aliases:
        name = ".env.production"
    else:
        # custom, e.g. "staging" -> .env.staging if present, else fallbacks
        name = f".env.{app_env}"

    # search a few likely roots
    candidates = [
        Path.cwd() / name,
        Path(__file__).resolve().parent / name,
        Path(__file__).resolve().parent.parent / name,
        Path.cwd() / ".env.development",
        Path.cwd() / ".env",
    ]

    for p in candidates:
        if p.exists():
            print(f"🌍 Using environment file: {p}")
            return str(p)

    # last resort: a non-existent .env so Pydantic continues normally
    fallback = Path.cwd() / ".env"
    print(f"📝 No environment files found; using default path: {fallback}")
    return str(fallback)

class Settings(BaseSettings):
    # ---------- Core ----------
    app_env: Literal["development", "prod", "production", "dev", "sit", "prd"] = "development"

    host: str = "0.0.0.0"
    port: int = 8008

    api_title: str = "EDGP Rules Engine API"
    api_version: str = "1.0.0"
    api_description: str = "Data Quality Validation API using Great Expectations rules"

    # ---------- External deps ----------
    openai_api_key: SecretStr
    rule_microservice_url: AnyHttpUrl

    # OpenSearch
    aoss_host: str  # e.g., xxxx.ap-southeast-1.aoss.amazonaws.com (no scheme)
    aws_region: str = "ap-southeast-1"

    # ---------- CORS ----------
    allowed_origins: Union[List[str], str] = [
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:8080",
        "http://localhost:4200",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8080",
        "http://127.0.0.1:4200",
        "http://127.0.0.1:5173",
        "*"  # ok for DEV only; we block in validator for PROD
    ]

    @field_validator("allowed_origins")
    @classmethod
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            # Try JSON first (["...","..."]), else comma-separated
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return [o.strip() for o in v.split(",") if o.strip()]
        return v

    @model_validator(mode="after")
    def validate_required_and_security(self):
        # ensure secrets exist
        if not self.openai_api_key or not self.openai_api_key.get_secret_value().strip():
            raise ValueError("OPENAI_API_KEY is required")

        if not self.rule_microservice_url:
            raise ValueError("RULE_MICROSERVICE_URL is required")

        # PROD hardening
        env_norm = (os.getenv("APP_ENV") or os.getenv("ENVIRONMENT") or self.app_env).lower()
        if env_norm in {"prod", "production", "prd"}:
            if "*" in (self.allowed_origins or []):
                raise ValueError('In production, CORS cannot include "*" — set explicit origins.')

        # Basic AOSS sanity
        if self.aoss_host.startswith("https://"):
            raise ValueError('AOSS_HOST must not include scheme (use host only, e.g., "xxxx.aoss.ap-southeast-1.amazonaws.com").')

        return self

    model_config = SettingsConfigDict(
        env_file=resolve_env_file(),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        env_prefix="",            # keep var names as-is
        env_nested_delimiter="__" # opt-in for nested later, if needed
    )

# singleton
settings = Settings()

if __name__ == "__main__":
   
    safe = settings.model_dump(exclude={"openai_api_key"})
    print(safe)