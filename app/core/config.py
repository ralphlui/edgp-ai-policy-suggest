from pydantic_settings import BaseSettings
from pydantic import ConfigDict, field_validator
from typing import List, Union
import json
import os
from pathlib import Path
from dotenv import load_dotenv
import os

# Dynamic env loader: .env.development, .env.production, etc.
env = os.getenv("ENVIRONMENT", "development")
load_dotenv(dotenv_path=f".env.{env}")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
RULE_MICROSERVICE_URL = os.getenv("RULE_MICROSERVICE_URL")

# Ensure critical environment variables are set
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is missing from environment")
if not RULE_MICROSERVICE_URL:
    raise RuntimeError("RULE_MICROSERVICE_URL is missing from environment")


def get_env_file_path() -> str:
    """
    Determine which .env file to use based on APP_ENV environment variable.
    
    Environment mapping:
    - SIT -> .env.development
    - PRD -> .env.production
    - DEV/development -> .env.development
    - PROD/production -> .env.production
    - Default -> .env (if exists, otherwise .env.development)
    
    Returns:
        Path to the appropriate .env file
    """
    app_env = os.getenv("APP_ENV", "").upper()
    
    # Get the project root directory (where .env files are located)
    project_root = Path(__file__).parent.parent.parent
    
    # Environment mapping
    env_mapping = {
        "SIT": ".env.development",
        "DEV": ".env.development", 
        "DEVELOPMENT": ".env.development",
        "PRD": ".env.production",
        "PROD": ".env.production",
        "PRODUCTION": ".env.production"
    }
    
    if app_env in env_mapping:
        env_file = project_root / env_mapping[app_env]
        if env_file.exists():
            print(f"🌍 Using environment file: {env_file.name} (APP_ENV={app_env})")
            return str(env_file)
        else:
            print(f"⚠️ Environment file {env_file.name} not found for APP_ENV={app_env}")
    
    # Fallback logic - only use .env if no specific environment was requested
    if not app_env:  # Only fallback to .env if APP_ENV is not set
        fallback_files = [".env", ".env.development"]
    else:
        fallback_files = [".env.development"]  # Skip .env if specific env was requested but not found
        
    for fallback in fallback_files:
        fallback_path = project_root / fallback
        if fallback_path.exists():
            print(f"🔄 Falling back to: {fallback_path.name}")
            return str(fallback_path)
    
    # If no env files exist, return default path
    default_path = project_root / ".env"
    print(f"📝 No environment files found, using default: {default_path.name}")
    return str(default_path)

class Settings(BaseSettings):
    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8008

    # Environment
    environment: str = "development"

    # API Configuration
    api_title: str = "EDGP Rules Engine API"
    api_version: str = "1.0.0"
    api_description: str = "Data Quality Validation API using Great Expectations rules"

       # Required for AOSS
    aoss_host: str
    aws_region: str = "ap-southeast-1"

    # Optional: if you want to validate or log them
    column_index_name: str = "columns"

    # CORS Configuration
    allowed_origins: Union[List[str], str] = [
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:8080",
        "http://localhost:4200",
        "http://localhost:5173",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8080",
        "http://127.0.0.1:4200",
        "http://127.0.0.1:5173",
        "*"
    ]

    @field_validator('allowed_origins')
    @classmethod
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return [origin.strip() for origin in v.split(',')]
        return v

    model_config = ConfigDict(
        env_file=get_env_file_path(),
        env_file_encoding='utf-8',
        case_sensitive=False,
        extra="ignore",
        env_prefix=""
    )

settings = Settings()
