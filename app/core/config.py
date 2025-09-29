from pydantic_settings import BaseSettings
from pydantic import ConfigDict, field_validator
from typing import List, Union
import json
import os
from pathlib import Path
from dotenv import load_dotenv
import boto3
from botocore.exceptions import ClientError
import logging

logger = logging.getLogger(__name__)

def get_secret_from_aws(secret_name: str, region_name: str = "ap-southeast-1") -> str:
    """
    Retrieve a secret from AWS Secrets Manager
    
    Args:
        secret_name: Name of the secret in AWS Secrets Manager
        region_name: AWS region where the secret is stored
        
    Returns:
        Secret value as string, or None if retrieval fails
    """
    try:
        # Create a Secrets Manager client
        session = boto3.session.Session()
        client = session.client(
            service_name='secretsmanager',
            region_name=region_name
        )
        
        logger.info(f" Attempting to retrieve secret: {secret_name}")
        
        response = client.get_secret_value(SecretId=secret_name)
        
        # Parse the secret value
        if 'SecretString' in response:
            secret_data = json.loads(response['SecretString'])
            # If it's a JSON object, return the entire object or specific key
            if isinstance(secret_data, dict):
                # Try common key names for OpenAI API key
                for key in ['OPENAI_API_KEY', 'openai_api_key', 'api_key', 'key']:
                    if key in secret_data:
                        logger.info(f" Successfully retrieved secret: {secret_name}")
                        return secret_data[key]
                # If no standard key found, return the first value
                if secret_data:
                    first_key = list(secret_data.keys())[0]
                    logger.info(f" Using first key '{first_key}' from secret: {secret_name}")
                    return secret_data[first_key]
            else:
                # If it's a plain string
                logger.info(f" Successfully retrieved secret: {secret_name}")
                return secret_data
        elif 'SecretBinary' in response:
            # Handle binary secrets if needed
            logger.info(f" Successfully retrieved binary secret: {secret_name}")
            return response['SecretBinary'].decode('utf-8')
        
        logger.warning(f" Secret {secret_name} retrieved but no valid content found")
        return None
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'DecryptionFailureException':
            logger.error(f" Failed to decrypt secret {secret_name}: {e}")
        elif error_code == 'InternalServiceErrorException':
            logger.error(f" AWS Secrets Manager internal error for {secret_name}: {e}")
        elif error_code == 'InvalidParameterException':
            logger.error(f" Invalid parameter for secret {secret_name}: {e}")
        elif error_code == 'InvalidRequestException':
            logger.error(f" Invalid request for secret {secret_name}: {e}")
        elif error_code == 'ResourceNotFoundException':
            logger.error(f" Secret {secret_name} not found in AWS Secrets Manager")
        else:
            logger.error(f" Unexpected error retrieving secret {secret_name}: {e}")
        return None
    except Exception as e:
        logger.error(f" Unexpected error retrieving secret {secret_name}: {e}")
        return None

# Dynamic env loader: .env.development, .env.production, etc.
# Kubernetes will inject values into .env files, and our app reads from those files
env = os.getenv("ENVIRONMENT", "development")
env_file_path = f".env.{env}"

logger.info(f"📁 Loading environment from: {env_file_path}")
load_dotenv(dotenv_path=env_file_path)

# Configuration for AWS Secrets Manager - read from .env files
OPENAI_SECRET_NAME = os.getenv("OPENAI_SECRET_NAME", "test/edgp/secret2")
AWS_REGION = os.getenv("AWS_REGION", "ap-southeast-1")
USE_AWS_SECRETS = os.getenv("USE_AWS_SECRETS", "true").lower() == "true"

# Get OPENAI_API_KEY from AWS Secrets Manager only
OPENAI_API_KEY = None
if USE_AWS_SECRETS:
    logger.info("🔐 Retrieving OpenAI API key from AWS Secrets Manager...")
    OPENAI_API_KEY = get_secret_from_aws(OPENAI_SECRET_NAME, AWS_REGION)
    
    if not OPENAI_API_KEY:
        logger.error(" Failed to retrieve OpenAI API key from AWS Secrets Manager")
        logger.error("   Please ensure:")
        logger.error(f"   1. Secret '{OPENAI_SECRET_NAME}' exists in AWS Secrets Manager")
        logger.error(f"   2. AWS credentials are configured correctly")
        logger.error(f"   3. IAM permissions allow secretsmanager:GetSecretValue")
        
        # For development only - fallback to direct key in .env file
        dev_key = os.getenv("OPENAI_API_KEY_FALLBACK")
        if dev_key and env == "development":
            logger.warning("🔄 Using fallback OpenAI API key for development")
            OPENAI_API_KEY = dev_key
else:
    logger.warning(" AWS Secrets Manager is disabled (USE_AWS_SECRETS=false)")

RULE_MICROSERVICE_URL = os.getenv("RULE_MICROSERVICE_URL")

if RULE_MICROSERVICE_URL and RULE_MICROSERVICE_URL.startswith("{") and RULE_MICROSERVICE_URL.endswith("}"):
    logger.warning(f" RULE_MICROSERVICE_URL contains placeholder value: {RULE_MICROSERVICE_URL}")
    RULE_MICROSERVICE_URL = "http://localhost:8090/api/rules"  # Default fallback

# Log configuration status
if OPENAI_API_KEY:
    logger.info(" OpenAI API key loaded successfully from AWS Secrets Manager")
    logger.info(f"   Secret name: {OPENAI_SECRET_NAME}")
    logger.info(f"   AWS region: {AWS_REGION}")
    logger.info(f"   Key starts with: {OPENAI_API_KEY[:8]}...")
else:
    logger.error(" OPENAI_API_KEY is not available - embeddings will fail")
    logger.error("   This service requires AWS Secrets Manager configuration")

if not RULE_MICROSERVICE_URL:
    logger.warning("⚠️ RULE_MICROSERVICE_URL is missing from environment - using fallback")
    RULE_MICROSERVICE_URL = "http://localhost:8090/api/rules"


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
            print(f" Using environment file: {env_file.name} (APP_ENV={app_env})")
            return str(env_file)
        else:
            print(f" Environment file {env_file.name} not found for APP_ENV={app_env}")
    
    # Fallback logic - only use .env if no specific environment was requested
    if not app_env:  # Only fallback to .env if APP_ENV is not set
        fallback_files = [".env", ".env.development"]
    else:
        fallback_files = [".env.development"]  # Skip .env if specific env was requested but not found
        
    for fallback in fallback_files:
        fallback_path = project_root / fallback
        if fallback_path.exists():
            print(f" Falling back to: {fallback_path.name}")
            return str(fallback_path)
    
    # If no env files exist, return default path
    default_path = project_root / ".env"
    print(f" No environment files found, using default: {default_path.name}")
    return str(default_path)

class Settings(BaseSettings):
    # Server Configuration - read from environment variables
    host: str = "0.0.0.0"
    port: int = 8022

    # Environment
    environment: str = "development"

    # API Configuration
    api_title: str = "EDGP AI Policy Suggest Microservice"
    api_version: str = "1.0.0"
    api_description: str = "AI-powered data quality policy and rule suggestion microservice"

    # AWS and OpenSearch Configuration - will be injected by Kubernetes
    aoss_host: str = ""
    aws_region: str = "ap-southeast-1"
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""

    # OpenSearch Configuration
    column_index_name: str = "mdm-columns"
    opensearch_index: str = "mdm-columns"
    embed_model: str = "text-embedding-3-small"
    embed_dim: int = 1536

    # AWS Secrets Manager Configuration
    use_aws_secrets: bool = True
    openai_secret_name: str = "test/edgp/secret2"

    # Service URLs
    rule_microservice_url: str = "http://localhost:8090/api/rules"

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

    @field_validator('aoss_host')
    @classmethod
    def validate_aoss_host(cls, v):
        if not v:
            logger.info("📝 AOSS_HOST will be injected by Kubernetes")
        return v

    @field_validator('aws_access_key_id')
    @classmethod
    def validate_aws_access_key(cls, v):
        if not v:
            logger.info("🔑 AWS_ACCESS_KEY_ID will be injected by Kubernetes")
        return v

    @field_validator('aws_secret_access_key')
    @classmethod
    def validate_aws_secret_key(cls, v):
        if not v:
            logger.info("🔑 AWS_SECRET_ACCESS_KEY will be injected by Kubernetes")
        return v

    model_config = ConfigDict(
        env_file=get_env_file_path(),
        env_file_encoding='utf-8',
        case_sensitive=False,
        extra="ignore",
        env_prefix="",
        # Prioritize .env file values over environment variables
        env_nested_delimiter='__'
    )

settings = Settings()

# Get values from settings (which reads from environment variables)
RULE_MICROSERVICE_URL = settings.rule_microservice_url

# Kubernetes environment validation
logger.info("� Environment Configuration Status (from .env files):")
logger.info(f"   Environment: {settings.environment}")
logger.info(f"   Host: {settings.host}:{settings.port}")
logger.info(f"   AWS Region: {settings.aws_region}")
logger.info(f"   AWS_ACCESS_KEY_ID: {' Set' if settings.aws_access_key_id else ' Missing (should be populated in .env by K8s)'}")
logger.info(f"   AWS_SECRET_ACCESS_KEY: {' Set' if settings.aws_secret_access_key else ' Missing (should be populated in .env by K8s)'}")
logger.info(f"   AOSS_HOST: {' Set' if settings.aoss_host else ' Missing (should be populated in .env by K8s)'}")
logger.info(f"   OpenSearch Index: {settings.opensearch_index}")
logger.info(f"   USE_AWS_SECRETS: {settings.use_aws_secrets}")
logger.info(f"   OPENAI_SECRET_NAME: {settings.openai_secret_name}")

# Update the global variables for backward compatibility
USE_AWS_SECRETS = settings.use_aws_secrets
OPENAI_SECRET_NAME = settings.openai_secret_name
AWS_REGION = settings.aws_region
