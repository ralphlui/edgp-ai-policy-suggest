"""
Configuration settings for the EDGP AI Policy Suggestion API.
Uses Pydantic Settings for environment variable management and AWS Secrets Manager for secure API key storage.
"""

from pydantic_settings import BaseSettings
from pydantic import ConfigDict, field_validator, Field
from typing import List, Union
import json
import os
from pathlib import Path
from dotenv import load_dotenv
import boto3
from botocore.exceptions import ClientError
import logging

logger = logging.getLogger(__name__)

def get_secret_from_aws(secret_name: str, region_name: str) -> str:
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
        
        logger.info(f"🔍 Attempting to retrieve secret: {secret_name}")
        
        response = client.get_secret_value(SecretId=secret_name)
        
        # Parse the secret value
        if 'SecretString' in response:
            secret_data = json.loads(response['SecretString'])
            # If it's a JSON object, look for the AI agent API key
            if isinstance(secret_data, dict):
                # Look for the specific AI agent API key first
                if 'ai_agent_api_key' in secret_data:
                    logger.info(f"✅ Successfully retrieved 'ai_agent_api_key' from secret: {secret_name}")
                    return secret_data['ai_agent_api_key']
                # Try other common key names as fallback
                for key in ['OPENAI_API_KEY', 'openai_api_key', 'api_key', 'key']:
                    if key in secret_data:
                        logger.info(f"✅ Successfully retrieved '{key}' from secret: {secret_name}")
                        return secret_data[key]
                # If no standard key found, log available keys and return None
                available_keys = list(secret_data.keys())
                logger.error(f"❌ AI agent API key not found in secret. Available keys: {available_keys}")
                logger.error(f"   Expected key: 'ai_agent_api_key'")
                return None
            else:
                # If it's a plain string
                logger.info(f"✅ Successfully retrieved secret: {secret_name}")
                return secret_data
        elif 'SecretBinary' in response:
            # Handle binary secrets if needed
            logger.info(f"✅ Successfully retrieved binary secret: {secret_name}")
            return response['SecretBinary'].decode('utf-8')
        
        logger.warning(f"⚠️ Secret {secret_name} retrieved but no valid content found")
        return None
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'DecryptionFailureException':
            logger.error(f"❌ Failed to decrypt secret {secret_name}: {e}")
        elif error_code == 'InternalServiceErrorException':
            logger.error(f"❌ AWS Secrets Manager internal error for {secret_name}: {e}")
        elif error_code == 'InvalidParameterException':
            logger.error(f"❌ Invalid parameter for secret {secret_name}: {e}")
        elif error_code == 'InvalidRequestException':
            logger.error(f"❌ Invalid request for secret {secret_name}: {e}")
        elif error_code == 'ResourceNotFoundException':
            logger.error(f"❌ Secret {secret_name} not found in AWS Secrets Manager")
        else:
            logger.error(f"❌ Unexpected error retrieving secret {secret_name}: {e}")
        return None
    except Exception as e:
        logger.error(f"❌ Unexpected error retrieving secret {secret_name}: {e}")
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

# Get OPENAI_API_KEY from AWS Secrets Manager or fallback to environment variable
OPENAI_API_KEY = None
if USE_AWS_SECRETS:
    logger.info("🔐 Retrieving AI agent API key from AWS Secrets Manager...")
    OPENAI_API_KEY = get_secret_from_aws(OPENAI_SECRET_NAME, AWS_REGION)
    
    if not OPENAI_API_KEY:
        logger.warning("⚠️ Failed to retrieve AI agent API key from AWS Secrets Manager")
        logger.warning("   Trying fallback environment variable: OPENAI_API_KEY")
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        
        if OPENAI_API_KEY:
            logger.info("✅ Using OPENAI_API_KEY from environment variable as fallback")
        else:
            logger.error("❌ AI agent API key not available from either:")
            logger.error(f"   1. AWS Secrets Manager: {OPENAI_SECRET_NAME}")
            logger.error("   2. Environment variable: OPENAI_API_KEY")
            logger.error("   Please ensure:")
            logger.error(f"   - Secret '{OPENAI_SECRET_NAME}' exists in AWS Secrets Manager with key 'ai_agent_api_key'")
            logger.error(f"   - AWS credentials are configured correctly")
            logger.error(f"   - OR set OPENAI_API_KEY environment variable for testing")
            raise Exception(f"AI agent API key not available")
else:
    logger.warning("⚠️ AWS Secrets Manager is disabled (USE_AWS_SECRETS=false)")
    logger.info("🔐 Using OPENAI_API_KEY from environment variable...")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    if not OPENAI_API_KEY:
        logger.error("❌ OPENAI_API_KEY environment variable is not set")
        raise Exception("AI agent API key must be provided via OPENAI_API_KEY environment variable when AWS Secrets Manager is disabled")

RULE_MICROSERVICE_URL = os.getenv("rule.api.url")

if RULE_MICROSERVICE_URL and RULE_MICROSERVICE_URL.startswith("{") and RULE_MICROSERVICE_URL.endswith("}"):
    logger.warning(f"⚠️ RULE_MICROSERVICE_URL contains placeholder value: {RULE_MICROSERVICE_URL}")
    RULE_MICROSERVICE_URL = "http://localhost:8090/api/rules"  # Default fallback

# Log configuration status
if OPENAI_API_KEY:
    logger.info("✅ Configuration Status:")
    logger.info(f"   Environment: {env}")
    logger.info(f"   Secret Name: {OPENAI_SECRET_NAME}")
    logger.info(f"   Key starts with: {OPENAI_API_KEY[:8]}...")
else:
    logger.error("❌ OPENAI_API_KEY is not available - service cannot start")

class Settings(BaseSettings):
    """
    Application settings with environment variable support and field aliases.
    Supports dot-notation environment variables for Kubernetes compatibility.
    """
    
    model_config = ConfigDict(
        env_file=env_file_path,
        env_file_encoding='utf-8',
        case_sensitive=False,
        extra='ignore'
    )
    
    # Server Configuration
    host: str = Field(default="localhost", alias="HOST")
    port: int = Field(default=8091, alias="PORT")
    environment: str = Field(default="development", alias="ENVIRONMENT")
    
    # API Configuration
    api_title: str = Field(default="EDGP Policy Suggestion API", alias="API_TITLE")
    api_version: str = Field(default="1.0.0", alias="API_VERSION")
    api_description: str = Field(default="Data Quality Validation API using Great Expectations rules", alias="API_DESCRIPTION")
    
    # CORS Configuration
    allowed_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"],
        alias="ALLOWED_ORIGINS"
    )
    
    # AWS Configuration
    aws_region: str = Field(default="ap-southeast-1", alias="AWS_REGION")
    aoss_host: str = Field(alias="AOSS_HOST")
    aws_access_key_id: str = Field(alias="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: str = Field(alias="AWS_SECRET_ACCESS_KEY")
    
    # OpenSearch Configuration
    opensearch_index: str = Field(default="mdm-columns", alias="OPENSEARCH_INDEX")
    embed_model: str = Field(default="text-embedding-3-small", alias="EMBED_MODEL")
    embed_dim: int = Field(default=1536, alias="EMBED_DIM")
    
    # JWT Authentication Settings
    jwt_public_key: str = Field(alias="jwt.public.key")
    jwt_algorithm: str = "RS256"
    
    # Authentication microservice URL
    admin_api_url: str = Field(alias="admin.api.url")
    
    # Rule microservice URL
    rule_api_url: str = Field(alias="rule.api.url")
    
    # Logging
    log_level: str = Field(default="info", alias="LOG_LEVEL")
    
    @field_validator('allowed_origins', mode='before')
    @classmethod
    def parse_allowed_origins(cls, v) -> List[str]:
        """Parse ALLOWED_ORIGINS from string representation of list"""
        if isinstance(v, str):
            try:
                # Handle JSON-like string format
                import json
                return json.loads(v)
            except json.JSONDecodeError:
                # Handle comma-separated format
                return [origin.strip() for origin in v.split(',')]
        return v
    
    @field_validator('jwt_public_key', mode='before')
    @classmethod
    def parse_jwt_public_key(cls, v) -> str:
        """Parse JWT public key and handle escaped newlines"""
        if isinstance(v, str):
            # Replace escaped newlines with actual newlines
            return v.replace('\\n', '\n')
        return v

# Create settings instance
settings = Settings()

# Log settings validation
logger.info("🔧 Settings validation:")
logger.info(f"   JWT public key configured: {'Yes' if settings.jwt_public_key else 'No'}")
logger.info(f"   Admin API URL: {settings.admin_api_url}")
logger.info(f"   Rule API URL: {settings.rule_api_url}")
logger.info(f"   Environment: {settings.environment}")
logger.info(f"   Log level: {settings.log_level}")