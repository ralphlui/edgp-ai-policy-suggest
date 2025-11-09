"""
AWS Secrets Manager Service
Handles all interactions with AWS Secrets Manager for secure credential retrieval.
ENV-first (SIT unblock), then AWS SM (IRSA path), with lazy caching.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional, Dict, Any

import boto3
from botocore.exceptions import ClientError, BotoCoreError

logger = logging.getLogger(__name__)


def _get_config_value(key: str, default: str) -> str:
    """Read from process env to avoid import cycles."""
    return os.getenv(key, default)


def _get_openai_secret_name() -> str:
    return _get_config_value("OPENAI_SECRET_NAME", "sit/edgp/secret")


def _get_aws_region() -> str:
    return _get_config_value("AWS_REGION", os.getenv("AWS_REGION", "ap-southeast-1"))


def _format_jwt_public_key(raw_key: str) -> str:
    """
    Format JWT public key for use with PyJWT.
    If the key doesn't have PEM headers, add them.
    
    Args:
        raw_key: Raw JWT public key content (may or may not have PEM headers)
        
    Returns:
        Properly formatted PEM public key for PyJWT
    """
    if not raw_key:
        return raw_key
    
    key = raw_key.strip()
    
    # If it already has PEM headers, return as-is
    if key.startswith('-----BEGIN') and key.endswith('-----'):
        logger.info("   JWT key already in PEM format")
        return key
    
    # If it's raw base64, wrap it in PEM headers
    if key and not key.startswith('-----BEGIN'):
        logger.info("   Converting raw JWT key to PEM format")
        # Add PEM headers for RSA public key
        return f"-----BEGIN PUBLIC KEY-----\n{key}\n-----END PUBLIC KEY-----"
    
    return key


class AWSSecretsManagerService:
    """Thin wrapper over boto3 for Secrets Manager with lazy client creation."""

    def __init__(self, region_name: Optional[str] = None):
        self.region_name = region_name or _get_aws_region()
        self._client = None

    @property
    def client(self):
        if self._client is None and self.region_name:
            try:
                # Debug logging for AWS client creation
                logger.info(" Creating AWS Secrets Manager client for region: %s", self.region_name)
                
                # Clear any existing credential cache
                import boto3.session
                
                # Get credentials from environment - force fresh lookup
                import os
                aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID')
                aws_secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
                aws_region = os.environ.get('AWS_REGION') or self.region_name
                
                # Debug credential availability
                logger.info(" Environment credential check:")
                logger.info("   AWS_ACCESS_KEY_ID: %s", "Set" if aws_access_key_id and len(aws_access_key_id) > 10 else f"Invalid/Missing: '{aws_access_key_id}'")
                logger.info("   AWS_SECRET_ACCESS_KEY: %s", "Set" if aws_secret_access_key and len(aws_secret_access_key) > 10 else f"Invalid/Missing: '{aws_secret_access_key}'")
                logger.info("   AWS_REGION: %s", aws_region)
                
                if (aws_access_key_id and aws_secret_access_key and 
                    len(aws_access_key_id) > 10 and len(aws_secret_access_key) > 10 and
                    not aws_access_key_id.startswith('AWS_')):
                    
                    logger.info(" Using explicit environment credentials")
                    session = boto3.session.Session(
                        aws_access_key_id=aws_access_key_id,
                        aws_secret_access_key=aws_secret_access_key,
                        region_name=aws_region
                    )
                else:
                    logger.warning(" Invalid or missing AWS credentials, trying default credential chain")
                    session = boto3.session.Session(region_name=aws_region)
                
                self._client = session.client("secretsmanager", region_name=aws_region)
                logger.info(" AWS Secrets Manager client created successfully")
            except Exception as e:
                logger.warning(" Failed to create AWS Secrets Manager client: %s", e)
        return self._client

    def get_secret(self, secret_name: str, secret_key: str | None = None) -> Optional[str]:
        """Return the secret value (optionally a field of a JSON secret)."""
        if not self.region_name:
            logger.warning("AWS_REGION not configured - cannot retrieve secrets")
            return None
        if not self.client:
            logger.warning("AWS Secrets Manager client not available")
            return None

        try:
            logger.info(" Retrieving secret: %s", secret_name)
            logger.info(" Using region: %s", self.region_name)
            if secret_key:
                logger.info(" Looking for key: %s", secret_key)

            resp = self.client.get_secret_value(SecretId=secret_name)

            if "SecretString" in resp:
                return self._parse_secret_string(resp["SecretString"], secret_key, secret_name)
            if "SecretBinary" in resp:
                logger.info("   Retrieved binary secret: %s", secret_name)
                return resp["SecretBinary"].decode("utf-8")
            logger.warning("Secret %s retrieved but empty", secret_name)
            return None

        except ClientError as e:
            self._handle_client_error(e, secret_name)
            return None
        except BotoCoreError as e:
            logger.error("Boto error retrieving secret %s: %s", secret_name, e)
            return None
        except Exception as e:
            logger.error("Unexpected error retrieving secret %s: %s", secret_name, e)
            return None

    def _parse_secret_string(self, secret_string: str, secret_key: Optional[str], secret_name: str) -> Optional[str]:
        try:
            data = json.loads(secret_string)
            if isinstance(data, dict):
                if secret_key:
                    if secret_key in data:
                        logger.info("   ✓ Retrieved '%s' from secret: %s", secret_key, secret_name)
                        return str(data[secret_key])
                    logger.error("   Key '%s' not found. Available: %s", secret_key, list(data.keys()))
                    return None
                # fallback common key names
                for k in ("ai_agent_api_key", "OPENAI_API_KEY", "openai_api_key", "api_key", "key"):
                    if k in data and data[k]:
                        logger.info("   ✓ Retrieved '%s' from secret: %s", k, secret_name)
                        return str(data[k])
                logger.error("   No API key-like field found. Available: %s", list(data.keys()))
                return None
            # plain text in SecretString
            logger.info("   ✓ Retrieved plain text secret: %s", secret_name)
            return str(data)
        except json.JSONDecodeError:
            # not JSON, use as-is
            logger.info("   ✓ Retrieved plain text secret: %s", secret_name)
            return secret_string

    def _handle_client_error(self, error: ClientError, secret_name: str):
        code = error.response.get("Error", {}).get("Code")
        mapping = {
            "DecryptionFailureException": "Failed to decrypt secret",
            "InternalServiceErrorException": "AWS SM internal error",
            "InvalidParameterException": "Invalid parameter",
            "InvalidRequestException": "Invalid request",
            "ResourceNotFoundException": "Secret not found",
            "UnrecognizedClientException": "Invalid AWS credentials / bad token",
            "AccessDeniedException": "Access denied to secret",
        }
        msg = mapping.get(code, "Unexpected AWS error")
        logger.error("    %s for %s: %s", msg, secret_name, error)


class CredentialManager:
    """
    High-level credential manager with ENV-first fallback and AWS SM second.
    Lazy values cached after first load; can be reloaded on demand.
    """

    def __init__(self, aws_service: Optional[AWSSecretsManagerService] = None):
        self.aws_service = aws_service or AWSSecretsManagerService()
        self._openai_api_key: Optional[str] = None
        self._langsmith_api_key: Optional[str] = None
        self._jwt_public_key: Optional[str] = None
        self._loaded: bool = False

    def load_credentials(self, secret_name: Optional[str] = None, force_reload: bool = False):
        if self._loaded and not force_reload:
            return

        secret_name = secret_name or _get_openai_secret_name()
        logger.info(" Loading credentials... (ENV first, then Secrets Manager)")

        # 1) ENV first (great for SIT)
        self._openai_api_key = os.getenv("OPENAI_API_KEY")
        # LangSmith key loaded similarly, but uses its own secret name if provided
        self._langsmith_api_key = os.getenv("LANGCHAIN_API_KEY")
        jwt_env = os.getenv("JWT_PUBLIC_KEY")
        self._jwt_public_key = _format_jwt_public_key(jwt_env) if jwt_env else None

        # 2) Secrets Manager if still missing
        if not self._openai_api_key:
            try:
                # Try to get the specific key first
                self._openai_api_key = self.aws_service.get_secret(secret_name, "ai_agent_api_key")
                if not self._openai_api_key:
                    # Fallback to default logic
                    self._openai_api_key = self.aws_service.get_secret(secret_name)
            except Exception as e:
                logger.warning("AWS SM failed (OpenAI key): %s", e)

        # LangSmith secret retrieval (optional)
        if (not self._langsmith_api_key) or self._langsmith_api_key == "sk-your-key":
            try:
                # Prefer dedicated secret name, else fall back to OpenAI secret
                langsmith_secret_name = os.getenv("LANGCHAIN_SECRET_NAME") or secret_name or _get_openai_secret_name()
                if langsmith_secret_name:
                    # attempt common field names (including project-specific naming)
                    for field in (
                        "ai_agent_langsmith_api_key",  # project-specific
                        "langsmith_api_key",
                        "LANGCHAIN_API_KEY",
                        "api_key",
                        "key",
                    ):
                        self._langsmith_api_key = self.aws_service.get_secret(langsmith_secret_name, field)
                        if self._langsmith_api_key:
                            break
                    # fallback plain secret value (entire secret string)
                    if not self._langsmith_api_key:
                        self._langsmith_api_key = self.aws_service.get_secret(langsmith_secret_name)
                # export to process env so downstream libs see it
                if self._langsmith_api_key:
                    os.environ["LANGCHAIN_API_KEY"] = self._langsmith_api_key
            except Exception as e:
                logger.warning("AWS SM failed (LangSmith key): %s", e)

        if not self._jwt_public_key:
            try:
                raw = self.aws_service.get_secret(secret_name, "jwt_public_key")
                self._jwt_public_key = _format_jwt_public_key(raw) if raw else None
            except Exception as e:
                logger.warning("AWS SM failed (JWT key): %s", e)

        logger.info("    OpenAI API Key loaded: %s", bool(self._openai_api_key))
        logger.info("    LangSmith API Key loaded: %s", bool(self._langsmith_api_key))
        logger.info("    JWT Public Key loaded: %s", bool(self._jwt_public_key))
        self._loaded = True

    def get_openai_api_key(self, retry: bool = True) -> Optional[str]:
        if not self._loaded:
            self.load_credentials()
        if not self._openai_api_key and retry:
            self.load_credentials(force_reload=True)
        return self._openai_api_key

    def get_langsmith_api_key(self, retry: bool = True) -> Optional[str]:
        if not self._loaded:
            self.load_credentials()
        if not self._langsmith_api_key and retry:
            self.load_credentials(force_reload=True)
        return self._langsmith_api_key

    def get_jwt_public_key(self, retry: bool = True) -> Optional[str]:
        if not self._loaded:
            self.load_credentials()
        if not self._jwt_public_key and retry:
            self.load_credentials(force_reload=True)
        return self._jwt_public_key

    def require_openai_api_key(self) -> str:
        key = self.get_openai_api_key(retry=True)
        if not key:
            raise RuntimeError(
                "OpenAI API key is required but not available. "
                "Ensure it is set via OPENAI_API_KEY or in Secrets Manager."
            )
        return key

    def require_jwt_public_key(self) -> str:
        key = self.get_jwt_public_key(retry=True)
        if not key:
            raise RuntimeError(
                "JWT public key is required but not available. "
                "Ensure it is set via JWT_PUBLIC_KEY or in Secrets Manager (key: jwt_public_key)."
            )
        return key

    def get_credentials_status(self) -> Dict[str, Any]:
        return {
            "openai_api_key_available": bool(self._openai_api_key),
            "jwt_public_key_available": bool(self._jwt_public_key),
            "credentials_loaded": self._loaded,
            "aws_region_configured": bool(self.aws_service.region_name),
        }


# Global instance (simple DI)
_credential_manager = CredentialManager()

# Backward-compatible helpers
def load_credentials(secret_name: Optional[str] = None, force_reload: bool = False):
    _credential_manager.load_credentials(secret_name, force_reload)

def get_openai_api_key(retry: bool = True) -> Optional[str]:
    return _credential_manager.get_openai_api_key(retry)

def get_langsmith_api_key(retry: bool = True) -> Optional[str]:
    return _credential_manager.get_langsmith_api_key(retry)

def get_jwt_public_key(retry: bool = True) -> Optional[str]:
    return _credential_manager.get_jwt_public_key(retry)

def require_openai_api_key() -> str:
    return _credential_manager.require_openai_api_key()

def require_jwt_public_key() -> str:
    return _credential_manager.require_jwt_public_key()
