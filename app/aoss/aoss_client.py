from typing import Optional
import os
import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
import logging

from app.core.config import settings

logger = logging.getLogger(__name__)

def create_aoss_client(timeout_sec: int = 30) -> OpenSearch:
    """
    Create an OpenSearch Serverless client with modern AWSV4SignerAuth.
    
    This uses the newer AWSV4SignerAuth which is more reliable than AWS4Auth
    for OpenSearch Serverless connections.
    """
    if not settings.aoss_host or not settings.aws_region:
        raise RuntimeError("Missing AOSS host or region in settings.")
    
    logger.info(f"Creating AOSS client for host: {settings.aoss_host}")
    logger.info(f"AWS region: {settings.aws_region}")
    
    try:
        # Create boto3 session for credential resolution
        session = boto3.Session()
        credentials = session.get_credentials()
        
        if credentials is None:
            raise RuntimeError(
                "AWS credentials not found. Please ensure:\n"
                "1. AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are set, OR\n"
                "2. AWS CLI is configured with 'aws configure', OR\n" 
                "3. IAM role is attached to your instance"
            )
        
        # Log credential source (but not the actual keys!)
        cred_source = "environment variables" if os.getenv("AWS_ACCESS_KEY_ID") else "boto3 default chain"
        logger.info(f"Using AWS credentials from: {cred_source}")
        
        # Create modern AWSV4SignerAuth
        awsauth = AWSV4SignerAuth(credentials, settings.aws_region, 'aoss')
        
        # Create OpenSearch client
        client = OpenSearch(
            hosts=[{"host": settings.aoss_host, "port": 443}],
            http_auth=awsauth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
            timeout=timeout_sec,
            max_retries=3,
            retry_on_timeout=True,
            # Additional headers for better debugging
            headers={"Content-Type": "application/json"}
        )
        
        # Test the connection (but don't fail startup if this doesn't work)
        try:
            info = client.info()
            logger.info("✅ AOSS connection successful!")
            logger.info(f"Cluster name: {info.get('cluster_name', 'Unknown')}")
            logger.info(f"Version: {info.get('version', {}).get('number', 'Unknown')}")
        except Exception as test_error:
            logger.warning(f"⚠️ AOSS connection test failed: {test_error}")
            logger.warning("This might be due to:")
            logger.warning("1. Missing IAM permissions for your AWS user")
            logger.warning("2. Missing OpenSearch Serverless Data Access Policy")
            logger.warning("3. Incorrect collection name or region")
            logger.warning("See AWS_SETUP_GUIDE.md for detailed setup instructions")
            # Don't raise the error - let the client be created but warn about issues
            logger.warning("⚠️ Creating client anyway - operations may fail until permissions are fixed")
            
        return client
        
    except Exception as e:
        logger.error(f"Failed to create AOSS client: {e}")
        logger.error(f"Host: {settings.aoss_host}")
        logger.error(f"Region: {settings.aws_region}")
        
        # Provide helpful error messages
        if "AuthorizationException" in str(e):
            logger.error(" Authorization failed - check AWS permissions:")
            logger.error("1. IAM policy attached to your user")
            logger.error("2. OpenSearch Serverless Data Access Policy")
            logger.error("3. Correct user ARN in the data access policy")
        elif "credentials" in str(e).lower():
            logger.error(" Credential resolution failed:")
            logger.error("1. Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY")
            logger.error("2. Or run 'aws configure' to set up CLI")
            logger.error("3. Or attach IAM role to your instance")
        
        raise

def test_aoss_connection() -> bool:
    """
    Test AOSS connection and return True if successful.
    Useful for health checks and debugging.
    """
    try:
        client = create_aoss_client()
        client.info()
        return True
    except Exception as e:
        logger.error(f"AOSS connection test failed: {e}")
        return False
