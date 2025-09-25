from typing import Optional
import os
import boto3
from opensearchpy import OpenSearch
from requests_aws4auth import AWS4Auth
import multiprocessing

from app.core.config import settings  # your singleton

def _resolve_auth(region: str) -> AWS4Auth:
    """
    Resolve AWS credentials for SigV4:
      1) Environment vars (AWS_ACCESS_KEY_ID/SECRET/TOKEN)
      2) Default boto3 chain (profile, instance role, etc.)
    """
    ak = os.getenv("AWS_ACCESS_KEY_ID")
    sk = os.getenv("AWS_SECRET_ACCESS_KEY")
    st = os.getenv("AWS_SESSION_TOKEN")

    if ak and sk:
        return AWS4Auth(ak, sk, region, "aoss", session_token=st)

    session = boto3.Session()
    creds = session.get_credentials()
    if creds is None:
        raise RuntimeError("AWS credentials not found. Configure IAM role or env vars.")
    frozen = creds.get_frozen_credentials()
    return AWS4Auth(frozen.access_key, frozen.secret_key, region, "aoss", session_token=frozen.token)

def create_aoss_client(timeout_sec: int = 10) -> OpenSearch:
    """
    Create an OpenSearch Serverless client with SigV4 auth.
    """
    if not settings.aoss_host or not settings.aws_region:
        raise RuntimeError("Missing AOSS host or region in settings.")
    
    # Handle multiprocessing context issues
    try:
        auth = _resolve_auth(settings.aws_region)
    except Exception as e:
        # If there's an issue with multiprocessing, log and re-raise
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to resolve AWS auth: {e}")
        raise

    return OpenSearch(
        hosts=[{"host": settings.aoss_host, "port": 443}],
        http_auth=auth,
        use_ssl=True,
        verify_certs=True,
        timeout=timeout_sec,
        pool_maxsize=30,
        max_retries=3,
        retry_on_timeout=True,
    )
