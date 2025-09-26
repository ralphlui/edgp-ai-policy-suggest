import os
import logging
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth

from app.core.config import settings

logger = logging.getLogger(__name__)

def _resolve_auth(region: str) -> AWS4Auth:
    """
    Resolve AWS credentials from environment variables.
    Only supports long-lived IAM access keys (no session token).
    """
    ak = os.getenv("AWS_ACCESS_KEY_ID")
    sk = os.getenv("AWS_SECRET_ACCESS_KEY")

    if not ak or not sk:
        raise RuntimeError("Missing AWS_ACCESS_KEY_ID or AWS_SECRET_ACCESS_KEY in environment.")

    logger.debug(" Using static IAM credentials from environment.")
    return AWS4Auth(ak, sk, region, "aoss")

def create_aoss_client(timeout_sec: int = 10) -> OpenSearch:
    """
    Create an OpenSearch Serverless client using IAM access keys.
    """
    host = settings.aoss_host
    region = settings.aws_region

    if not host or not region:
        raise RuntimeError("Missing AOSS host or region in settings.")

    if host.startswith("http"):
        raise ValueError("AOSS_HOST must be a hostname, not a URL. Example: xxxx.ap-southeast-1.aoss.amazonaws.com")

    auth = _resolve_auth(region)

    logger.info(f" Connecting to OpenSearch Serverless at {host} (region: {region})")

    return OpenSearch(
        hosts=[{"host": host, "port": 443}],
        http_auth=auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        timeout=timeout_sec,
        pool_maxsize=30,
        max_retries=3,
        retry_on_timeout=True,
    )

def test_connection() -> dict:
    """
    Test OpenSearch connectivity by calling .info()
    """
    client = create_aoss_client()
    try:
        info = client.info()
        logger.info(" OpenSearch connection successful.")
        return info
    except Exception as e:
        logger.error(f" OpenSearch connection failed: {e}")
        raise RuntimeError(f"OpenSearch connection failed: {e}")
