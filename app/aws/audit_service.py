#!/usr/bin/env python3
"""
Audit Logging Service with SQS Integration
Python implementation of the Java SQS publishing service for audit logs
"""

import json
import logging
import os
import asyncio
from typing import Optional
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from app.aws.audit_models import AuditLogDTO
from app.core.config import settings
import time

logger = logging.getLogger(__name__)

class AuditSQSService:
    """
    SQS-based audit logging service
    Equivalent to the Java SQSPublishingService
    """
    
    def __init__(self):
        self.audit_queue_url = self._get_audit_queue_url()
        self.sqs_client = None
        self.max_message_size = 256 * 1024  # 256 KB SQS limit
        self._initialize_sqs_client()
    
    def _get_audit_queue_url(self) -> Optional[str]:
        """Get audit queue URL from configuration"""
        audit_url = os.getenv("AUDIT_SQS_URL") or getattr(settings, 'audit_sqs_url', None)
        
        if not audit_url or audit_url in ["{AUDIT_SQS_URL}", "AUDIT_SQS_URL"]:
            logger.warning("Audit SQS URL not configured. Audit logging will be disabled.")
            return None
        
        return audit_url
    
    def _initialize_sqs_client(self):
        """Initialize SQS client with AWS credentials"""
        if not self.audit_queue_url:
            logger.info("SQS client not initialized - audit queue URL not configured")
            return
            
        try:
            # Try to use existing AWS session or create new one
            session = boto3.Session()
            
            # Use region from environment or default
            region = os.getenv("AWS_REGION", "us-east-1")
            
            self.sqs_client = session.client(
                'sqs',
                region_name=region,
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
            )
            
            logger.info(f"SQS client initialized for audit logging to queue: {self.audit_queue_url}")
            
        except NoCredentialsError:
            logger.error("AWS credentials not found. Audit logging will be disabled.")
            self.sqs_client = None
        except Exception as e:
            logger.error(f"Failed to initialize SQS client: {e}")
            self.sqs_client = None
    
    def send_message(self, audit_dto: AuditLogDTO) -> bool:
        """
        Send audit message to SQS queue
        Returns True if successful, False otherwise
        """
        if not self.sqs_client or not self.audit_queue_url:
            logger.debug("SQS audit logging disabled - client or queue URL not available")
            return False
        
        try:
            # Convert to SQS message format
            message_dict = audit_dto.to_sqs_message()
            message_body = json.dumps(message_dict, ensure_ascii=False)
            message_bytes = message_body.encode('utf-8')
            message_size = len(message_bytes)
            
            logger.debug("Serialized Audit Log JSON")
            
            # Check message size and truncate if necessary
            if message_size > self.max_message_size:
                logger.warning(f"Message size exceeds 256 KB limit: {message_size} bytes, truncating remarks.")
                
                truncated_remarks = self._truncate_message(
                    audit_dto.remarks, 
                    self.max_message_size, 
                    message_body
                )
                
                # Update the message with truncated remarks
                message_dict['remarks'] = truncated_remarks + "..."
                message_body = json.dumps(message_dict, ensure_ascii=False)
                message_bytes = message_body.encode('utf-8')
                
                logger.info(f"Truncated message size: {len(message_bytes)} bytes")
            
            # Send message to SQS
            response = self.sqs_client.send_message(
                QueueUrl=self.audit_queue_url,
                MessageBody=message_body,
                DelaySeconds=5  # 5 second delay as in Java version
            )
            
            message_id = response.get('MessageId')
            logger.info(f"Audit message sent to SQS with message ID: {message_id}")
            return True
            
        except ClientError as e:
            logger.error(f"AWS SQS error sending audit message: {e}")
            return False
        except Exception as e:
            logger.error(f"Error sending audit message to SQS: {e}")
            return False
    
    async def send_message_async(self, audit_dto: AuditLogDTO) -> bool:
        """
        Async wrapper for send_message to avoid blocking
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.send_message, audit_dto)
    
    def _truncate_message(self, remarks: str, max_message_size: int, current_message: str) -> str:
        """
        Truncate remarks to fit within SQS message size limit
        Equivalent to the Java truncateMessage method
        """
        try:
            current_message_bytes = current_message.encode('utf-8')
            current_size = len(current_message_bytes)
            
            remarks_bytes = remarks.encode('utf-8')
            remarks_size = len(remarks_bytes)
            
            diff_msg_size = current_size - max_message_size
            
            if diff_msg_size >= remarks_size:
                return ""
            
            allowed_bytes_for_remarks = remarks_size - (diff_msg_size + 5)  # +5 for "..." suffix
            
            if len(remarks_bytes) <= allowed_bytes_for_remarks:
                return remarks
            
            # Truncate to allowed bytes, ensuring valid UTF-8
            truncated_bytes = remarks_bytes[:allowed_bytes_for_remarks]
            
            # Handle potential UTF-8 character boundary issues
            try:
                truncated_remarks = truncated_bytes.decode('utf-8')
            except UnicodeDecodeError:
                # If we cut in the middle of a UTF-8 character, trim more
                for i in range(4):  # UTF-8 characters are at most 4 bytes
                    try:
                        truncated_remarks = truncated_bytes[:-i].decode('utf-8')
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    truncated_remarks = ""
            
            return truncated_remarks
            
        except Exception as e:
            logger.error(f"Error while truncating message remarks: {e}")
            return remarks
    
    def test_connection(self) -> bool:
        """Test SQS connection and queue accessibility"""
        if not self.sqs_client or not self.audit_queue_url:
            return False
        
        try:
            # Try to get queue attributes to test connection
            self.sqs_client.get_queue_attributes(
                QueueUrl=self.audit_queue_url,
                AttributeNames=['QueueArn']
            )
            logger.info("SQS audit queue connection test successful")
            return True
        except Exception as e:
            logger.error(f"SQS audit queue connection test failed: {e}")
            return False

# Global audit service instance
_audit_service: Optional[AuditSQSService] = None

def get_audit_service() -> AuditSQSService:
    """Get global audit service instance (singleton pattern)"""
    global _audit_service
    if _audit_service is None:
        _audit_service = AuditSQSService()
    return _audit_service

def send_audit_log(audit_dto: AuditLogDTO) -> bool:
    """
    Convenience function to send audit log
    """
    service = get_audit_service()
    return service.send_message(audit_dto)

async def send_audit_log_async(audit_dto: AuditLogDTO) -> bool:
    """
    Async convenience function to send audit log
    """
    service = get_audit_service()
    return await service.send_message_async(audit_dto)

def log_audit_locally(audit_dto: AuditLogDTO):
    """
    Fallback: log audit information locally if SQS is not available
    """
    audit_data = audit_dto.to_sqs_message()
    logger.info(f"AUDIT LOG: {json.dumps(audit_data, indent=2)}")

# Health check function for the audit system
def audit_system_health() -> dict:
    """Get audit system health status"""
    service = get_audit_service()
    
    return {
        "sqs_configured": service.audit_queue_url is not None,
        "sqs_client_initialized": service.sqs_client is not None,
        "queue_url": service.audit_queue_url,
        "connection_test": service.test_connection() if service.sqs_client else False
    }