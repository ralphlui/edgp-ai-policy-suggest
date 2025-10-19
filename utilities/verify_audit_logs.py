#!/usr/bin/env python3
"""
Audit Log Verification Utility
Helps verify that audit logs are being sent correctly to SQS
"""

import boto3
import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from botocore.exceptions import ClientError, NoCredentialsError

class AuditLogVerifier:
    """Utility to verify audit logs in SQS queue"""
    
    def __init__(self):
        self.audit_queue_url = os.getenv("AUDIT_SQS_URL")
        self.sqs_client = None
        self._initialize_sqs_client()
    
    def _initialize_sqs_client(self):
        """Initialize SQS client"""
        if not self.audit_queue_url or self.audit_queue_url in ["{AUDIT_SQS_URL}", "AUDIT_SQS_URL"]:
            print(" AUDIT_SQS_URL not configured")
            return
        
        try:
            region = os.getenv("AWS_REGION", "us-east-1")
            self.sqs_client = boto3.client(
                'sqs',
                region_name=region,
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
            )
            print(f" SQS client initialized for queue: {self.audit_queue_url}")
        except NoCredentialsError:
            print(" AWS credentials not found")
        except Exception as e:
            print(f" Failed to initialize SQS client: {e}")
    
    def check_queue_status(self) -> Dict:
        """Check SQS queue status and attributes"""
        if not self.sqs_client:
            return {"error": "SQS client not initialized"}
        
        try:
            # Get queue attributes
            response = self.sqs_client.get_queue_attributes(
                QueueUrl=self.audit_queue_url,
                AttributeNames=['All']
            )
            
            attributes = response['Attributes']
            
            return {
                "queue_url": self.audit_queue_url,
                "approximate_messages": int(attributes.get('ApproximateNumberOfMessages', 0)),
                "approximate_not_visible": int(attributes.get('ApproximateNumberOfMessagesNotVisible', 0)),
                "message_retention_period": int(attributes.get('MessageRetentionPeriod', 0)) // 86400,  # Convert to days
                "visibility_timeout": int(attributes.get('VisibilityTimeout', 0)),
                "max_message_size": int(attributes.get('MaxMessageSize', 0)),
                "created_timestamp": datetime.fromtimestamp(int(attributes.get('CreatedTimestamp', 0))),
                "last_modified": datetime.fromtimestamp(int(attributes.get('LastModifiedTimestamp', 0)))
            }
            
        except ClientError as e:
            return {"error": f"AWS error: {e}"}
        except Exception as e:
            return {"error": f"Error: {e}"}
    
    def receive_recent_messages(self, max_messages: int = 10) -> List[Dict]:
        """Receive recent audit messages from the queue"""
        if not self.sqs_client:
            return []
        
        try:
            response = self.sqs_client.receive_message(
                QueueUrl=self.audit_queue_url,
                MaxNumberOfMessages=min(max_messages, 10),  # SQS limit is 10
                WaitTimeSeconds=5,  # Short poll
                AttributeNames=['All'],
                MessageAttributeNames=['All']
            )
            
            messages = response.get('Messages', [])
            audit_logs = []
            
            for message in messages:
                try:
                    # Parse message body
                    body = json.loads(message['Body'])
                    
                    audit_log = {
                        "message_id": message['MessageId'],
                        "receipt_handle": message['ReceiptHandle'],
                        "sent_timestamp": datetime.fromtimestamp(
                            int(message['Attributes']['SentTimestamp']) / 1000
                        ),
                        "audit_data": body
                    }
                    
                    audit_logs.append(audit_log)
                    
                except json.JSONDecodeError:
                    print(f"  Invalid JSON in message: {message['MessageId']}")
                except Exception as e:
                    print(f"  Error parsing message {message['MessageId']}: {e}")
            
            return audit_logs
            
        except ClientError as e:
            print(f" AWS error receiving messages: {e}")
            return []
        except Exception as e:
            print(f" Error receiving messages: {e}")
            return []
    
    def delete_message(self, receipt_handle: str) -> bool:
        """Delete a message from the queue after processing"""
        if not self.sqs_client:
            return False
        
        try:
            self.sqs_client.delete_message(
                QueueUrl=self.audit_queue_url,
                ReceiptHandle=receipt_handle
            )
            return True
        except Exception as e:
            print(f" Error deleting message: {e}")
            return False
    
    def verify_audit_format(self, audit_data: Dict) -> Dict:
        """Verify audit log format matches expected structure"""
        required_fields = [
            'userId', 'userName', 'activityType', 'endPoint', 
            'requestId', 'clientIp', 'remarks', 'timestamp'
        ]
        
        verification = {
            "valid": True,
            "missing_fields": [],
            "field_analysis": {},
            "format_issues": []
        }
        
        # Check required fields
        for field in required_fields:
            if field not in audit_data:
                verification["missing_fields"].append(field)
                verification["valid"] = False
            else:
                value = audit_data[field]
                verification["field_analysis"][field] = {
                    "present": True,
                    "type": type(value).__name__,
                    "length": len(str(value)) if value else 0,
                    "sample": str(value)[:100] if value else ""
                }
        
        # Validate timestamp format
        if 'timestamp' in audit_data:
            try:
                datetime.fromisoformat(audit_data['timestamp'].replace('Z', '+00:00'))
                verification["field_analysis"]['timestamp']["valid_iso_format"] = True
            except:
                verification["format_issues"].append("Invalid timestamp format")
                verification["valid"] = False
        
        # Validate remarks JSON
        if 'remarks' in audit_data:
            try:
                remarks_data = json.loads(audit_data['remarks'])
                verification["field_analysis"]['remarks']["valid_json"] = True
                verification["field_analysis"]['remarks']["keys"] = list(remarks_data.keys())
            except:
                verification["format_issues"].append("Invalid remarks JSON format")
        
        return verification

def print_queue_status(verifier: AuditLogVerifier):
    """Print queue status information"""
    print("📊 SQS QUEUE STATUS")
    print("=" * 50)
    
    status = verifier.check_queue_status()
    
    if "error" in status:
        print(f" Error: {status['error']}")
        return
    
    print(f"Queue URL: {status['queue_url']}")
    print(f"Available Messages: {status['approximate_messages']}")
    print(f"Messages in Flight: {status['approximate_not_visible']}")
    print(f"Message Retention: {status['message_retention_period']} days")
    print(f"Visibility Timeout: {status['visibility_timeout']} seconds")
    print(f"Max Message Size: {status['max_message_size']:,} bytes")
    print(f"Created: {status['created_timestamp']}")
    print(f"Last Modified: {status['last_modified']}")

def print_audit_messages(messages: List[Dict]):
    """Print audit messages in a readable format"""
    print(f"\n RECENT AUDIT MESSAGES ({len(messages)} found)")
    print("=" * 50)
    
    if not messages:
        print("No messages found in queue")
        return
    
    for i, message in enumerate(messages, 1):
        audit_data = message['audit_data']
        sent_time = message['sent_timestamp']
        
        print(f"\nMessage {i}:")
        print(f"  Message ID: {message['message_id']}")
        print(f"  Sent Time: {sent_time}")
        print(f"  User: {audit_data.get('userName', 'N/A')} ({audit_data.get('userId', 'N/A')})")
        print(f"  Activity: {audit_data.get('activityType', 'N/A')}")
        print(f"  Endpoint: {audit_data.get('endPoint', 'N/A')}")
        print(f"  Request ID: {audit_data.get('requestId', 'N/A')}")
        print(f"  Client IP: {audit_data.get('clientIp', 'N/A')}")
        
        # Parse remarks for key details
        try:
            remarks = json.loads(audit_data.get('remarks', '{}'))
            print(f"  Status Code: {remarks.get('status_code', 'N/A')}")
            print(f"  Processing Time: {remarks.get('processing_time_ms', 'N/A')}ms")
            print(f"  Success: {remarks.get('success', 'N/A')}")
            
            if 'full_url' in remarks:
                print(f"  Full URL: {remarks['full_url']}")
        except:
            print(f"  Remarks: {audit_data.get('remarks', 'N/A')[:100]}...")

def verify_message_format(verifier: AuditLogVerifier, messages: List[Dict]):
    """Verify message format compliance"""
    print(f"\n MESSAGE FORMAT VERIFICATION")
    print("=" * 50)
    
    if not messages:
        print("No messages to verify")
        return
    
    all_valid = True
    
    for i, message in enumerate(messages, 1):
        audit_data = message['audit_data']
        verification = verifier.verify_audit_format(audit_data)
        
        status_icon = "✅" if verification['valid'] else "❌"
        print(f"{status_icon} Message {i} (ID: {message['message_id'][:8]}...)")
        
        if not verification['valid']:
            all_valid = False
            if verification['missing_fields']:
                print(f"     Missing fields: {', '.join(verification['missing_fields'])}")
            if verification['format_issues']:
                print(f"     Format issues: {', '.join(verification['format_issues'])}")
        
        # Show field analysis for first message
        if i == 1:
            print("     Field Analysis:")
            for field, analysis in verification['field_analysis'].items():
                print(f"       {field}: {analysis['type']} ({analysis['length']} chars)")
    
    if all_valid:
        print("\n All messages have valid format!")
    else:
        print("\n  Some messages have format issues")

def main():
    """Main verification function"""
    print(" AUDIT LOG VERIFICATION UTILITY")
    print("=" * 60)
    print("This utility helps verify that audit logs are being")
    print("sent correctly to your SQS queue.\n")
    
    # Initialize verifier
    verifier = AuditLogVerifier()
    
    if not verifier.sqs_client:
        print(" Cannot connect to SQS. Please check your configuration:")
        print("   - AUDIT_SQS_URL environment variable")
        print("   - AWS credentials (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)")
        print("   - AWS_REGION environment variable")
        return
    
    # Check queue status
    print_queue_status(verifier)
    
    # Receive and display recent messages
    print("\n Receiving recent audit messages...")
    messages = verifier.receive_recent_messages(max_messages=5)
    
    if messages:
        print_audit_messages(messages)
        verify_message_format(verifier, messages)
        
        # Ask if user wants to delete messages (optional)
        print("\n Options:")
        print("1. Keep messages in queue for further processing")
        print("2. Delete messages from queue (they've been verified)")
        
        choice = input("Enter choice (1 or 2, default=1): ").strip()
        
        if choice == "2":
            print("  Deleting verified messages...")
            for message in messages:
                success = verifier.delete_message(message['receipt_handle'])
                if success:
                    print(f"    Deleted message {message['message_id'][:8]}...")
                else:
                    print(f"    Failed to delete message {message['message_id'][:8]}...")
    
    print("\n✨ Verification completed!")
    print("\nNext steps:")
    print("1. Run your API tests to generate audit logs")
    print("2. Run this script again to verify new messages")
    print("3. Check that all expected endpoints are being captured")
    print("4. Verify message format meets your compliance requirements")

if __name__ == "__main__":
    main()