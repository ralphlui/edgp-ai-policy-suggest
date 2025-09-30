#!/usr/bin/env python3
"""
AWS OpenSearch Serverless Connection Test Script

This script tests your AWS credentials and AOSS permissions.
Run this to diagnose authentication issues before running your main application.

Usage:
    python test_aws_connection.py
"""

import os
import sys
import boto3
from opensearchpy import OpenSearch, AWSV4SignerAuth
import json
from datetime import datetime

def print_header(title):
    """Print a formatted header"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)

def test_aws_credentials():
    """Test basic AWS credential resolution"""
    print_header("Testing AWS Credentials")
    
    try:
        # Test STS (Security Token Service) call
        sts = boto3.client('sts')
        identity = sts.get_caller_identity()
        
        print(" AWS Credentials Working!")
        print(f"Account ID: {identity['Account']}")
        print(f"User ARN: {identity['Arn']}")
        print(f"User ID: {identity['UserId']}")
        
        return True, identity
        
    except Exception as e:
        print(f" AWS Credential Test Failed: {e}")
        print("\n Solutions:")
        print("1. Set environment variables:")
        print("   export AWS_ACCESS_KEY_ID=your_key")
        print("   export AWS_SECRET_ACCESS_KEY=your_secret")
        print("2. Or run: aws configure")
        print("3. Or attach IAM role to your instance")
        return False, None

def test_aoss_permissions(identity_arn):
    """Test OpenSearch Serverless permissions"""
    print_header("Testing OpenSearch Serverless Permissions")
    
    region = os.getenv("AWS_REGION", "ap-southeast-1")
    host = os.getenv("AOSS_HOST", "lysn7ssghohujq3wlxa3.ap-southeast-1.aoss.amazonaws.com")
    
    print(f"Testing connection to: {host}")
    print(f"Region: {region}")
    print(f"Using identity: {identity_arn}")
    
    try:
        # Create AOSS client
        session = boto3.Session()
        credentials = session.get_credentials()
        awsauth = AWSV4SignerAuth(credentials, region, 'aoss')
        
        client = OpenSearch(
            hosts=[{'host': host, 'port': 443}],
            http_auth=awsauth,
            use_ssl=True,
            verify_certs=True,
            timeout=30
        )
        
        # Test basic cluster info
        info = client.info()
        print(" AOSS Connection Successful!")
        print(f"Cluster Name: {info.get('cluster_name', 'Unknown')}")
        print(f"Version: {info.get('version', {}).get('number', 'Unknown')}")
        
        return True, client
        
    except Exception as e:
        print(f" AOSS Connection Failed: {e}")
        print(f"\nError type: {type(e).__name__}")
        
        if "AuthorizationException" in str(e):
            print("\n Authorization Error Solutions:")
            print("1. Add IAM policy to your user with AOSS permissions")
            print("2. Create/update AOSS Data Access Policy")
            print("3. Include your user ARN in the data access policy:")
            print(f"   {identity_arn}")
            
        elif "Forbidden" in str(e):
            print("\n Forbidden Error Solutions:")
            print("1. Check AOSS Data Access Policy")
            print("2. Ensure your user ARN is in the principals list")
            
        elif "timeout" in str(e).lower():
            print("\n Timeout Error Solutions:")
            print("1. Check network connectivity")
            print("2. Verify the AOSS host is correct")
            print("3. Check security group rules")
            
        return False, None

def test_index_operations(client):
    """Test basic index operations"""
    print_header("Testing Index Operations")
    
    test_index = "test-connection-index"
    
    try:
        # Test index creation
        index_body = {
            "settings": {
                "index": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0
                }
            },
            "mappings": {
                "properties": {
                    "test_field": {"type": "text"},
                    "timestamp": {"type": "date"}
                }
            }
        }
        
        # Create index if it doesn't exist
        if not client.indices.exists(index=test_index):
            client.indices.create(index=test_index, body=index_body)
            print(f" Created test index: {test_index}")
        else:
            print(f" Test index already exists: {test_index}")
        
        # Test document insertion
        doc = {
            "test_field": "Hello from connection test",
            "timestamp": datetime.now().isoformat()
        }
        
        result = client.index(index=test_index, id="test-doc", body=doc)
        print(f" Document inserted successfully: {result['result']}")
        
        # Test document retrieval
        retrieved = client.get(index=test_index, id="test-doc")
        print(f" Document retrieved successfully: {retrieved['found']}")
        
        # Clean up test index
        client.indices.delete(index=test_index)
        print(f" Cleaned up test index: {test_index}")
        
        return True
        
    except Exception as e:
        print(f" Index Operations Failed: {e}")
        
        # Try to clean up even if operations failed
        try:
            if client.indices.exists(index=test_index):
                client.indices.delete(index=test_index)
                print(f"🧹 Cleaned up test index after failure")
        except:
            pass
            
        return False

def generate_iam_policy(account_id):
    """Generate the required IAM policy"""
    print_header("Required IAM Policy")
    
    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "aoss:ReadDocument",
                    "aoss:WriteDocument", 
                    "aoss:CreateIndex",
                    "aoss:DeleteIndex",
                    "aoss:UpdateIndex",
                    "aoss:DescribeIndex",
                    "aoss:APIAccessAll"
                ],
                "Resource": f"arn:aws:aoss:*:{account_id}:collection/*"
            },
            {
                "Effect": "Allow",
                "Action": [
                    "aoss:ListCollections",
                    "aoss:BatchGetCollection"
                ],
                "Resource": "*"
            }
        ]
    }
    
    print("Add this IAM policy to your user:")
    print(json.dumps(policy, indent=2))

def generate_data_access_policy(user_arn):
    """Generate the required AOSS Data Access Policy"""
    print_header("Required AOSS Data Access Policy")
    
    policy = [
        {
            "Rules": [
                {
                    "ResourceType": "index",
                    "Resource": ["index/*/*"],
                    "Permission": [
                        "aoss:CreateIndex",
                        "aoss:DeleteIndex",
                        "aoss:UpdateIndex",
                        "aoss:DescribeIndex",
                        "aoss:ReadDocument",
                        "aoss:WriteDocument"
                    ]
                },
                {
                    "ResourceType": "collection",
                    "Resource": ["collection/*"],
                    "Permission": [
                        "aoss:CreateCollectionItems",
                        "aoss:UpdateCollectionItems"
                    ]
                }
            ],
            "Principal": [user_arn]
        }
    ]
    
    print("Create this AOSS Data Access Policy:")
    print(json.dumps(policy, indent=2))

def main():
    """Main test function"""
    print("🔧 AWS OpenSearch Serverless Connection Diagnostic")
    print(f"Timestamp: {datetime.now()}")
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=".env.development")
    
    # Test 1: AWS Credentials
    creds_ok, identity = test_aws_credentials()
    if not creds_ok:
        sys.exit(1)
    
    # Test 2: AOSS Connection
    aoss_ok, client = test_aoss_permissions(identity['Arn'])
    if not aoss_ok:
        print_header("Setup Instructions")
        generate_iam_policy(identity['Account'])
        generate_data_access_policy(identity['Arn'])
        sys.exit(1)
    
    # Test 3: Index Operations
    ops_ok = test_index_operations(client)
    if not ops_ok:
        print("\n Basic connection works but index operations failed")
        print("Check your data access policy permissions")
        sys.exit(1)
    
    print_header(" All Tests Passed!")
    print("Your AWS setup is working correctly!")
    print("You can now run your main application.")

if __name__ == "__main__":
    main()