#!/usr/bin/env python3
"""
Test script to debug OpenSearch bulk indexing issues
"""
import os
import sys
import traceback
import asyncio

def test_opensearch_bulk_indexing():
    print(" Testing OpenSearch bulk indexing...")
    
    # Test 1: Check basic OpenSearch connection
    print("\n Testing OpenSearch connection...")
    try:
        from app.aoss.aoss_client import create_aoss_client
        client = create_aoss_client()
        info = client.info()
        print(f" OpenSearch connection successful")
        print(f"   Cluster: {info.get('cluster_name', 'Unknown')}")
        print(f"   Version: {info.get('version', {}).get('number', 'Unknown')}")
    except Exception as e:
        print(f" OpenSearch connection failed: {e}")
        return

    # Test 2: Test embedding generation
    print("\n  Testing embedding generation...")
    try:
        from app.embedding.embedder import embed_column_names_batched_async
        test_columns = ["name", "email"]
        embeddings = asyncio.run(embed_column_names_batched_async(test_columns))
        print(f" Generated embeddings for {len(test_columns)} columns")
        print(f"   Embedding dimensions: {[len(emb) for emb in embeddings]}")
        print(f"   Expected dimension: 1536")
        
        # Check if dimensions match
        for i, emb in enumerate(embeddings):
            if len(emb) != 1536:
                print(f"⚠️ Warning: Embedding {i} has dimension {len(emb)}, expected 1536")
    except Exception as e:
        print(f" Embedding generation failed: {e}")
        traceback.print_exc()
        return

    # Test 3: Test ColumnDoc creation
    print("\n Testing ColumnDoc creation...")
    try:
        from app.aoss.column_store import ColumnDoc
        
        docs = []
        for i, col_name in enumerate(test_columns):
            doc = ColumnDoc(
                column_id=f"test.{col_name}",
                column_name=col_name,
                embedding=embeddings[i],
                sample_values=["sample1", "sample2"],
                metadata={
                    "domain": "test",
                    "type": "string",
                    "pii": False,
                    "table": "test_table",
                    "source": "test"
                }
            )
            docs.append(doc)
        
        print(f" Created {len(docs)} ColumnDoc objects")
        
        # Validate document structure
        for i, doc in enumerate(docs):
            doc_dict = doc.to_doc()
            print(f"   Doc {i}: column_id={doc.column_id}, embedding_dim={len(doc.embedding)}")
            
    except Exception as e:
        print(f" ColumnDoc creation failed: {e}")
        traceback.print_exc()
        return

    # Test 4: Test OpenSearchColumnStore initialization
    print("\n Testing OpenSearchColumnStore initialization...")
    try:
        from app.aoss.column_store import OpenSearchColumnStore
        from app.core.config import settings
        
        store = OpenSearchColumnStore(index_name="test-columns")
        print(f" OpenSearchColumnStore initialized")
        print(f"   Index name: {store.index_name}")
        print(f"   Embedding dimension: {store.embedding_dim}")
        
    except Exception as e:
        print(f" OpenSearchColumnStore initialization failed: {e}")
        traceback.print_exc()
        return

    # Test 5: Test index creation
    print("\n Testing index creation...")
    try:
        store.ensure_index()
        print(f" Index creation/verification successful")
    except Exception as e:
        print(f" Index creation failed: {e}")
        traceback.print_exc()
        return

    # Test 6: Test single document upsert
    print("\n Testing single document upsert...")
    try:
        single_doc = docs[0]  # Use first document
        store.upsert_columns([single_doc])
        print(f" Single document upsert successful")
    except Exception as e:
        print(f" Single document upsert failed: {e}")
        traceback.print_exc()
        return

    # Test 7: Test bulk upsert
    print("\n Testing bulk document upsert...")
    try:
        store.upsert_columns(docs)
        print(f" Bulk upsert successful")
    except Exception as e:
        print(f" Bulk upsert failed: {e}")
        traceback.print_exc()
        return

    # Test 8: Clean up test index
    print("\n Cleaning up test index...")
    try:
        if client.indices.exists(index="test-columns"):
            client.indices.delete(index="test-columns")
            print(f" Test index cleaned up")
    except Exception as e:
        print(f" Test index cleanup failed: {e}")

    print("\n All OpenSearch bulk indexing tests passed!")

if __name__ == "__main__":
    test_opensearch_bulk_indexing()