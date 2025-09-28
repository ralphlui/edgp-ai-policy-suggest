#!/usr/bin/env python3
"""
Test script to isolate embedding initialization issues
"""
import os
import sys
import traceback

def test_openai_embeddings():
    print("🧪 Testing OpenAI embeddings initialization...")
    
    # Test 1: Check environment loading
    print("\n1️⃣ Testing environment loading...")
    try:
        from app.core.config import OPENAI_API_KEY
        print(f"✅ OPENAI_API_KEY loaded: {'set' if OPENAI_API_KEY else 'missing'}")
        if OPENAI_API_KEY:
            print(f"   Key starts with: {OPENAI_API_KEY[:8]}...")
            print(f"   Key length: {len(OPENAI_API_KEY)}")
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        return
    
    # Test 2: Test OpenAI client directly
    print("\n2️⃣ Testing OpenAI client initialization...")
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        print("✅ OpenAI client initialized successfully")
    except Exception as e:
        print(f"❌ OpenAI client initialization failed: {e}")
        traceback.print_exc()
        return
    
    # Test 3: Test LangChain OpenAI embeddings
    print("\n3️⃣ Testing LangChain OpenAIEmbeddings...")
    try:
        from langchain_openai import OpenAIEmbeddings
        embedder = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)
        print("✅ LangChain OpenAIEmbeddings initialized successfully")
    except Exception as e:
        print(f"❌ LangChain OpenAIEmbeddings initialization failed: {e}")
        traceback.print_exc()
        return
    
    # Test 4: Test actual embedding generation
    print("\n4️⃣ Testing embedding generation...")
    try:
        test_text = ["hello", "world"]
        embeddings = embedder.embed_documents(test_text)
        print(f"✅ Embeddings generated successfully")
        print(f"   Generated {len(embeddings)} embeddings")
        print(f"   First embedding dimension: {len(embeddings[0])}")
    except Exception as e:
        print(f"❌ Embedding generation failed: {e}")
        traceback.print_exc()
        return
    
    # Test 5: Test our embedder function
    print("\n5️⃣ Testing our embedder function...")
    try:
        from app.embedding.embedder import get_embedder, _embed_batch
        our_embedder = get_embedder()
        print("✅ Our embedder function works")
        
        batch_result = _embed_batch(["test", "column"])
        print(f"✅ Batch embedding works: {len(batch_result)} embeddings generated")
    except Exception as e:
        print(f"❌ Our embedder function failed: {e}")
        traceback.print_exc()
        return
    
    # Test 6: Test async function
    print("\n6️⃣ Testing async embedding function...")
    try:
        import asyncio
        from app.embedding.embedder import embed_column_names_batched_async
        
        async def test_async():
            result = await embed_column_names_batched_async(["test", "async", "function"])
            return result
        
        async_result = asyncio.run(test_async())
        print(f"✅ Async embedding works: {len(async_result)} embeddings generated")
    except Exception as e:
        print(f"❌ Async embedding failed: {e}")
        traceback.print_exc()
        return
    
    print("\n🎉 All embedding tests passed!")

if __name__ == "__main__":
    test_openai_embeddings()