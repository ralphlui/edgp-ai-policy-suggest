"""Async LLM processing utilities"""
import asyncio
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
import logging
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)

async def process_single(llm: ChatOpenAI, prompt: str) -> str:
    """Process a single prompt with the LLM"""
    try:
        messages = [HumanMessage(content=prompt)]
        response = await llm.ainvoke(messages)
        return response.content.strip() if response else "[]"
    except Exception as e:
        logger.error(f"Error processing prompt: {e}")
        return "[]"

async def process_batch_async(
    llm: ChatOpenAI, 
    prompts: List[str],
    max_concurrent: int = 5  # Increased concurrency
) -> List[str]:
    """Process multiple prompts concurrently with rate limiting"""
    
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def _bounded_process(prompt: str) -> str:
        async with semaphore:
            return await process_single(llm, prompt)
    
    try:
        results = await asyncio.gather(
            *[_bounded_process(p) for p in prompts],
            return_exceptions=True
        )
        
        # Handle any exceptions
        processed = []
        for r in results:
            if isinstance(r, Exception):
                logger.error(f"Batch processing error: {r}")
                processed.append("[]")
            else:
                processed.append(r)
        return processed
        
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        return ["[]" * len(prompts)]