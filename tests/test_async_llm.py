"""
Tests for app/core/async_llm.py module
"""
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage
from app.core.async_llm import process_single, process_batch_async

class TestAsyncLLM:
    """Test async_llm.py functionality"""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM with async capabilities"""
        mock = Mock(spec=ChatOpenAI)
        mock.ainvoke = AsyncMock()
        return mock

    @pytest.mark.asyncio
    async def test_process_single_success(self, mock_llm):
        """Test successful processing of a single prompt"""
        # Setup
        mock_llm.ainvoke.return_value = AIMessage(content="Test response")
        
        # Execute
        result = await process_single(mock_llm, "Test prompt")
        
        # Verify
        assert result == "Test response"
        mock_llm.ainvoke.assert_called_once()
        args = mock_llm.ainvoke.call_args[0][0]
        assert len(args) == 1
        assert args[0].content == "Test prompt"

    @pytest.mark.asyncio
    async def test_process_single_empty_response(self, mock_llm):
        """Test handling of empty response from LLM"""
        # Setup
        mock_llm.ainvoke.return_value = AIMessage(content="")
        
        # Execute
        result = await process_single(mock_llm, "Test prompt")
        
        # Verify
        assert result == ""  # Empty response should be preserved
        mock_llm.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_single_error(self, mock_llm):
        """Test error handling in single prompt processing"""
        # Setup
        mock_llm.ainvoke.side_effect = Exception("Test error")
        
        # Execute
        result = await process_single(mock_llm, "Test prompt")
        
        # Verify
        assert result == "[]"
        mock_llm.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_batch_success(self, mock_llm):
        """Test successful processing of multiple prompts"""
        # Setup
        prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
        responses = ["Response 1", "Response 2", "Response 3"]
        mock_llm.ainvoke.side_effect = [
            AIMessage(content=resp) for resp in responses
        ]
        
        # Execute
        results = await process_batch_async(mock_llm, prompts)
        
        # Verify
        assert results == responses
        assert mock_llm.ainvoke.call_count == len(prompts)

    @pytest.mark.asyncio
    async def test_process_batch_with_errors(self, mock_llm):
        """Test batch processing with mixed success and errors"""
        # Setup
        prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
        mock_llm.ainvoke.side_effect = [
            AIMessage(content="Success"),
            Exception("Error"),
            AIMessage(content="Success again")
        ]
        
        # Execute
        results = await process_batch_async(mock_llm, prompts)
        
        # Verify
        assert len(results) == len(prompts)
        assert results[0] == "Success"
        assert results[1] == "[]"  # Error case
        assert results[2] == "Success again"
        assert mock_llm.ainvoke.call_count == len(prompts)

    @pytest.mark.asyncio
    async def test_process_batch_rate_limiting(self, mock_llm):
        """Test that batch processing respects rate limiting"""
        # Setup
        prompts = ["P1", "P2", "P3", "P4", "P5", "P6"]
        max_concurrent = 2
        completion_times = []
        
        async def delayed_response(messages):
            await asyncio.sleep(0.1)  # Simulate API delay
            completion_times.append(asyncio.get_event_loop().time())
            return AIMessage(content="Response")
            
        mock_llm.ainvoke.side_effect = delayed_response
        
        # Execute
        start_time = asyncio.get_event_loop().time()
        results = await process_batch_async(mock_llm, prompts, max_concurrent=max_concurrent)
        
        # Verify
        assert len(results) == len(prompts)
        
        # Check that completions were rate-limited
        time_groups = []
        current_group = []
        for t in completion_times:
            if not current_group or t - current_group[-1] < 0.05:  # Group times within 50ms
                current_group.append(t)
            else:
                time_groups.append(current_group)
                current_group = [t]
        if current_group:
            time_groups.append(current_group)
            
        # Verify no group is larger than max_concurrent
        for group in time_groups:
            assert len(group) <= max_concurrent

    @pytest.mark.asyncio
    async def test_process_batch_empty_prompts(self, mock_llm):
        """Test batch processing with empty prompt list"""
        results = await process_batch_async(mock_llm, [])
        assert results == []
        mock_llm.ainvoke.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_batch_all_errors(self, mock_llm):
        """Test batch processing when all prompts result in errors"""
        # Setup
        prompts = ["P1", "P2", "P3"]
        mock_llm.ainvoke.side_effect = Exception("Test error")
        
        # Execute
        results = await process_batch_async(mock_llm, prompts)
        
        # Verify
        assert all(r == "[]" for r in results)
        assert len(results) == len(prompts)
        assert mock_llm.ainvoke.call_count == len(prompts)

    @pytest.mark.asyncio
    async def test_process_batch_catastrophic_failure(self, mock_llm):
        """Test handling of catastrophic failure in batch processing"""
        # Setup
        prompts = ["P1", "P2", "P3"]
        mock_llm.ainvoke.side_effect = SystemError("Catastrophic failure")
        
        # Execute
        results = await process_batch_async(mock_llm, prompts)
        
        # Verify
        assert all(r == "[]" for r in results)
        assert len(results) == len(prompts)

    @pytest.mark.asyncio
    async def test_process_batch_timeout(self, mock_llm):
        """Test batch processing with timeouts"""
        # Setup
        prompts = ["P1", "P2"]
        
        async def slow_response(messages):
            await asyncio.sleep(0.5)  # Simulate slow response
            return AIMessage(content="Response")
            
        mock_llm.ainvoke.side_effect = slow_response
        
        # Set a shorter timeout for the test
        with pytest.raises(asyncio.TimeoutError):
            async with asyncio.timeout(0.1):
                await process_batch_async(mock_llm, prompts)

    @pytest.mark.asyncio
    async def test_process_batch_concurrent_limit(self, mock_llm):
        """Test that concurrent processing limit is respected"""
        # Setup
        max_concurrent = 3
        num_prompts = 10
        active_tasks = 0
        max_observed_tasks = 0
        lock = asyncio.Lock()
        
        async def tracked_response(messages):
            nonlocal active_tasks, max_observed_tasks
            async with lock:
                active_tasks += 1
                max_observed_tasks = max(max_observed_tasks, active_tasks)
            await asyncio.sleep(0.1)  # Simulate work
            async with lock:
                active_tasks -= 1
            return AIMessage(content="Response")
            
        mock_llm.ainvoke.side_effect = tracked_response
        
        # Execute
        prompts = [f"P{i}" for i in range(num_prompts)]
        await process_batch_async(mock_llm, prompts, max_concurrent=max_concurrent)
        
        # Verify
        assert max_observed_tasks <= max_concurrent