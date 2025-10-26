"""
Tests for app/core/timeout_config.py module
"""
import pytest
from app.core.timeout_config import (
    LLM_TIMEOUT_SECONDS,
    VECTOR_DB_TIMEOUT,
    AGENT_STEP_TIMEOUT,
    TOTAL_REQUEST_TIMEOUT,
    LLM_MAX_TOKENS
)

class TestTimeoutConfig:
    """Test timeout_config.py configuration values"""

    def test_timeout_values_exist(self):
        """Test that all timeout values are defined and not None"""
        assert LLM_TIMEOUT_SECONDS is not None
        assert VECTOR_DB_TIMEOUT is not None
        assert AGENT_STEP_TIMEOUT is not None
        assert TOTAL_REQUEST_TIMEOUT is not None
        assert LLM_MAX_TOKENS is not None

    def test_timeout_values_are_positive(self):
        """Test that all timeout values are positive numbers"""
        assert LLM_TIMEOUT_SECONDS > 0, "LLM timeout must be positive"
        assert VECTOR_DB_TIMEOUT > 0, "Vector DB timeout must be positive"
        assert AGENT_STEP_TIMEOUT > 0, "Agent step timeout must be positive"
        assert TOTAL_REQUEST_TIMEOUT > 0, "Total request timeout must be positive"
        assert LLM_MAX_TOKENS > 0, "LLM max tokens must be positive"

    def test_timeout_values_are_integers(self):
        """Test that all timeout values are integers"""
        assert isinstance(LLM_TIMEOUT_SECONDS, int), "LLM timeout must be an integer"
        assert isinstance(VECTOR_DB_TIMEOUT, int), "Vector DB timeout must be an integer"
        assert isinstance(AGENT_STEP_TIMEOUT, int), "Agent step timeout must be an integer"
        assert isinstance(TOTAL_REQUEST_TIMEOUT, int), "Total request timeout must be an integer"
        assert isinstance(LLM_MAX_TOKENS, int), "LLM max tokens must be an integer"

    def test_timeout_values_are_reasonable(self):
        """Test that timeout values are within reasonable ranges"""
        # LLM timeout should be between 1 and 60 seconds
        assert 1 <= LLM_TIMEOUT_SECONDS <= 60, "LLM timeout should be between 1 and 60 seconds"
        
        # Vector DB timeout should be between 1 and 30 seconds
        assert 1 <= VECTOR_DB_TIMEOUT <= 30, "Vector DB timeout should be between 1 and 30 seconds"
        
        # Agent step timeout should be between 1 and 30 seconds
        assert 1 <= AGENT_STEP_TIMEOUT <= 30, "Agent step timeout should be between 1 and 30 seconds"
        
        # Total request timeout should be between 5 and 120 seconds
        assert 5 <= TOTAL_REQUEST_TIMEOUT <= 120, "Total request timeout should be between 5 and 120 seconds"
        
        # LLM max tokens should be between 100 and 4096
        assert 100 <= LLM_MAX_TOKENS <= 4096, "LLM max tokens should be between 100 and 4096"

    def test_total_timeout_exceeds_individual_timeouts(self):
        """Test that total request timeout is greater than individual timeouts"""
        # Total timeout should exceed LLM timeout
        assert TOTAL_REQUEST_TIMEOUT > LLM_TIMEOUT_SECONDS, \
            "Total timeout should be greater than LLM timeout"
        
        # Total timeout should exceed vector DB timeout
        assert TOTAL_REQUEST_TIMEOUT > VECTOR_DB_TIMEOUT, \
            "Total timeout should be greater than vector DB timeout"
        
        # Total timeout should exceed agent step timeout
        assert TOTAL_REQUEST_TIMEOUT > AGENT_STEP_TIMEOUT, \
            "Total timeout should be greater than agent step timeout"

    def test_llm_timeout_reasonable_for_token_count(self):
        """Test that LLM timeout is reasonable for the max token count"""
        # Assuming processing speed of at least 100 tokens per second
        min_required_timeout = LLM_MAX_TOKENS / 100
        assert LLM_TIMEOUT_SECONDS >= min_required_timeout, \
            f"LLM timeout ({LLM_TIMEOUT_SECONDS}s) may be too short for {LLM_MAX_TOKENS} tokens"

    def test_timeout_precedence(self):
        """Test the logical precedence of timeout values"""
        # Individual timeouts should be less than total timeout
        individual_timeouts = [LLM_TIMEOUT_SECONDS, VECTOR_DB_TIMEOUT, AGENT_STEP_TIMEOUT]
        assert max(individual_timeouts) < TOTAL_REQUEST_TIMEOUT, \
            "Individual timeouts should be less than total timeout"
        
        # Sum of core timeouts should be less than or equal to total timeout
        # Adding some buffer for overhead
        core_timeout_sum = LLM_TIMEOUT_SECONDS + VECTOR_DB_TIMEOUT + AGENT_STEP_TIMEOUT
        assert core_timeout_sum <= TOTAL_REQUEST_TIMEOUT, \
            "Sum of core timeouts should not exceed total timeout"

    @pytest.mark.parametrize("timeout_value,min_value,max_value,name", [
        (LLM_TIMEOUT_SECONDS, 1, 60, "LLM_TIMEOUT_SECONDS"),
        (VECTOR_DB_TIMEOUT, 1, 30, "VECTOR_DB_TIMEOUT"),
        (AGENT_STEP_TIMEOUT, 1, 30, "AGENT_STEP_TIMEOUT"),
        (TOTAL_REQUEST_TIMEOUT, 5, 120, "TOTAL_REQUEST_TIMEOUT"),
        (LLM_MAX_TOKENS, 100, 4096, "LLM_MAX_TOKENS")
    ])
    def test_timeout_bounds(self, timeout_value, min_value, max_value, name):
        """Test that all timeout values are within their specified bounds"""
        assert min_value <= timeout_value <= max_value, \
            f"{name} ({timeout_value}) should be between {min_value} and {max_value}"