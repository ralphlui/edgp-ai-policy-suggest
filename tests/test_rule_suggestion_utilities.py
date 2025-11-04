"""
Simplified comprehensive test suite for rule suggestion routes utility functions.
Focuses on increasing coverage for utility functions and business logic.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import time

class TestUtilityFunctions:
    """Test utility functions in rule suggestion routes."""
    
    def test_calculate_overall_confidence_with_good_metrics(self):
        """Test confidence calculation with good execution metrics."""
        from app.api.rule_suggestion_routes import _calculate_overall_confidence
        
        # Create mock state with good metrics
        state = Mock()
        state.errors = []
        state.step_history = ["step1", "step2", "step3"]
        state.execution_metrics = {"total_execution_time": 2.0}
        state.rule_suggestions = ["rule1", "rule2", "rule3", "rule4", "rule5"]
        state.data_schema = {"domain": "test", "col1": "string", "col2": "int"}
        
        confidence = _calculate_overall_confidence(state)
        
        # Should be high confidence with good metrics
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.7  # Good metrics should yield high confidence

    def test_calculate_overall_confidence_with_errors(self):
        """Test confidence calculation with errors present."""
        from app.api.rule_suggestion_routes import _calculate_overall_confidence
        
        # Create mock state with errors
        state = Mock()
        state.errors = ["error1", "error2", "error3"]
        state.step_history = ["step1", "step2", "step3"]
        state.execution_metrics = {"total_execution_time": 5.0}
        state.rule_suggestions = ["rule1"]
        state.data_schema = {"domain": "test"}
        
        confidence = _calculate_overall_confidence(state)
        
        # Should be lower confidence with errors
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0

    def test_get_confidence_level_mapping(self):
        """Test confidence level string mapping."""
        from app.api.rule_suggestion_routes import _get_confidence_level
        
        # Test with different confidence states
        high_confidence_state = Mock()
        high_confidence_state.errors = []
        high_confidence_state.step_history = ["step1", "step2", "step3"]
        high_confidence_state.execution_metrics = {"total_execution_time": 1.5}
        high_confidence_state.rule_suggestions = ["rule1", "rule2", "rule3", "rule4", "rule5", "rule6"]
        high_confidence_state.data_schema = {"domain": "test"}
        
        level = _get_confidence_level(high_confidence_state)
        assert level in ["high", "medium", "low", "very_low"]

    def test_get_confidence_factors_detailed_breakdown(self):
        """Test detailed confidence factors breakdown."""
        from app.api.rule_suggestion_routes import _get_confidence_factors
        
        # Create comprehensive mock state
        state = Mock()
        state.errors = ["error1"]
        state.step_history = ["step1", "step2", "step3"]
        state.execution_metrics = {"total_execution_time": 3.5}
        state.rule_suggestions = ["rule1", "rule2", "rule3"]
        state.thoughts = ["thought1", "thought2", "thought3", "thought4"]
        state.observations = ["obs1", "obs2"]
        state.reflections = ["ref1"]
        
        factors = _get_confidence_factors(state)
        
        # Check structure
        assert "rule_generation" in factors
        assert "error_handling" in factors
        assert "execution_performance" in factors
        assert "reasoning_depth" in factors
        
        # Check rule generation details
        assert factors["rule_generation"]["rules_generated"] == 3
        assert "status" in factors["rule_generation"]
        assert "score" in factors["rule_generation"]
        
        # Check error handling details
        assert factors["error_handling"]["errors_encountered"] == 1
        assert "status" in factors["error_handling"]
        
        # Check execution performance
        assert factors["execution_performance"]["duration_seconds"] == 3.5
        assert "status" in factors["execution_performance"]
        
        # Check reasoning depth
        assert factors["reasoning_depth"]["thoughts_generated"] == 4
        assert factors["reasoning_depth"]["observations_made"] == 2
        assert factors["reasoning_depth"]["reflections_completed"] == 1

    def test_build_execution_trace_comprehensive(self):
        """Test comprehensive execution trace building."""
        from app.api.rule_suggestion_routes import _build_execution_trace
        
        # Create detailed mock state
        state = Mock()
        state.execution_start_time = time.time() - 10
        state.gx_rules = [{"rule1": "data"}, {"rule2": "data"}]
        state.raw_suggestions = "Rule 1\nRule 2\nRule 3"
        state.formatted_rules = [{"formatted1": "data"}, {"formatted2": "data"}]
        state.normalized_suggestions = {"norm1": "data", "norm2": "data"}
        state.rule_suggestions = [{"final1": "data"}, {"final2": "data"}, {"final3": "data"}]
        state.data_schema = {"domain": "test", "col1": "string", "col2": "int"}
        state.thoughts = ["thought1", "thought2"]
        state.observations = ["obs1"]
        state.step_history = []
        state.errors = []
        state.execution_metrics = {"total_execution_time": 2.5}
        
        trace = _build_execution_trace(state)
        
        # Check structure
        assert "workflow_path" in trace
        assert "tool_invocations" in trace
        assert "reasoning_chain" in trace
        assert "observations" in trace
        assert "execution_metrics" in trace
        
        # Check tool invocations
        assert len(trace["tool_invocations"]) > 0
        
        # Check each tool invocation has required fields
        for invocation in trace["tool_invocations"]:
            assert "tool" in invocation
            assert "status" in invocation
            assert "timestamp" in invocation
            
        # Check execution metrics
        metrics = trace["execution_metrics"]
        assert "total_duration_ms" in metrics
        assert "final_rule_count" in metrics
        assert metrics["final_rule_count"] == 3


class TestLoggingAndSanitization:
    """Test logging and sanitization functions."""
    
    def test_sanitize_for_logging_various_inputs(self):
        """Test sanitization function with various inputs."""
        from app.api.rule_suggestion_routes import sanitize_for_logging
        
        # Test normal string
        assert sanitize_for_logging("normal_string") == "normal_string"
        
        # Test string with control characters
        result = sanitize_for_logging("test\x00\x1F")
        assert "\x00" not in result
        assert "\x1F" not in result
        
        # Test string with special characters
        result = sanitize_for_logging("test(){}[]<>'\"\\")
        assert "(" not in result or result.count("_") > 0
        
        # Test very long string
        long_string = "a" * 200
        result = sanitize_for_logging(long_string)
        assert len(result) <= 100
        
        # Test non-string input
        result = sanitize_for_logging(123)
        assert result == '<non-string>'

    @patch('app.api.rule_suggestion_routes.logger')
    def test_log_domain_operation(self, mock_logger):
        """Test domain operation logging."""
        from app.api.rule_suggestion_routes import log_domain_operation
        
        # Test with details
        log_domain_operation("TEST_OP", "test_domain", "test details")
        mock_logger.info.assert_called_with("TEST_OP - domain: test_domain - test details")
        
        # Test without details
        mock_logger.reset_mock()
        log_domain_operation("TEST_OP", "test_domain")
        mock_logger.info.assert_called_with("TEST_OP - domain: test_domain")

    @patch('app.api.rule_suggestion_routes.logger')
    def test_log_error(self, mock_logger):
        """Test error logging."""
        from app.api.rule_suggestion_routes import log_error
        
        error = Exception("test error message")
        log_error("TEST_OP", "test_domain", error)
        mock_logger.error.assert_called_with("TEST_OP failed - domain: test_domain - error: test error message")

    @patch('app.api.rule_suggestion_routes.logger')
    @patch('time.time')
    def test_log_duration_context_manager(self, mock_time, mock_logger):
        """Test duration logging context manager."""
        from app.api.rule_suggestion_routes import log_duration
        
        # Mock time to simulate duration
        mock_time.side_effect = [0.0, 2.5]
        
        with log_duration("test_operation"):
            pass  # Simulate some work
        
        mock_logger.info.assert_called_with(" test_operation took 2.50s")


if __name__ == "__main__":
    pytest.main([__file__])