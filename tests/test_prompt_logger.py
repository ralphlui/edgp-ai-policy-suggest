"""
Tests for app/prompt/prompt_logger.py module
"""
import pytest
from logging import Logger
from app.prompt.prompt_logger import PromptLogger
import logging
from unittest.mock import patch, Mock, call

class TestPromptLogger:
    """Test PromptLogger class functionality"""

    @pytest.fixture(autouse=True)
    def setup_logger(self):
        """Set up logger for each test"""
        with patch('app.prompt.prompt_logger.logger') as mock_logger:
            self.mock_logger = mock_logger
            yield

    def test_log_prompt_generation_basic(self):
        """Test basic prompt generation logging without context"""
        PromptLogger.log_prompt_generation("test", "customer")
        
        self.mock_logger.info.assert_called_once_with(
            "Generating test prompt for domain: customer"
        )
        self.mock_logger.debug.assert_not_called()

    def test_log_prompt_generation_with_context(self):
        """Test prompt generation logging with context"""
        context = {
            "items": [1, 2, 3],
            "config": {"key": "value"},
            "name": "test"
        }
        
        PromptLogger.log_prompt_generation("test", "customer", context)
        
        assert self.mock_logger.info.call_count == 1
        assert self.mock_logger.debug.call_count == 4
        
        # Verify debug messages for each context item
        debug_calls = [
            call("Context for test prompt:"),
            call("- items: list with 3 items"),
            call("- config: dict with 1 items"),
            call("- name: test")
        ]
        self.mock_logger.debug.assert_has_calls(debug_calls, any_order=True)

    def test_log_prompt_content_with_markers(self):
        """Test logging prompt content with section markers"""
        test_prompt = "This is a test prompt\nWith multiple lines\n**Section 1**\nContent"
        
        PromptLogger.log_prompt_content("test", test_prompt)
        
        expected_calls = [
            call("=== TEST PROMPT START ==="),
            call(test_prompt),
            call("=== TEST PROMPT END ==="),
            call("Prompt statistics:"),
            call("- Total length: 63 characters"),
            call("- Section count: 1")
        ]
        self.mock_logger.info.assert_has_calls(expected_calls, any_order=False)

    def test_log_prompt_content_without_markers(self):
        """Test logging prompt content without section markers"""
        test_prompt = "A" * 250  # Create a long prompt
        
        PromptLogger.log_prompt_content("test", test_prompt, section_markers=False)
        
        expected_calls = [
            call(f"test prompt content: {test_prompt[:200]}..."),
            call("Prompt statistics:"),
            call("- Total length: 250 characters"),
            call("- Section count: 0")
        ]
        self.mock_logger.info.assert_has_calls(expected_calls)

    def test_log_prompt_components(self):
        """Test logging prompt components"""
        components = {
            "header": "Short header",
            "body": "A" * 150,  # Long body
            "footer": "Short footer"
        }
        
        PromptLogger.log_prompt_components("test", components)
        
        self.mock_logger.info.assert_called_once_with("Components for test prompt:")
        
        debug_calls = [
            call("- header: 12 characters"),
            call("- body: 150 characters"),
            call(f"  Preview: {'A' * 100}..."),
            call("- footer: 12 characters")
        ]
        self.mock_logger.debug.assert_has_calls(debug_calls)

    def test_log_prompt_combination(self):
        """Test logging prompt combination"""
        PromptLogger.log_prompt_combination("base", "enhancement")
        
        self.mock_logger.info.assert_called_once_with(
            "Combining base prompt with enhancement"
        )

    def test_log_prompt_error(self):
        """Test logging prompt errors"""
        test_error = ValueError("Test error message")
        
        PromptLogger.log_prompt_error("test", test_error)
        
        self.mock_logger.error.assert_called_once_with(
            "Error in test prompt: Test error message",
            exc_info=True
        )

    def test_log_prompt_generation_empty_context(self):
        """Test prompt generation logging with empty context"""
        PromptLogger.log_prompt_generation("test", "customer", {})
        
        self.mock_logger.info.assert_called_once_with(
            "Generating test prompt for domain: customer"
        )
        self.mock_logger.debug.assert_not_called()

    def test_log_prompt_content_empty_prompt(self):
        """Test logging empty prompt content"""
        PromptLogger.log_prompt_content("test", "")
        
        expected_calls = [
            call("=== TEST PROMPT START ==="),
            call(""),
            call("=== TEST PROMPT END ==="),
            call("Prompt statistics:"),
            call("- Total length: 0 characters"),
            call("- Section count: 0")
        ]
        self.mock_logger.info.assert_has_calls(expected_calls)

    def test_log_prompt_components_empty_dict(self):
        """Test logging empty components dictionary"""
        PromptLogger.log_prompt_components("test", {})
        
        self.mock_logger.info.assert_called_once_with("Components for test prompt:")
        self.mock_logger.debug.assert_not_called()

    def test_log_prompt_error_without_message(self):
        """Test logging error without message"""
        test_error = Exception()
        
        PromptLogger.log_prompt_error("test", test_error)
        
        self.mock_logger.error.assert_called_once_with(
            "Error in test prompt: ", 
            exc_info=True
        )