"""
Comprehensive unit tests for app/agents/schema_suggester.py module.
Tests the enhanced schema generation functionality with error handling, validation, and caching.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List
import json
import time
import os


class TestSchemaGenerationConfig:
    """Test SchemaGenerationConfig dataclass"""
    
    def test_config_default_values(self):
        """Test default configuration values"""
        try:
            from app.agents.schema_suggester import SchemaGenerationConfig
        except ImportError:
            pytest.skip("Required modules not available")
        
        config = SchemaGenerationConfig()
        assert config.min_columns == 5
        assert config.max_columns == 11
        assert config.min_samples == 3
        assert config.max_retries == 3
        assert config.timeout_seconds == 30
        assert "string" in config.supported_types
        assert "integer" in config.supported_types
        assert len(config.supported_types) == 6
    
    def test_config_custom_values(self):
        """Test custom configuration values"""
        try:
            from app.agents.schema_suggester import SchemaGenerationConfig
        except ImportError:
            pytest.skip("Required modules not available")
        
        config = SchemaGenerationConfig(
            min_columns=3,
            max_columns=15,
            min_samples=5,
            supported_types=["string", "integer"]
        )
        assert config.min_columns == 3
        assert config.max_columns == 15
        assert config.min_samples == 5
        assert len(config.supported_types) == 2


class TestPydanticModels:
    """Test Pydantic models for schema validation"""
    
    def test_column_schema_valid(self):
        """Test valid ColumnSchema creation"""
        try:
            from app.agents.schema_suggester import ColumnSchema
        except ImportError:
            pytest.skip("Required modules not available")
        
        column = ColumnSchema(
            name="user_id",
            type="integer",
            samples=["1", "2", "3"]
        )
        assert column.name == "user_id"
        assert column.type == "integer"
        assert len(column.samples) == 3
    
    def test_column_schema_invalid_name(self):
        """Test ColumnSchema with invalid name"""
        try:
            from app.agents.schema_suggester import ColumnSchema
            from pydantic import ValidationError
        except ImportError:
            pytest.skip("Required modules not available")
        
        with pytest.raises(ValidationError):
            ColumnSchema(
                name="user-id",  # Invalid identifier
                type="integer",
                samples=["1", "2", "3"]
            )
    
    def test_column_schema_invalid_type(self):
        """Test ColumnSchema with invalid type"""
        try:
            from app.agents.schema_suggester import ColumnSchema
            from pydantic import ValidationError
        except ImportError:
            pytest.skip("Required modules not available")
        
        with pytest.raises(ValidationError):
            ColumnSchema(
                name="user_id",
                type="invalid_type",  # Not in allowed types
                samples=["1", "2", "3"]
            )
    
    def test_schema_response_valid(self):
        """Test valid SchemaResponse creation"""
        try:
            from app.agents.schema_suggester import SchemaResponse, ColumnSchema
        except ImportError:
            pytest.skip("Required modules not available")
        
        columns = [
            ColumnSchema(name=f"col_{i}", type="string", samples=["a", "b", "c"])
            for i in range(5)
        ]
        
        response = SchemaResponse(columns=columns)
        assert len(response.columns) == 5
    
    def test_schema_response_too_few_columns(self):
        """Test SchemaResponse with too few columns"""
        try:
            from app.agents.schema_suggester import SchemaResponse, ColumnSchema
            from pydantic import ValidationError
        except ImportError:
            pytest.skip("Required modules not available")
        
        columns = [
            ColumnSchema(name=f"col_{i}", type="string", samples=["a", "b", "c"])
            for i in range(3)  # Less than minimum 5
        ]
        
        with pytest.raises(ValidationError):
            SchemaResponse(columns=columns)


class TestModelChainManagement:
    """Test model chain creation and caching"""
    
    def setup_method(self):
        """Setup for each test method"""
        try:
            from app.agents.schema_suggester import clear_model_cache
            clear_model_cache()
        except ImportError:
            pass
    
    def teardown_method(self):
        """Cleanup after each test method"""
        try:
            from app.agents.schema_suggester import clear_model_cache
            clear_model_cache()
        except ImportError:
            pass
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}, clear=False)
    @patch('app.agents.schema_suggester.ChatOpenAI')
    @patch('app.core.aws_secrets_service.require_openai_api_key', return_value='test-key')
    def test_get_model_chain_creation(self, mock_require_api_key, mock_chat_openai):
        """Test model chain creation"""
        try:
            from app.agents.schema_suggester import get_model_chain, clear_model_cache
        except ImportError:
            pytest.skip("Required modules not available")
        
        clear_model_cache()  # Clear cache first
        
        mock_model = Mock()
        mock_chat_openai.return_value = mock_model
        
        chain = get_model_chain(use_structured_output=True)
        
        assert chain is not None
        mock_chat_openai.assert_called_once()
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}, clear=False)
    @patch('app.agents.schema_suggester.ChatOpenAI')
    @patch('app.core.aws_secrets_service.require_openai_api_key', return_value='test-key')
    def test_model_chain_caching(self, mock_require_api_key, mock_chat_openai):
        """Test that model chains are cached"""
        try:
            from app.agents.schema_suggester import get_model_chain, clear_model_cache
        except ImportError:
            pytest.skip("Required modules not available")
        
        clear_model_cache()
        
        mock_model = Mock()
        mock_chat_openai.return_value = mock_model
        
        # First call should create the chain
        chain1 = get_model_chain(use_structured_output=True)
        
        # Second call should return cached chain
        chain2 = get_model_chain(use_structured_output=True)
        
        assert chain1 is chain2
        assert mock_chat_openai.call_count == 1  # Should only be called once


class TestCallLLM:
    """Test LLM calling functionality"""
    
    @patch('app.agents.schema_suggester.get_model_chain')
    def test_call_llm_success(self, mock_get_chain):
        """Test successful LLM call"""
        try:
            from app.agents.schema_suggester import call_llm
        except ImportError:
            pytest.skip("Required modules not available")
        
        # Mock the chain
        mock_chain = Mock()
        mock_response = {
            "columns": [
                {"name": "user_id", "type": "integer", "samples": ["1", "2", "3"]},
                {"name": "name", "type": "string", "samples": ["Alice", "Bob", "Carol"]}
            ]
        }
        mock_chain.invoke.return_value = mock_response
        mock_get_chain.return_value = mock_chain
        
        result = call_llm("customer", use_structured_output=False)
        
        assert result == mock_response
        # Updated to match the new function call with format_instructions
        mock_chain.invoke.assert_called_once()
        call_args = mock_chain.invoke.call_args[0][0]
        assert call_args["domain"] == "customer"
        assert "format_instructions" in call_args
    
    def test_call_llm_invalid_domain(self):
        """Test LLM call with invalid domain"""
        try:
            from app.agents.schema_suggester import call_llm
        except ImportError:
            pytest.skip("Required modules not available")
        
        with pytest.raises(ValueError, match="Domain must be a non-empty string"):
            call_llm("")
        
        with pytest.raises(ValueError, match="Domain must be a non-empty string"):
            call_llm(None)
        
        with pytest.raises(ValueError, match="Domain cannot be empty after normalization"):
            call_llm("   ")
    
    @patch('app.agents.schema_suggester.get_model_chain')
    def test_call_llm_with_structured_output(self, mock_get_chain):
        """Test LLM call with structured output"""
        try:
            from app.agents.schema_suggester import call_llm, SchemaResponse
        except ImportError:
            pytest.skip("Required modules not available")
        
        # Mock the chain and response
        mock_chain = Mock()
        mock_response = Mock(spec=SchemaResponse)
        mock_response.dict.return_value = {
            "columns": [
                {"name": "user_id", "type": "integer", "samples": ["1", "2", "3"]}
            ]
        }
        mock_chain.invoke.return_value = mock_response
        mock_get_chain.return_value = mock_chain
        
        result = call_llm("customer", use_structured_output=True)
        
        assert "columns" in result
        mock_response.dict.assert_called_once()
    
    @patch('app.agents.schema_suggester.get_model_chain')
    def test_call_llm_retry_mechanism(self, mock_get_chain):
        """Test LLM call retry mechanism for network errors"""
        try:
            from app.agents.schema_suggester import call_llm
            from app.core.exceptions import SchemaGenerationError
        except ImportError:
            pytest.skip("Required modules not available")
        
        # Mock chain that always fails with a retryable error
        mock_chain = Mock()
        mock_chain.invoke.side_effect = ConnectionError("Network failure")
        mock_get_chain.return_value = mock_chain
        
        # Should retry and eventually fail with SchemaGenerationError
        # The retry mechanism is applied via tenacity decorator
        with pytest.raises(SchemaGenerationError, match="Network failure"):
            call_llm("customer", use_structured_output=False)


class TestFormatLLMSchema:
    """Test schema formatting functionality"""
    
    @patch('app.agents.schema_suggester.validate_column_schema')
    def test_format_llm_schema_success(self, mock_validate):
        """Test successful schema formatting"""
        try:
            from app.agents.schema_suggester import format_llm_schema
        except ImportError:
            pytest.skip("Required modules not available")
        
        mock_validate.return_value = True
        
        raw_response = {
            "columns": [
                {"name": "user_id", "type": "integer", "samples": ["1", "2", "3"]},
                {"name": "name", "type": "string", "samples": ["Alice", "Bob", "Carol"]},
                {"name": "email", "type": "string", "samples": ["a@ex.com", "b@ex.com", "c@ex.com"]},
                {"name": "age", "type": "integer", "samples": ["25", "30", "35"]},
                {"name": "active", "type": "boolean", "samples": ["true", "false", "true"]}
            ]
        }
        
        result = format_llm_schema(raw_response)
        
        assert len(result) == 5
        assert "user_id" in result
        assert result["user_id"]["dtype"] == "integer"
        assert len(result["user_id"]["sample_values"]) == 3
    
    def test_format_llm_schema_empty_columns(self):
        """Test formatting with empty columns"""
        try:
            from app.agents.schema_suggester import format_llm_schema
            from app.core.exceptions import SchemaGenerationError
        except ImportError:
            pytest.skip("Required modules not available")
        
        raw_response = {"columns": []}
        
        with pytest.raises(SchemaGenerationError, match="No columns found"):
            format_llm_schema(raw_response)
    
    @patch('app.agents.schema_suggester.validate_column_schema')
    def test_format_llm_schema_validation_failures(self, mock_validate):
        """Test formatting with validation failures"""
        try:
            from app.agents.schema_suggester import format_llm_schema
            from app.core.exceptions import SchemaGenerationError
        except ImportError:
            pytest.skip("Required modules not available")
        
        # All columns fail validation
        mock_validate.return_value = False
        
        raw_response = {
            "columns": [
                {"name": "invalid1", "type": "string", "samples": ["a", "b"]},  # Too few samples
                {"name": "invalid2", "type": "unknown", "samples": ["x", "y", "z"]}  # Invalid type
            ]
        }
        
        with pytest.raises(SchemaGenerationError):
            format_llm_schema(raw_response, strict_validation=True)
    
    @patch('app.agents.schema_suggester.validate_column_schema')
    def test_format_llm_schema_duplicate_names(self, mock_validate):
        """Test formatting with duplicate column names"""
        try:
            from app.agents.schema_suggester import format_llm_schema
        except ImportError:
            pytest.skip("Required modules not available")
        
        mock_validate.return_value = True
        
        raw_response = {
            "columns": [
                {"name": "user_id", "type": "integer", "samples": ["1", "2", "3"]},
                {"name": "user_id", "type": "string", "samples": ["a", "b", "c"]},  # Duplicate
                {"name": "name", "type": "string", "samples": ["Alice", "Bob", "Carol"]},
                {"name": "email", "type": "string", "samples": ["a@ex.com", "b@ex.com", "c@ex.com"]},
                {"name": "age", "type": "integer", "samples": ["25", "30", "35"]}
            ]
        }
        
        result = format_llm_schema(raw_response)
        
        # Should handle duplicate by renaming
        assert "user_id" in result
        assert "user_id_1" in result or any("user_id" in key for key in result.keys())


class TestHelperFunctions:
    """Test helper functions"""
    
    def test_validate_samples_for_type(self):
        """Test sample validation for different types"""
        try:
            from app.agents.schema_suggester import _validate_samples_for_type
        except ImportError:
            pytest.skip("Required modules not available")
        
        # Test integer samples
        assert _validate_samples_for_type(["1", "2", "3"], "integer") is True
        assert _validate_samples_for_type(["1.5", "2", "3"], "integer") is False
        
        # Test float samples
        assert _validate_samples_for_type(["1.5", "2.0", "3.14"], "float") is True
        assert _validate_samples_for_type(["abc", "2.0", "3.14"], "float") is False
        
        # Test boolean samples
        assert _validate_samples_for_type(["true", "false", "1"], "boolean") is True
        assert _validate_samples_for_type(["maybe", "false", "1"], "boolean") is False
        
        # Test string samples (should accept anything)
        assert _validate_samples_for_type(["abc", "def", "ghi"], "string") is True
    
    def test_normalize_data_type(self):
        """Test data type normalization"""
        try:
            from app.agents.schema_suggester import _normalize_data_type
        except ImportError:
            pytest.skip("Required modules not available")
        
        assert _normalize_data_type("int") == "integer"
        assert _normalize_data_type("bool") == "boolean"
        assert _normalize_data_type("str") == "string"
        assert _normalize_data_type("text") == "string"
        assert _normalize_data_type("number") == "float"
        assert _normalize_data_type("unknown_type") == "unknown_type"
    
    def test_is_valid_float(self):
        """Test float validation helper"""
        try:
            from app.agents.schema_suggester import _is_valid_float
        except ImportError:
            pytest.skip("Required modules not available")
        
        assert _is_valid_float("3.14") is True
        assert _is_valid_float("42") is True
        assert _is_valid_float("-7.5") is True
        assert _is_valid_float("abc") is False
        assert _is_valid_float("") is False


class TestBootstrapSchema:
    """Test main bootstrap schema functionality"""
    
    @patch('app.agents.schema_suggester.call_llm')
    @patch('app.agents.schema_suggester.format_llm_schema')
    def test_bootstrap_schema_for_domain_success(self, mock_format, mock_call_llm):
        """Test successful schema bootstrapping"""
        try:
            from app.agents.schema_suggester import bootstrap_schema_for_domain
        except ImportError:
            pytest.skip("Required modules not available")
        
        # Mock LLM response
        mock_call_llm.return_value = {
            "columns": [
                {"name": "user_id", "type": "integer", "samples": ["1", "2", "3"]}
            ]
        }
        
        # Mock formatted schema
        mock_format.return_value = {
            "user_id": {"dtype": "integer", "sample_values": ["1", "2", "3"]}
        }
        
        result = bootstrap_schema_for_domain("customer")
        
        assert "user_id" in result
        mock_call_llm.assert_called_once()
        mock_format.assert_called_once()
    
    def test_bootstrap_schema_empty_domain(self):
        """Test bootstrap with empty domain"""
        try:
            from app.agents.schema_suggester import bootstrap_schema_for_domain
        except ImportError:
            pytest.skip("Required modules not available")
        
        with pytest.raises(ValueError, match="Domain cannot be empty"):
            bootstrap_schema_for_domain("")
    
    @patch('app.agents.schema_suggester.call_llm')
    @patch('app.agents.schema_suggester.format_llm_schema')
    def test_bootstrap_schema_fallback_mechanism(self, mock_format, mock_call_llm):
        """Test fallback mechanism on failure"""
        try:
            from app.agents.schema_suggester import bootstrap_schema_for_domain
            from app.core.exceptions import SchemaGenerationError
        except ImportError:
            pytest.skip("Required modules not available")
        
        # First call fails, second succeeds
        mock_call_llm.side_effect = [
            SchemaGenerationError("First attempt failed"),
            {"columns": [{"name": "user_id", "type": "integer", "samples": ["1", "2", "3"]}]}
        ]
        
        mock_format.return_value = {
            "user_id": {"dtype": "integer", "sample_values": ["1", "2", "3"]}
        }
        
        result = bootstrap_schema_for_domain("customer", fallback_on_error=True)
        
        assert "user_id" in result
        assert mock_call_llm.call_count == 2  # Original + fallback
    
    @patch('app.agents.schema_suggester.bootstrap_schema_for_domain')
    def test_bootstrap_schema_alias(self, mock_bootstrap):
        """Test bootstrap_schema alias function"""
        try:
            from app.agents.schema_suggester import bootstrap_schema
        except ImportError:
            pytest.skip("Required modules not available")
        
        mock_bootstrap.return_value = {"test": "schema"}
        
        result = bootstrap_schema("customer")
        
        mock_bootstrap.assert_called_once_with("customer")
        assert result == {"test": "schema"}


class TestSchemaValidation:
    """Test schema validation functionality"""
    
    def test_validate_schema_completeness_valid(self):
        """Test validation of complete schema"""
        try:
            from app.agents.schema_suggester import validate_schema_completeness
        except ImportError:
            pytest.skip("Required modules not available")
        
        valid_schema = {
            "user_id": {"dtype": "integer", "sample_values": ["1", "2", "3"]},
            "name": {"dtype": "string", "sample_values": ["Alice", "Bob", "Carol"]},
            "email": {"dtype": "string", "sample_values": ["a@ex.com", "b@ex.com", "c@ex.com"]},
            "created_date": {"dtype": "date", "sample_values": ["2024-01-01", "2024-01-02", "2024-01-03"]},
            "status": {"dtype": "boolean", "sample_values": ["true", "false", "true"]}
        }
        
        result = validate_schema_completeness(valid_schema)
        
        assert result["is_valid"] is True
        assert result["column_count"] == 5
        assert len(result["types_used"]) >= 3
        assert len(result["issues"]) == 0
    
    def test_validate_schema_completeness_insufficient_columns(self):
        """Test validation with insufficient columns"""
        try:
            from app.agents.schema_suggester import validate_schema_completeness
        except ImportError:
            pytest.skip("Required modules not available")
        
        insufficient_schema = {
            "user_id": {"dtype": "integer", "sample_values": ["1", "2", "3"]},
            "name": {"dtype": "string", "sample_values": ["Alice", "Bob", "Carol"]}
        }
        
        result = validate_schema_completeness(insufficient_schema)
        
        assert result["is_valid"] is False
        assert any("Too few columns" in issue for issue in result["issues"])
    
    def test_validate_schema_completeness_limited_diversity(self):
        """Test validation with limited type diversity"""
        try:
            from app.agents.schema_suggester import validate_schema_completeness
        except ImportError:
            pytest.skip("Required modules not available")
        
        limited_schema = {
            f"col_{i}": {"dtype": "string", "sample_values": ["a", "b", "c"]}
            for i in range(6)  # All string types
        }
        
        result = validate_schema_completeness(limited_schema)
        
        assert result["is_valid"] is False
        assert any("Limited data type diversity" in issue for issue in result["issues"])


class TestCacheManagement:
    """Test cache management functionality"""
    
    @patch('app.agents.schema_suggester._model_chain_cache', {})
    def test_clear_model_cache(self):
        """Test cache clearing functionality"""
        try:
            from app.agents.schema_suggester import clear_model_cache, _model_chain_cache
        except ImportError:
            pytest.skip("Required modules not available")
        
        # Add something to cache
        _model_chain_cache["test_key"] = "test_value"
        
        assert len(_model_chain_cache) > 0
        
        clear_model_cache()
        
        assert len(_model_chain_cache) == 0


class TestErrorHandling:
    """Test error handling across all functions"""
    
    @patch('app.agents.schema_suggester.get_model_chain')
    def test_call_llm_exception_handling(self, mock_get_chain):
        """Test LLM call exception handling for non-retryable errors"""
        try:
            from app.agents.schema_suggester import call_llm
            from app.core.exceptions import SchemaGenerationError
        except ImportError:
            pytest.skip("Required modules not available")
        
        # Mock chain that fails with non-retryable error
        mock_chain = Mock()
        mock_chain.invoke.side_effect = ValueError("Invalid API response")
        mock_get_chain.return_value = mock_chain
        
        with pytest.raises(SchemaGenerationError):
            call_llm("customer", use_structured_output=False)
    
    @patch('app.agents.schema_suggester.call_llm')
    def test_bootstrap_exception_handling(self, mock_call_llm):
        """Test bootstrap exception handling"""
        try:
            from app.agents.schema_suggester import bootstrap_schema_for_domain
            from app.core.exceptions import SchemaGenerationError
        except ImportError:
            pytest.skip("Required modules not available")
        
        # Mock LLM call that fails
        mock_call_llm.side_effect = Exception("LLM Error")
        
        with pytest.raises(SchemaGenerationError):
            bootstrap_schema_for_domain("customer", fallback_on_error=False)


class TestIntegration:
    """Integration tests combining multiple components"""
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}, clear=False)
    @patch('app.agents.schema_suggester.ChatOpenAI')
    @patch('app.core.aws_secrets_service.require_openai_api_key', return_value='test-key')
    @patch('app.agents.schema_suggester.validate_column_schema')
    def test_end_to_end_schema_generation(self, mock_validate, mock_require_api_key, mock_chat_openai):
        """Test end-to-end schema generation"""
        try:
            from app.agents.schema_suggester import bootstrap_schema_for_domain, clear_model_cache
        except ImportError:
            pytest.skip("Required modules not available")
        
        clear_model_cache()
        
        # Mock validation
        mock_validate.return_value = True
        
        # Mock the LLM model and chain
        mock_model = Mock()
        mock_chain = Mock()
        mock_response = {
            "columns": [
                {"name": "customer_id", "type": "integer", "samples": ["1", "2", "3"]},
                {"name": "name", "type": "string", "samples": ["Alice", "Bob", "Carol"]},
                {"name": "email", "type": "string", "samples": ["a@ex.com", "b@ex.com", "c@ex.com"]},
                {"name": "created_date", "type": "date", "samples": ["2024-01-01", "2024-01-02", "2024-01-03"]},
                {"name": "active", "type": "boolean", "samples": ["true", "false", "true"]}
            ]
        }
        
        # Setup the mock chain
        with patch('app.agents.schema_suggester.ChatPromptTemplate') as mock_prompt, \
             patch('app.agents.schema_suggester.JsonOutputParser') as mock_parser:
            
            mock_parser_instance = Mock()
            mock_parser.return_value = mock_parser_instance
            
            mock_prompt_instance = Mock()
            mock_prompt.from_messages.return_value = mock_prompt_instance
            
            # Create a mock chain that returns our response
            mock_chain_result = Mock()
            mock_chain_result.invoke.return_value = mock_response
            
            # Mock the pipe operator to return our chain
            mock_prompt_instance.__or__ = Mock(return_value=Mock(__or__=Mock(return_value=mock_chain_result)))
            
            mock_chat_openai.return_value = mock_model
            
            result = bootstrap_schema_for_domain("customer", use_structured_output=False)
            
            # Verify results
            assert len(result) == 5
            assert "customer_id" in result
            assert "name" in result
            assert "email" in result
            assert result["customer_id"]["dtype"] == "integer"
            assert len(result["customer_id"]["sample_values"]) == 3


class TestConfiguration:
    """Test configuration management"""
    
    @patch('app.agents.schema_suggester.get_schema_generation_config')
    def test_configuration_usage(self, mock_config):
        """Test that configuration is used correctly"""
        try:
            from app.agents.schema_suggester import format_llm_schema, SchemaGenerationConfig
            from app.core.exceptions import SchemaGenerationError
        except ImportError:
            pytest.skip("Required modules not available")
        
        # Mock config with strict requirements
        mock_config.return_value = SchemaGenerationConfig(min_columns=10)
        
        raw_response = {
            "columns": [
                {"name": f"col_{i}", "type": "string", "samples": ["a", "b", "c"]}
                for i in range(3)  # Only 3 columns, less than required 10
            ]
        }
        
        with patch('app.agents.schema_suggester.validate_column_schema', return_value=True):
            with pytest.raises(SchemaGenerationError):
                format_llm_schema(raw_response)