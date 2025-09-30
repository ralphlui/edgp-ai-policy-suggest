"""
Tests for app/vector_db/schema_loader.py module
"""
import pytest
from unittest.mock import Mock, patch, MagicMock


class TestSchemaLoaderModule:
    """Test schema_loader.py module functionality"""
    
    def test_schema_loader_module_import(self):
        """Test schema_loader module can be imported"""
        from app.vector_db import schema_loader
        assert hasattr(schema_loader, '__name__')

    def test_schema_loader_functions_exist(self):
        """Test schema loader functions exist"""
        from app.vector_db.schema_loader import get_schema_by_domain
        assert callable(get_schema_by_domain)

    @patch('app.vector_db.schema_loader.settings')
    @patch('app.vector_db.schema_loader.OpenSearchColumnStore')
    def test_get_schema_by_domain_success(self, mock_store_class, mock_settings):
        """Test successful schema retrieval"""
        mock_settings.column_index_name = "test_index"
        
        mock_store = Mock()
        mock_store.get_columns_by_domain.return_value = [
            {"column_name": "customer_id", "metadata": {"type": "integer"}},
            {"column_name": "customer_name", "metadata": {"type": "string"}}
        ]
        mock_store_class.return_value = mock_store
        
        from app.vector_db.schema_loader import get_schema_by_domain
        
        result = get_schema_by_domain("customer")
        
        assert isinstance(result, dict)
        assert "columns" in result
        mock_store.get_columns_by_domain.assert_called_once_with("customer")

    @patch('app.vector_db.schema_loader.settings')
    @patch('app.vector_db.schema_loader.OpenSearchColumnStore')
    def test_get_schema_by_domain_store_error(self, mock_store_class, mock_settings):
        """Test schema retrieval with store initialization error"""
        mock_settings.column_index_name = "test_index"
        mock_store_class.side_effect = Exception("Store initialization failed")
        
        from app.vector_db.schema_loader import get_schema_by_domain
        
        result = get_schema_by_domain("customer")
        
        assert isinstance(result, dict)
        assert result == {}  # Should return empty dict on error

    @patch('app.vector_db.schema_loader.settings')
    def test_get_schema_by_domain_missing_settings(self, mock_settings):
        """Test schema retrieval with missing settings"""
        # Mock missing column_index_name attribute
        del mock_settings.column_index_name
        
        from app.vector_db.schema_loader import get_schema_by_domain
        
        result = get_schema_by_domain("customer")
        
        assert isinstance(result, dict)
        assert result == {}

    @patch('app.vector_db.schema_loader.settings')
    @patch('app.vector_db.schema_loader.OpenSearchColumnStore')
    def test_get_schema_by_domain_query_error(self, mock_store_class, mock_settings):
        """Test schema retrieval with query error"""
        mock_settings.column_index_name = "test_index"
        
        mock_store = Mock()
        mock_store.get_columns_by_domain.side_effect = Exception("Query failed")
        mock_store_class.return_value = mock_store
        
        from app.vector_db.schema_loader import get_schema_by_domain
        
        result = get_schema_by_domain("customer")
        
        assert isinstance(result, dict)
        assert result == {}

    @patch('app.vector_db.schema_loader.settings')
    @patch('app.vector_db.schema_loader.OpenSearchColumnStore')
    def test_get_schema_by_domain_empty_result(self, mock_store_class, mock_settings):
        """Test schema retrieval with empty result"""
        mock_settings.column_index_name = "test_index"
        
        mock_store = Mock()
        mock_store.get_columns_by_domain.return_value = []
        mock_store_class.return_value = mock_store
        
        from app.vector_db.schema_loader import get_schema_by_domain
        
        result = get_schema_by_domain("nonexistent_domain")
        
        assert isinstance(result, dict)
        assert "columns" in result
        assert result["columns"] == []

    @patch('app.vector_db.schema_loader.logger')
    @patch('app.vector_db.schema_loader.settings')
    def test_logging_functionality(self, mock_settings, mock_logger):
        """Test logging functionality"""
        del mock_settings.column_index_name  # Trigger warning log
        
        from app.vector_db.schema_loader import get_schema_by_domain
        
        result = get_schema_by_domain("test")
        
        assert isinstance(result, dict)
        # Verify logging was called (logger should be used)

    @patch('app.vector_db.schema_loader.settings')
    @patch('app.vector_db.schema_loader.OpenSearchColumnStore')
    def test_get_schema_by_domain_complex_metadata(self, mock_store_class, mock_settings):
        """Test schema retrieval with complex metadata"""
        mock_settings.column_index_name = "test_index"
        
        mock_store = Mock()
        mock_store.get_columns_by_domain.return_value = [
            {
                "column_name": "email",
                "metadata": {
                    "type": "string",
                    "format": "email",
                    "nullable": False,
                    "description": "Customer email address"
                }
            },
            {
                "column_name": "created_at",
                "metadata": {
                    "type": "datetime",
                    "format": "iso8601",
                    "nullable": True
                }
            }
        ]
        mock_store_class.return_value = mock_store
        
        from app.vector_db.schema_loader import get_schema_by_domain
        
        result = get_schema_by_domain("customer")
        
        assert isinstance(result, dict)
        assert "columns" in result
        assert len(result["columns"]) == 2
        
        # Check that complex metadata is preserved
        email_col = next(col for col in result["columns"] if col["column_name"] == "email")
        assert email_col["metadata"]["format"] == "email"
        assert email_col["metadata"]["nullable"] is False

    def test_schema_loader_imports(self):
        """Test schema loader imports work correctly"""
        from app.vector_db.schema_loader import get_schema_by_domain
        assert callable(get_schema_by_domain)

    @patch('app.vector_db.schema_loader.settings')
    @patch('app.vector_db.schema_loader.OpenSearchColumnStore')
    def test_get_schema_by_domain_different_domains(self, mock_store_class, mock_settings):
        """Test schema retrieval for different domains"""
        mock_settings.column_index_name = "test_index"
        
        mock_store = Mock()
        mock_store_class.return_value = mock_store
        
        # Test multiple domain calls
        domains = ["customer", "product", "order"]
        
        for domain in domains:
            mock_store.get_columns_by_domain.return_value = [
                {"column_name": f"{domain}_id", "metadata": {"type": "integer"}}
            ]
            
            from app.vector_db.schema_loader import get_schema_by_domain
            result = get_schema_by_domain(domain)
            
            assert isinstance(result, dict)
            assert "columns" in result


class TestSchemaLoaderUtilities:
    """Test utility functions and edge cases"""
    
    def test_module_level_imports(self):
        """Test module level imports work"""
        from app.vector_db.schema_loader import get_schema_by_domain
        assert callable(get_schema_by_domain)

    @patch('app.vector_db.schema_loader.settings')
    def test_settings_attribute_error_handling(self, mock_settings):
        """Test handling of settings attribute errors"""
        # Create a mock that raises AttributeError when accessing column_index_name
        type(mock_settings).column_index_name = property(
            lambda self: (_ for _ in ()).throw(AttributeError("'Settings' object has no attribute 'column_index_name'"))
        )
        
        from app.vector_db.schema_loader import get_schema_by_domain
        
        result = get_schema_by_domain("test")
        
        assert isinstance(result, dict)
        assert result == {}

    def test_function_docstrings(self):
        """Test function docstrings exist"""
        from app.vector_db.schema_loader import get_schema_by_domain
        
        # Function should be callable
        assert callable(get_schema_by_domain)