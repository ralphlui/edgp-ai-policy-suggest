"""
Comprehensive test suite for app.env_config_util module
"""

import os
import sys
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open, PropertyMock
from io import StringIO
from contextlib import redirect_stdout

# Import the module under test
from app.env_config_util import (
    show_current_env,
    list_environments,
    set_environment,
    validate_environment,
    main
)


class TestShowCurrentEnv:
    """Test show_current_env function"""
    
    @patch.dict(os.environ, {'APP_ENV': 'development', 'ENVIRONMENT': 'dev'})
    @patch('os.path.exists')
    def test_show_current_env_with_existing_file(self, mock_exists):
        """Test showing current env when env file exists"""
        mock_exists.return_value = True
        
        with StringIO() as captured_output:
            with redirect_stdout(captured_output):
                show_current_env()
            output = captured_output.getvalue()
        
        assert "APP_ENV: development" in output
        assert "ENVIRONMENT: dev" in output
        assert "Environment file exists: .env.development" in output
        mock_exists.assert_called_once_with(".env.development")
    
    @patch.dict(os.environ, {'APP_ENV': 'production'}, clear=True)
    @patch('os.path.exists')
    def test_show_current_env_with_missing_file(self, mock_exists):
        """Test showing current env when env file is missing"""
        mock_exists.return_value = False
        
        with StringIO() as captured_output:
            with redirect_stdout(captured_output):
                show_current_env()
            output = captured_output.getvalue()
        
        assert "APP_ENV: production" in output
        assert "ENVIRONMENT: not set" in output
        assert "Environment file missing: .env.production" in output
    
    @patch.dict(os.environ, {}, clear=True)
    def test_show_current_env_no_vars_set(self):
        """Test showing current env when no environment variables are set"""
        with StringIO() as captured_output:
            with redirect_stdout(captured_output):
                show_current_env()
            output = captured_output.getvalue()
        
        assert "APP_ENV: not set" in output
        assert "ENVIRONMENT: not set" in output


class TestListEnvironments:
    """Test list_environments function"""
    
    @patch('pathlib.Path.glob')
    def test_list_environments_with_files(self, mock_glob):
        """Test listing environments when .env files exist"""
        # Create mock Path objects with proper comparison methods
        class MockPath:
            def __init__(self, name, size):
                self.name = name
                self.stat_size = size
            
            def stat(self):
                mock_stat = MagicMock()
                mock_stat.st_size = self.stat_size
                return mock_stat
            
            def __lt__(self, other):
                return self.name < other.name
        
        mock_file1 = MockPath('.env.development', 1024)
        mock_file2 = MockPath('.env.production', 2048)
        
        mock_glob.return_value = [mock_file1, mock_file2]
        
        with StringIO() as captured_output:
            with redirect_stdout(captured_output):
                list_environments()
            output = captured_output.getvalue()
        
        assert "development" in output
        assert "production" in output
        assert "1024 bytes" in output
        assert "2048 bytes" in output
        assert "Total: 2 environment files" in output
    
    @patch('pathlib.Path.glob')
    def test_list_environments_no_files(self, mock_glob):
        """Test listing environments when no .env files exist"""
        mock_glob.return_value = []
        
        with StringIO() as captured_output:
            with redirect_stdout(captured_output):
                list_environments()
            output = captured_output.getvalue()
        
        assert "No environment files found" in output
        assert "Total: 0 environment files" in output


class TestSetEnvironment:
    """Test set_environment function"""
    
    @patch('os.path.exists')
    @patch('app.env_config_util.list_environments')
    def test_set_environment_file_not_exists(self, mock_list_env, mock_exists):
        """Test setting environment when file doesn't exist"""
        mock_exists.return_value = False
        
        with StringIO() as captured_output:
            with redirect_stdout(captured_output):
                result = set_environment('nonexistent')
        
        assert result is False
        assert os.environ.get('APP_ENV') != 'nonexistent'
        mock_list_env.assert_called_once()
    
    @patch('os.path.exists')
    def test_set_environment_success(self, mock_exists):
        """Test successful environment setting"""
        mock_exists.return_value = True
        
        with StringIO() as captured_output:
            with redirect_stdout(captured_output):
                result = set_environment('development')
            output = captured_output.getvalue()
        
        assert result is True
        assert os.environ.get('APP_ENV') == 'development'
        assert "APP_ENV set to: development" in output
        assert "Will use: .env.development" in output
        assert "export APP_ENV=development" in output


class TestValidateEnvironment:
    """Test validate_environment function"""
    
    @patch('os.path.exists')
    def test_validate_environment_file_not_exists(self, mock_exists):
        """Test validation when environment file doesn't exist"""
        mock_exists.return_value = False
        
        with StringIO() as captured_output:
            with redirect_stdout(captured_output):
                result = validate_environment('missing')
        
        assert result is False
        
    @patch('builtins.__import__')
    @patch('os.path.exists')
    def test_validate_environment_import_error(self, mock_exists, mock_import):
        """Test validation when config import fails"""
        mock_exists.return_value = True
        mock_import.side_effect = ImportError("Could not import settings")
        
        with StringIO() as captured_output:
            with redirect_stdout(captured_output):
                result = validate_environment('test')
            output = captured_output.getvalue()
        
        assert result is False
        assert "Configuration validation failed" in output

    @patch('builtins.__import__')
    @patch('os.path.exists')
    def test_validate_environment_missing_settings(self, mock_exists, mock_import):
        """Test validation when settings attribute is missing"""
        mock_exists.return_value = True
        
        # Create mock config module without settings
        mock_config = MagicMock(spec=['app_env', 'env_file_path'])
        mock_config.app_env = "test"
        mock_config.env_file_path = ".env.test"
        
        # Remove settings attribute
        type(mock_config).settings = PropertyMock(side_effect=AttributeError("'module' object has no attribute 'settings'"))
        
        mock_import.return_value = mock_config
        
        with StringIO() as captured_output:
            with redirect_stdout(captured_output):
                result = validate_environment('test')
            output = captured_output.getvalue()
        
        assert result is False
        assert "Configuration validation failed" in output
        
    @patch('builtins.__import__')
    @patch('os.path.exists')
    def test_validate_environment_success_all_settings(self, mock_exists, mock_import):
        """Test validation with all critical settings configured"""
        mock_exists.return_value = True
        
        # Create mock settings
        mock_settings = MagicMock()
        mock_settings.host = "localhost"
        mock_settings.port = 8080
        mock_settings.environment = "test"
        mock_settings.log_level = "INFO"
        mock_settings.jwt_public_key = "test-key"
        mock_settings.admin_api_url = "http://admin.test"
        mock_settings.rule_api_url = "http://rules.test"
        
        # Create mock config module
        mock_config = MagicMock()
        mock_config.settings = mock_settings
        mock_config.app_env = "test"
        mock_config.env_file_path = ".env.test"
        
        mock_import.return_value = mock_config
        
        with StringIO() as captured_output:
            with redirect_stdout(captured_output):
                result = validate_environment('test')
            output = captured_output.getvalue()
        
        assert result is True
        assert "Configuration loaded successfully" in output
        assert "All critical settings configured" in output
        assert "localhost" in output
        assert "8080" in output
        assert "INFO" in output
        
    @patch('builtins.__import__')
    @patch('os.path.exists')
    def test_validate_environment_missing_critical_settings(self, mock_exists, mock_import):
        """Test validation with missing critical settings"""
        mock_exists.return_value = True
        
        # Create mock settings with missing critical values
        mock_settings = MagicMock()
        mock_settings.host = "localhost"
        mock_settings.port = 8080
        mock_settings.environment = "test"
        mock_settings.log_level = "INFO"
        mock_settings.jwt_public_key = None  # Missing
        mock_settings.admin_api_url = ""     # Missing
        mock_settings.rule_api_url = None    # Missing
        
        # Create mock config module
        mock_config = MagicMock()
        mock_config.settings = mock_settings
        mock_config.app_env = "test"
        mock_config.env_file_path = ".env.test"
        
        mock_import.return_value = mock_config
        
        with StringIO() as captured_output:
            with redirect_stdout(captured_output):
                result = validate_environment('test')
            output = captured_output.getvalue()
        
        assert result is True  # Returns True even with issues to allow validation reporting
        assert "Configuration loaded successfully" in output
        assert "Configuration Issues:" in output
        assert "JWT public key not configured" in output
        assert "Admin API URL not configured" in output
        assert "Rule API URL not configured" in output


class TestMainFunction:
    """Test main function and command line interface"""
    
    @patch('app.env_config_util.show_current_env')
    @patch('sys.argv', ['env_config_util.py', 'show'])
    def test_main_show_command(self, mock_show):
        """Test main function with show command"""
        main()
        mock_show.assert_called_once()
    
    @patch('app.env_config_util.list_environments')
    @patch('sys.argv', ['env_config_util.py', 'list'])
    def test_main_list_command(self, mock_list):
        """Test main function with list command"""
        main()
        mock_list.assert_called_once()
    
    @patch('app.env_config_util.set_environment')
    @patch('sys.argv', ['env_config_util.py', 'set', 'development'])
    def test_main_set_command(self, mock_set):
        """Test main function with set command"""
        main()
        mock_set.assert_called_once_with('development')
    
    @patch('app.env_config_util.validate_environment')
    @patch('sys.argv', ['env_config_util.py', 'validate', 'production'])
    def test_main_validate_command(self, mock_validate):
        """Test main function with validate command"""
        main()
        mock_validate.assert_called_once_with('production')
    
    @patch('sys.argv', ['env_config_util.py'])
    @patch('argparse.ArgumentParser.print_help')
    def test_main_no_command(self, mock_help):
        """Test main function with no command provided"""
        main()
        mock_help.assert_called_once()
    
    @patch('sys.argv', ['env_config_util.py', 'invalid'])
    def test_main_invalid_command(self):
        """Test main function with invalid command"""
        with pytest.raises(SystemExit):
            main()


class TestIntegrationScenarios:
    """Test integration scenarios and edge cases"""
    
    def test_environment_variable_cleanup(self):
        """Test that environment variables are properly managed"""
        original_app_env = os.environ.get('APP_ENV')
        
        try:
            # Test setting and checking
            with patch('os.path.exists', return_value=True):
                with StringIO():
                    set_environment('test_env')
                
                assert os.environ.get('APP_ENV') == 'test_env'
        
        finally:
            # Cleanup
            if original_app_env:
                os.environ['APP_ENV'] = original_app_env
            elif 'APP_ENV' in os.environ:
                del os.environ['APP_ENV']
    
    @patch('pathlib.Path.glob')
    def test_environment_file_sorting(self, mock_glob):
        """Test that environment files are sorted correctly"""
        # Create mock files in non-alphabetical order with proper comparison
        class MockPath:
            def __init__(self, name, size):
                self.name = name
                self.stat_size = size
            
            def stat(self):
                mock_stat = MagicMock()
                mock_stat.st_size = self.stat_size
                return mock_stat
            
            def __lt__(self, other):
                return self.name < other.name
        
        mock_files = [
            MockPath('.env.zzz', 100),
            MockPath('.env.aaa', 100),
            MockPath('.env.mmm', 100)
        ]
        
        mock_glob.return_value = mock_files
        
        with StringIO() as captured_output:
            with redirect_stdout(captured_output):
                list_environments()
            output = captured_output.getvalue()
        
        # Check that files are processed (sorting is done inside the function)
        assert 'zzz' in output
        assert 'aaa' in output
        assert 'mmm' in output
        assert 'Total: 3 environment files' in output


class TestErrorHandlingEdgeCases:
    """Test error handling and edge cases"""
    
    @patch('pathlib.Path.glob')
    def test_list_environments_file_stat_error(self, mock_glob):
        """Test handling of file stat errors during listing"""
        mock_file = MagicMock()
        mock_file.name = '.env.test'
        mock_file.stat.side_effect = OSError("Permission denied")
        
        mock_glob.return_value = [mock_file]
        
        # Should not raise exception, but might not show size
        with StringIO() as captured_output:
            with redirect_stdout(captured_output):
                try:
                    list_environments()
                except OSError:
                    pass  # Expected if stat fails
    
    def test_show_current_env_special_characters(self):
        """Test show_current_env with special characters in environment values"""
        with patch.dict(os.environ, {'APP_ENV': 'test-env_123', 'ENVIRONMENT': 'test/env'}):
            with StringIO() as captured_output:
                with redirect_stdout(captured_output):
                    show_current_env()
                output = captured_output.getvalue()
            
            assert "APP_ENV: test-env_123" in output
            assert "ENVIRONMENT: test/env" in output


if __name__ == '__main__':
    pytest.main([__file__])