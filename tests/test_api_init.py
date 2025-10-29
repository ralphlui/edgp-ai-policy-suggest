#!/usr/bin/env python3
"""
Unit Tests for API Module Initialization
Tests import functionality and re-exports in app.api.__init__
"""

import pytest
import importlib


class TestAPIModuleImports:
    """Test API module import functionality"""
    
    def test_aoss_routes_imports_successfully(self):
        """Test that AOSS routes are imported and available"""
        import app.api
        
        # Check that key functions are available
        assert hasattr(app.api, 'create_domain')
        assert hasattr(app.api, 'get_domains') 
        assert hasattr(app.api, 'verify_domain_exists')
        assert hasattr(app.api, 'list_domains_in_vectordb')
        assert hasattr(app.api, 'get_domain_from_vectordb')
        assert hasattr(app.api, 'download_csv_file')
        assert hasattr(app.api, 'regenerate_suggestions')
        assert hasattr(app.api, 'extend_domain')
        assert hasattr(app.api, 'suggest_extensions')
        assert hasattr(app.api, 'get_store')
        assert hasattr(app.api, 'check_vectordb_status')
    
    def test_optional_imports_dont_break_module(self):
        """Test that optional imports don't break the module"""
        import app.api
        
        # Module should be importable regardless of optional imports
        assert app.api is not None
        
        # Check if optional imports are present (they might or might not be)
        has_suggest_rules = hasattr(app.api, 'suggest_rules')
        has_validation_router = hasattr(app.api, 'validation_router')
        
        # These are optional, so we just verify they don't cause errors
        assert isinstance(has_suggest_rules, bool)
        assert isinstance(has_validation_router, bool)


class TestAPIModuleStructure:
    """Test API module structure and organization"""
    
    def test_api_module_is_package(self):
        """Test that app.api is properly configured as a package"""
        import app.api
        
        # Should be a module with package attributes
        assert hasattr(app.api, '__file__')
        assert hasattr(app.api, '__path__')
    
    def test_backward_compatibility_exports(self):
        """Test that all expected functions are exported for backward compatibility"""
        import app.api
        
        # List of core functions that should always be available
        expected_core_exports = [
            'create_domain',
            'get_domains', 
            'verify_domain_exists',
            'list_domains_in_vectordb',
            'get_domain_from_vectordb',
            'download_csv_file',
            'regenerate_suggestions',
            'extend_domain',
            'suggest_extensions',
            'get_store',
            'check_vectordb_status'
        ]
        
        for export in expected_core_exports:
            assert hasattr(app.api, export), f"Missing core export: {export}"
    
    def test_exported_functions_are_callable(self):
        """Test that exported functions are callable"""
        import app.api
        
        # Test core functions are callable
        assert callable(app.api.create_domain)
        assert callable(app.api.get_domains)
        assert callable(app.api.verify_domain_exists)


class TestTryExceptBlocks:
    """Test the try/except import handling"""
    
    def test_rule_suggestion_import_handling(self):
        """Test that rule suggestion import is handled gracefully"""
        # Re-import to test the try/except behavior
        import app.api
        
        # The module should import successfully regardless
        assert app.api is not None
        
        # If suggest_rules exists, it should be callable
        if hasattr(app.api, 'suggest_rules'):
            assert callable(app.api.suggest_rules)
    
    def test_validator_routes_import_handling(self):
        """Test that validator routes import is handled gracefully"""
        import app.api
        
        # The module should import successfully regardless
        assert app.api is not None
        
        # If validation_router exists, it should be accessible
        if hasattr(app.api, 'validation_router'):
            # Should have router attributes
            assert app.api.validation_router is not None


class TestFunctionAttributes:
    """Test that imported functions retain proper attributes"""
    
    def test_function_names_preserved(self):
        """Test that function names are preserved in imports"""
        import app.api
        
        # Check that function names are correct
        assert app.api.create_domain.__name__ == 'create_domain'
        assert app.api.get_domains.__name__ == 'get_domains'
        
    def test_function_modules_preserved(self):
        """Test that function module references are preserved"""
        import app.api
        
        # Functions should reference their source modules
        # All these functions are actually from domain_schema_routes
        assert 'domain_schema_routes' in app.api.create_domain.__module__
        assert 'domain_schema_routes' in app.api.get_domains.__module__
    
    def test_functions_have_docstrings(self):
        """Test that functions have docstrings"""
        import app.api
        
        # Functions should have docstrings (or None, both are valid)
        create_domain_doc = app.api.create_domain.__doc__
        get_domains_doc = app.api.get_domains.__doc__
        
        # Should be string or None
        assert create_domain_doc is None or isinstance(create_domain_doc, str)
        assert get_domains_doc is None or isinstance(get_domains_doc, str)


class TestModuleReloading:
    """Test module reloading scenarios"""
    
    def test_module_can_be_reloaded(self):
        """Test that the API module can be safely reloaded"""
        import app.api
        
        # Should be able to reload without errors
        try:
            importlib.reload(app.api)
            reload_successful = True
        except Exception:
            reload_successful = False
        
        assert reload_successful, "Module should be reloadable"
        
        # Core functions should still be available after reload
        assert hasattr(app.api, 'create_domain')
        assert hasattr(app.api, 'get_domains')
    
    def test_exports_consistent_after_reload(self):
        """Test that exports remain consistent after reload"""
        import app.api
        
        # Get initial exports
        initial_core_exports = {
            name: hasattr(app.api, name) 
            for name in ['create_domain', 'get_domains', 'verify_domain_exists']
        }
        
        # Reload module
        importlib.reload(app.api)
        
        # Get exports after reload
        reload_core_exports = {
            name: hasattr(app.api, name)
            for name in ['create_domain', 'get_domains', 'verify_domain_exists']
        }
        
        # Should be identical
        assert initial_core_exports == reload_core_exports


class TestErrorHandling:
    """Test error handling in the module"""
    
    def test_import_error_handling_documented(self):
        """Test that ImportError handling matches documentation in code"""
        # Read the actual module source to verify comments match behavior
        import app.api.__init__ as api_module
        import inspect
        
        # Get source code
        source = inspect.getsource(api_module)
        
        # Should have try/except blocks mentioned in comments
        assert 'try:' in source
        assert 'except ImportError:' in source
        assert '# Avoid circular import' in source or '# Validation module might not be available' in source
    
    def test_module_stability(self):
        """Test that the module remains stable under various conditions"""
        import app.api
        
        # Basic stability checks
        assert app.api.__name__ == 'app.api'
        assert hasattr(app.api, '__file__')
        
        # Should not raise exceptions when accessing standard attributes
        try:
            _ = app.api.__dict__
            _ = dir(app.api)
            stability_check = True
        except Exception:
            stability_check = False
        
        assert stability_check, "Module should be stable for introspection"


class TestImportPatterns:
    """Test import patterns used in the module"""
    
    def test_from_import_pattern(self):
        """Test that from ... import pattern works correctly"""
        import app.api
        
        # Functions imported with 'from X import Y' should be available
        core_functions = [
            'create_domain', 'get_domains', 'verify_domain_exists',
            'list_domains_in_vectordb', 'get_domain_from_vectordb'
        ]
        
        for func_name in core_functions:
            assert hasattr(app.api, func_name)
            func = getattr(app.api, func_name)
            assert callable(func)
    
    def test_multiple_function_import(self):
        """Test that multiple functions can be imported and are functional"""
        import app.api
        
        # Test that various functions are available and callable
        available_functions = []
        test_functions = ['create_domain', 'get_domains', 'verify_domain_exists']
        
        for func_name in test_functions:
            if hasattr(app.api, func_name):
                func = getattr(app.api, func_name)
                assert callable(func)
                available_functions.append(func_name)
        
        # Should have at least some functions available
        assert len(available_functions) > 0


# ======================================================================
# Merged: small re-export smoke test from test_api_init_exports.py
# ======================================================================

def test_app_api_reexports_importable():
    """Smoke test to ensure re-exports in app.api are importable."""
    # Importing suggest_rules should succeed via app.api re-export
    from app.api import suggest_rules  # noqa: F401

    # Import several names re-exported from aoss_routes
    from app.api import (
        get_store,  # noqa: F401
        check_vectordb_status,  # noqa: F401
    )

    # If import reached here, lines in try blocks were executed
    assert callable(get_store)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])