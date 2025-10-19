#!/usr/bin/env python3
"""
Comprehensive Unit Tests for Vector DB Utils
Tests schema filtering and ranking utilities used for data processing
"""

import pytest
from unittest.mock import Mock, patch
from typing import Dict, Any

from app.vector_db.utils import (
    filter_pii_columns,
    filter_by_dtype,
    rank_columns_by_sample_diversity
)


class TestFilterPIIColumns:
    """Test PII column filtering functionality"""
    
    def test_filter_pii_columns_basic_functionality(self):
        """Test basic PII filtering with mixed column types"""
        schema = {
            "customer_id": {
                "dtype": "string",
                "sample_values": ["CUST001", "CUST002"],
                "metadata": {"pii": False}
            },
            "email": {
                "dtype": "string", 
                "sample_values": ["user@example.com", "test@company.com"],
                "metadata": {"pii": True}
            },
            "phone": {
                "dtype": "string",
                "sample_values": ["+1-555-0123", "+1-555-0456"], 
                "metadata": {"pii": True}
            },
            "product_name": {
                "dtype": "string",
                "sample_values": ["Widget A", "Widget B"],
                "metadata": {"pii": False}
            }
        }
        
        result = filter_pii_columns(schema)
        
        assert len(result) == 2
        assert "email" in result
        assert "phone" in result
        assert "customer_id" not in result
        assert "product_name" not in result
        
        # Verify the filtered columns retain their original structure
        assert result["email"]["dtype"] == "string"
        assert result["email"]["metadata"]["pii"] is True
        assert result["phone"]["sample_values"] == ["+1-555-0123", "+1-555-0456"]
    
    def test_filter_pii_columns_no_pii_columns(self):
        """Test filtering when no columns are marked as PII"""
        schema = {
            "id": {
                "dtype": "integer",
                "sample_values": [1, 2, 3],
                "metadata": {"pii": False}
            },
            "status": {
                "dtype": "string",
                "sample_values": ["active", "inactive"],
                "metadata": {"pii": False}
            }
        }
        
        result = filter_pii_columns(schema)
        
        assert len(result) == 0
        assert result == {}
    
    def test_filter_pii_columns_all_pii_columns(self):
        """Test filtering when all columns are marked as PII"""
        schema = {
            "ssn": {
                "dtype": "string",
                "sample_values": ["123-45-6789", "987-65-4321"],
                "metadata": {"pii": True}
            },
            "credit_card": {
                "dtype": "string", 
                "sample_values": ["4111-1111-1111-1111", "5555-5555-5555-4444"],
                "metadata": {"pii": True}
            }
        }
        
        result = filter_pii_columns(schema)
        
        assert len(result) == 2
        assert "ssn" in result
        assert "credit_card" in result
        assert result == schema  # Should be identical to input
    
    def test_filter_pii_columns_missing_metadata(self):
        """Test filtering with columns missing metadata"""
        schema = {
            "name": {
                "dtype": "string",
                "sample_values": ["John Doe", "Jane Smith"]
                # No metadata field
            },
            "email": {
                "dtype": "string",
                "sample_values": ["user@example.com"],
                "metadata": {"pii": True}
            },
            "count": {
                "dtype": "integer", 
                "sample_values": [10, 20],
                "metadata": {}  # Empty metadata
            }
        }
        
        result = filter_pii_columns(schema)
        
        # Only email should be returned (has metadata with pii=True)
        assert len(result) == 1
        assert "email" in result
        assert "name" not in result  # No metadata
        assert "count" not in result  # No pii field in metadata
    
    def test_filter_pii_columns_missing_pii_field(self):
        """Test filtering with metadata present but no pii field"""
        schema = {
            "description": {
                "dtype": "string",
                "sample_values": ["Product description"],
                "metadata": {"category": "text", "indexed": True}
                # No pii field in metadata
            },
            "address": {
                "dtype": "string",
                "sample_values": ["123 Main St"],
                "metadata": {"pii": True, "sensitive": True}
            }
        }
        
        result = filter_pii_columns(schema)
        
        assert len(result) == 1
        assert "address" in result
        assert "description" not in result
    
    def test_filter_pii_columns_pii_false_explicitly(self):
        """Test filtering with pii explicitly set to False"""
        schema = {
            "public_info": {
                "dtype": "string", 
                "sample_values": ["Public data"],
                "metadata": {"pii": False}
            },
            "private_info": {
                "dtype": "string",
                "sample_values": ["Private data"],
                "metadata": {"pii": True}
            }
        }
        
        result = filter_pii_columns(schema)
        
        assert len(result) == 1
        assert "private_info" in result
        assert "public_info" not in result
    
    def test_filter_pii_columns_empty_schema(self):
        """Test filtering with empty schema"""
        result = filter_pii_columns({})
        
        assert result == {}
    
    def test_filter_pii_columns_non_boolean_pii_values(self):
        """Test filtering with non-boolean pii values"""
        schema = {
            "col1": {
                "dtype": "string",
                "metadata": {"pii": "yes"}  # String instead of boolean
            },
            "col2": {
                "dtype": "string", 
                "metadata": {"pii": 1}  # Integer instead of boolean
            },
            "col3": {
                "dtype": "string",
                "metadata": {"pii": True}  # Correct boolean
            }
        }
        
        result = filter_pii_columns(schema)
        
        # Only col3 should be returned (has pii=True as boolean)
        assert len(result) == 1
        assert "col3" in result


class TestFilterByDtype:
    """Test data type filtering functionality"""
    
    def test_filter_by_dtype_basic_functionality(self):
        """Test basic data type filtering"""
        schema = {
            "id": {
                "dtype": "integer",
                "sample_values": [1, 2, 3]
            },
            "name": {
                "dtype": "string",
                "sample_values": ["Alice", "Bob"]
            },
            "balance": {
                "dtype": "float", 
                "sample_values": [100.50, 200.75]
            },
            "active": {
                "dtype": "boolean",
                "sample_values": [True, False]
            }
        }
        
        # Filter for string and integer types only
        allowed_types = {"string", "integer"}
        result = filter_by_dtype(schema, allowed_types)
        
        assert len(result) == 2
        assert "id" in result
        assert "name" in result
        assert "balance" not in result  # float not allowed
        assert "active" not in result   # boolean not allowed
        
        # Verify filtered columns retain their structure
        assert result["id"]["dtype"] == "integer"
        assert result["name"]["sample_values"] == ["Alice", "Bob"]
    
    def test_filter_by_dtype_single_type(self):
        """Test filtering for a single data type"""
        schema = {
            "price": {
                "dtype": "float",
                "sample_values": [19.99, 29.99]
            },
            "cost": {
                "dtype": "float", 
                "sample_values": [10.50, 15.25]
            },
            "name": {
                "dtype": "string",
                "sample_values": ["Product A", "Product B"]
            }
        }
        
        # Filter for float type only
        result = filter_by_dtype(schema, {"float"})
        
        assert len(result) == 2
        assert "price" in result
        assert "cost" in result
        assert "name" not in result
    
    def test_filter_by_dtype_no_matching_types(self):
        """Test filtering when no columns match the allowed types"""
        schema = {
            "id": {
                "dtype": "integer",
                "sample_values": [1, 2]
            },
            "name": {
                "dtype": "string", 
                "sample_values": ["Test"]
            }
        }
        
        # Filter for types not present in schema
        result = filter_by_dtype(schema, {"date", "timestamp"})
        
        assert len(result) == 0
        assert result == {}
    
    def test_filter_by_dtype_all_types_match(self):
        """Test filtering when all columns match allowed types"""
        schema = {
            "first_name": {
                "dtype": "string",
                "sample_values": ["John", "Jane"]
            },
            "last_name": {
                "dtype": "string",
                "sample_values": ["Doe", "Smith"]
            }
        }
        
        result = filter_by_dtype(schema, {"string"})
        
        assert len(result) == 2
        assert result == schema  # Should be identical to input
    
    def test_filter_by_dtype_missing_dtype(self):
        """Test filtering with columns missing dtype field"""
        schema = {
            "valid_col": {
                "dtype": "string",
                "sample_values": ["test"]
            },
            "missing_dtype": {
                "sample_values": ["no dtype field"]
                # No dtype field
            },
            "none_dtype": {
                "dtype": None,
                "sample_values": ["dtype is None"]
            }
        }
        
        result = filter_by_dtype(schema, {"string"})
        
        # Only valid_col should be returned
        assert len(result) == 1
        assert "valid_col" in result
        assert "missing_dtype" not in result
        assert "none_dtype" not in result
    
    def test_filter_by_dtype_empty_schema(self):
        """Test filtering with empty schema"""
        result = filter_by_dtype({}, {"string", "integer"})
        
        assert result == {}
    
    def test_filter_by_dtype_empty_allowed_types(self):
        """Test filtering with empty allowed types set"""
        schema = {
            "col1": {
                "dtype": "string",
                "sample_values": ["test"]
            }
        }
        
        result = filter_by_dtype(schema, set())
        
        assert result == {}
    
    def test_filter_by_dtype_case_sensitivity(self):
        """Test that dtype matching is case sensitive"""
        schema = {
            "col1": {
                "dtype": "String",  # Capitalized
                "sample_values": ["test"]
            },
            "col2": {
                "dtype": "string",  # Lowercase
                "sample_values": ["test"]
            }
        }
        
        result = filter_by_dtype(schema, {"string"})
        
        # Only col2 should match (case sensitive)
        assert len(result) == 1
        assert "col2" in result
        assert "col1" not in result


class TestRankColumnsBySampleDiversity:
    """Test column ranking by sample diversity functionality"""
    
    def test_rank_columns_by_sample_diversity_basic_functionality(self):
        """Test basic ranking by sample value uniqueness"""
        schema = {
            "low_diversity": {
                "sample_values": ["A", "A", "A"]  # 1 unique value
            },
            "high_diversity": {
                "sample_values": ["X", "Y", "Z", "W"]  # 4 unique values
            },
            "medium_diversity": {
                "sample_values": ["1", "2", "1"]  # 2 unique values
            }
        }
        
        result = rank_columns_by_sample_diversity(schema)
        
        # Should be ranked by unique value count (descending)
        assert result == ["high_diversity", "medium_diversity", "low_diversity"]
    
    def test_rank_columns_by_sample_diversity_tie_handling(self):
        """Test ranking when columns have same diversity"""
        schema = {
            "col_b": {
                "sample_values": ["X", "Y"]  # 2 unique values
            },
            "col_a": {
                "sample_values": ["1", "2"]  # 2 unique values  
            },
            "col_c": {
                "sample_values": ["P", "Q"]  # 2 unique values
            }
        }
        
        result = rank_columns_by_sample_diversity(schema)
        
        # With equal diversity, should fall back to alphabetical order
        assert len(result) == 3
        assert set(result) == {"col_a", "col_b", "col_c"}
        # All have same diversity, so order depends on Python's sort stability
    
    def test_rank_columns_by_sample_diversity_empty_samples(self):
        """Test ranking with empty sample values"""
        schema = {
            "empty_samples": {
                "sample_values": []  # 0 unique values
            },
            "has_samples": {
                "sample_values": ["A", "B"]  # 2 unique values
            },
            "also_empty": {
                "sample_values": []  # 0 unique values
            }
        }
        
        result = rank_columns_by_sample_diversity(schema)
        
        # has_samples should be first, empty ones after
        assert result[0] == "has_samples"
        assert set(result[1:]) == {"empty_samples", "also_empty"}
    
    def test_rank_columns_by_sample_diversity_missing_sample_values(self):
        """Test ranking with columns missing sample_values field"""
        schema = {
            "has_samples": {
                "sample_values": ["A", "B", "C"]
            },
            "missing_samples": {
                "dtype": "string"
                # No sample_values field
            }
        }
        
        result = rank_columns_by_sample_diversity(schema)
        
        # has_samples should be first (3 unique values)
        # missing_samples should have 0 diversity (empty list default)
        assert result == ["has_samples", "missing_samples"]
    
    def test_rank_columns_by_sample_diversity_none_sample_values(self):
        """Test ranking when sample_values is None (should cause TypeError)"""
        schema = {
            "none_samples": {
                "sample_values": None
            }
        }
        
        # This should raise a TypeError because None is not iterable
        with pytest.raises(TypeError, match="'NoneType' object is not iterable"):
            rank_columns_by_sample_diversity(schema)
    
    def test_rank_columns_by_sample_diversity_duplicate_values(self):
        """Test ranking correctly handles duplicate values"""
        schema = {
            "many_duplicates": {
                "sample_values": ["A", "A", "A", "B", "B", "B"]  # 2 unique
            },
            "few_duplicates": {
                "sample_values": ["X", "Y", "Z", "X"]  # 3 unique
            },
            "no_duplicates": {
                "sample_values": ["P", "Q", "R", "S"]  # 4 unique
            }
        }
        
        result = rank_columns_by_sample_diversity(schema)
        
        assert result == ["no_duplicates", "few_duplicates", "many_duplicates"]
    
    def test_rank_columns_by_sample_diversity_mixed_types(self):
        """Test ranking with mixed data types in sample values"""
        schema = {
            "mixed_types": {
                "sample_values": ["1", 2, "3", 2, "1"]  # 3 unique: "1", 2, "3"
            },
            "strings_only": {
                "sample_values": ["A", "B"]  # 2 unique
            },
            "numbers_only": {
                "sample_values": [1, 2, 3, 4]  # 4 unique
            }
        }
        
        result = rank_columns_by_sample_diversity(schema)
        
        assert result == ["numbers_only", "mixed_types", "strings_only"]
    
    def test_rank_columns_by_sample_diversity_empty_schema(self):
        """Test ranking with empty schema"""
        result = rank_columns_by_sample_diversity({})
        
        assert result == []
    
    def test_rank_columns_by_sample_diversity_single_column(self):
        """Test ranking with single column"""
        schema = {
            "only_col": {
                "sample_values": ["A", "B", "C"]
            }
        }
        
        result = rank_columns_by_sample_diversity(schema)
        
        assert result == ["only_col"]
    
    def test_rank_columns_by_sample_diversity_complex_values(self):
        """Test ranking with complex sample values"""
        schema = {
            "dates": {
                "sample_values": ["2023-01-01", "2023-01-02", "2023-01-01"]  # 2 unique
            },
            "emails": {
                "sample_values": [
                    "user1@example.com",
                    "user2@example.com", 
                    "user3@example.com",
                    "user1@example.com"
                ]  # 3 unique
            },
            "ids": {
                "sample_values": ["ID001", "ID002", "ID003", "ID004", "ID005"]  # 5 unique
            }
        }
        
        result = rank_columns_by_sample_diversity(schema)
        
        assert result == ["ids", "emails", "dates"]


class TestIntegrationScenarios:
    """Test integration scenarios combining multiple functions"""
    
    def test_combined_filtering_workflow(self):
        """Test a typical workflow using multiple filtering functions"""
        schema = {
            "customer_id": {
                "dtype": "string",
                "sample_values": ["CUST001", "CUST002", "CUST003"],
                "metadata": {"pii": False}
            },
            "email": {
                "dtype": "string",
                "sample_values": ["user1@example.com", "user2@example.com"],
                "metadata": {"pii": True}
            },
            "age": {
                "dtype": "integer",
                "sample_values": [25, 30, 35],
                "metadata": {"pii": True}
            },
            "balance": {
                "dtype": "float",
                "sample_values": [100.0, 200.0], 
                "metadata": {"pii": False}
            }
        }
        
        # Step 1: Filter PII columns
        pii_columns = filter_pii_columns(schema)
        assert set(pii_columns.keys()) == {"email", "age"}
        
        # Step 2: From PII columns, filter by string type only
        string_pii = filter_by_dtype(pii_columns, {"string"})
        assert set(string_pii.keys()) == {"email"}
        
        # Step 3: Rank remaining columns by diversity
        ranked = rank_columns_by_sample_diversity(string_pii)
        assert ranked == ["email"]
    
    def test_filter_then_rank_workflow(self):
        """Test filtering by type then ranking by diversity"""
        schema = {
            "name": {
                "dtype": "string",
                "sample_values": ["Alice", "Bob", "Charlie"]  # 3 unique
            },
            "status": {
                "dtype": "string",
                "sample_values": ["active", "active", "inactive"]  # 2 unique
            },
            "category": {
                "dtype": "string", 
                "sample_values": ["A"]  # 1 unique
            },
            "count": {
                "dtype": "integer",
                "sample_values": [1, 2, 3, 4]  # Would be 4 unique, but filtered out
            }
        }
        
        # Filter for strings only
        string_cols = filter_by_dtype(schema, {"string"})
        assert len(string_cols) == 3
        
        # Rank by diversity
        ranked = rank_columns_by_sample_diversity(string_cols)
        assert ranked == ["name", "status", "category"]


class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_none_values_in_schema(self):
        """Test handling of None values in schema (should cause errors)"""
        schema = {
            "normal_col": {
                "dtype": "string",
                "sample_values": ["A", "B"],
                "metadata": {"pii": True}
            },
            "none_col": None  # Entire column definition is None
        }
        
        # filter_pii_columns should raise AttributeError when encountering None
        with pytest.raises(AttributeError, match="'NoneType' object has no attribute 'get'"):
            filter_pii_columns(schema)
        
        # filter_by_dtype should also raise AttributeError when encountering None
        with pytest.raises(AttributeError, match="'NoneType' object has no attribute 'get'"):
            filter_by_dtype(schema, {"string"})
    
    def test_malformed_column_definitions(self):
        """Test handling of malformed column definitions"""
        schema = {
            "good_col": {
                "dtype": "string",
                "sample_values": ["test"],
                "metadata": {"pii": True}
            },
            "bad_col": {
                "this_is": "malformed"
                # Missing expected fields
            }
        }
        
        # Functions should handle gracefully - bad_col won't match filters
        pii_result = filter_pii_columns(schema)
        assert len(pii_result) == 1
        assert "good_col" in pii_result
        assert "bad_col" not in pii_result  # No pii metadata
        
        dtype_result = filter_by_dtype(schema, {"string"})
        assert len(dtype_result) == 1
        assert "good_col" in dtype_result
        assert "bad_col" not in dtype_result  # No dtype field
        
        rank_result = rank_columns_by_sample_diversity(schema)
        assert len(rank_result) == 2  # Both columns included in ranking
        # good_col should rank higher (has sample_values)
        assert rank_result == ["good_col", "bad_col"]


class TestPerformance:
    """Test performance characteristics"""
    
    def test_large_schema_performance(self):
        """Test functions work efficiently with large schemas"""
        import time
        
        # Create a large schema with 1000 columns
        large_schema = {}
        for i in range(1000):
            large_schema[f"col_{i}"] = {
                "dtype": "string" if i % 2 == 0 else "integer",
                "sample_values": [f"val_{i}_{j}" for j in range(10)],
                "metadata": {"pii": i % 3 == 0}  # Every 3rd column is PII
            }
        
        # Test PII filtering performance
        start_time = time.time()
        pii_result = filter_pii_columns(large_schema)
        pii_time = time.time() - start_time
        
        # Test dtype filtering performance  
        start_time = time.time()
        dtype_result = filter_by_dtype(large_schema, {"string"})
        dtype_time = time.time() - start_time
        
        # Test ranking performance
        start_time = time.time()
        rank_result = rank_columns_by_sample_diversity(large_schema)
        rank_time = time.time() - start_time
        
        # All operations should complete in reasonable time (< 1 second)
        assert pii_time < 1.0, f"PII filtering took {pii_time:.3f}s"
        assert dtype_time < 1.0, f"Dtype filtering took {dtype_time:.3f}s"
        assert rank_time < 1.0, f"Ranking took {rank_time:.3f}s"
        
        # Verify results are correct
        assert len(pii_result) == 334  # ~1000/3 columns are PII
        assert len(dtype_result) == 500  # Half the columns are strings
        assert len(rank_result) == 1000  # All columns should be ranked


if __name__ == "__main__":
    # Run pytest for all tests
    pytest.main([__file__, "-v"])