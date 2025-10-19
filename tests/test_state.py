#!/usr/bin/env python3
"""
Unit Tests for State Management Module
Tests LangGraphState and ColumnInfo Pydantic models
"""

import pytest
from typing import List, Dict
from pydantic import ValidationError

from app.state.state import ColumnInfo, LangGraphState


class TestColumnInfo:
    """Test ColumnInfo model"""
    
    def test_column_info_creation_valid(self):
        """Test creating ColumnInfo with valid data"""
        column = ColumnInfo(
            dtype="string",
            sample_values=["value1", "value2", "value3"]
        )
        
        assert column.dtype == "string"
        assert column.sample_values == ["value1", "value2", "value3"]
    
    def test_column_info_creation_empty_samples(self):
        """Test creating ColumnInfo with empty sample values"""
        column = ColumnInfo(
            dtype="integer",
            sample_values=[]
        )
        
        assert column.dtype == "integer"
        assert column.sample_values == []
    
    def test_column_info_creation_numeric_types(self):
        """Test ColumnInfo with various numeric data types"""
        column = ColumnInfo(
            dtype="float",
            sample_values=["1.5", "2.7", "3.14"]
        )
        
        assert column.dtype == "float"
        assert column.sample_values == ["1.5", "2.7", "3.14"]
    
    def test_column_info_creation_date_type(self):
        """Test ColumnInfo with date data type"""
        column = ColumnInfo(
            dtype="date",
            sample_values=["2023-01-01", "2023-12-31"]
        )
        
        assert column.dtype == "date"
        assert column.sample_values == ["2023-01-01", "2023-12-31"]
    
    def test_column_info_missing_dtype(self):
        """Test ColumnInfo creation fails without dtype"""
        with pytest.raises(ValidationError) as exc_info:
            ColumnInfo(sample_values=["test"])
        
        assert "dtype" in str(exc_info.value)
    
    def test_column_info_missing_sample_values(self):
        """Test ColumnInfo creation fails without sample_values"""
        with pytest.raises(ValidationError) as exc_info:
            ColumnInfo(dtype="string")
        
        assert "sample_values" in str(exc_info.value)
    
    def test_column_info_invalid_sample_values_type(self):
        """Test ColumnInfo creation fails with non-list sample_values"""
        with pytest.raises(ValidationError):
            ColumnInfo(dtype="string", sample_values="not_a_list")
    
    def test_column_info_json_serialization(self):
        """Test ColumnInfo can be serialized to JSON"""
        column = ColumnInfo(
            dtype="boolean",
            sample_values=["true", "false"]
        )
        
        json_data = column.model_dump()
        assert json_data == {
            "dtype": "boolean",
            "sample_values": ["true", "false"]
        }


class TestLangGraphState:
    """Test LangGraphState model"""
    
    def test_langraph_state_minimal_creation(self):
        """Test creating LangGraphState with only required field"""
        state = LangGraphState(domain="test_domain")
        
        assert state.domain == "test_domain"
        assert state.schema is None
        assert state.rules is None
        assert state.query_embedding is None
        assert state.results is None
        assert state.filtered_columns is None
        assert state.pii_only is False
        assert state.allowed_types == ["string", "integer", "date"]
        assert state.csv_ready is False
    
    def test_langraph_state_full_creation(self):
        """Test creating LangGraphState with all fields populated"""
        schema_data = {
            "column1": ColumnInfo(dtype="string", sample_values=["a", "b"]),
            "column2": ColumnInfo(dtype="integer", sample_values=["1", "2"])
        }
        
        state = LangGraphState(
            domain="test_domain",
            schema=schema_data,
            rules=["rule1", "rule2"],
            query_embedding=[0.1, 0.2, 0.3],
            results=[{"result": "value"}],
            filtered_columns=[{"column": "info"}],
            pii_only=True,
            allowed_types=["string", "float"],
            csv_ready=True
        )
        
        assert state.domain == "test_domain"
        assert state.schema == schema_data
        assert state.rules == ["rule1", "rule2"]
        assert state.query_embedding == [0.1, 0.2, 0.3]
        assert state.results == [{"result": "value"}]
        assert state.filtered_columns == [{"column": "info"}]
        assert state.pii_only is True
        assert state.allowed_types == ["string", "float"]
        assert state.csv_ready is True
    
    def test_langraph_state_missing_domain(self):
        """Test LangGraphState creation fails without domain"""
        with pytest.raises(ValidationError) as exc_info:
            LangGraphState()
        
        assert "domain" in str(exc_info.value)
    
    def test_langraph_state_schema_with_column_info(self):
        """Test LangGraphState with properly typed schema"""
        column1 = ColumnInfo(dtype="string", sample_values=["val1", "val2"])
        column2 = ColumnInfo(dtype="date", sample_values=["2023-01-01"])
        
        state = LangGraphState(
            domain="finance",
            schema={"col1": column1, "col2": column2}
        )
        
        assert state.domain == "finance"
        assert len(state.schema) == 2
        assert state.schema["col1"].dtype == "string"
        assert state.schema["col2"].dtype == "date"
    
    def test_langraph_state_empty_rules(self):
        """Test LangGraphState with empty rules list"""
        state = LangGraphState(
            domain="test_domain",
            rules=[]
        )
        
        assert state.domain == "test_domain"
        assert state.rules == []
    
    def test_langraph_state_complex_query_embedding(self):
        """Test LangGraphState with complex query embedding"""
        embedding = [0.123, -0.456, 0.789, -0.012, 0.345]
        
        state = LangGraphState(
            domain="test_domain",
            query_embedding=embedding
        )
        
        assert state.query_embedding == embedding
    
    def test_langraph_state_multiple_results(self):
        """Test LangGraphState with multiple results"""
        results = [
            {"id": 1, "score": 0.9},
            {"id": 2, "score": 0.8},
            {"id": 3, "score": 0.7}
        ]
        
        state = LangGraphState(
            domain="test_domain",
            results=results
        )
        
        assert state.results == results
        assert len(state.results) == 3
    
    def test_langraph_state_filtered_columns_structure(self):
        """Test LangGraphState with structured filtered columns"""
        filtered_cols = [
            {"name": "email", "type": "string", "pii": True},
            {"name": "age", "type": "integer", "pii": False}
        ]
        
        state = LangGraphState(
            domain="test_domain",
            filtered_columns=filtered_cols
        )
        
        assert state.filtered_columns == filtered_cols
    
    def test_langraph_state_pii_only_toggle(self):
        """Test LangGraphState with pii_only flag variations"""
        # Test explicit True
        state1 = LangGraphState(domain="test", pii_only=True)
        assert state1.pii_only is True
        
        # Test explicit False  
        state2 = LangGraphState(domain="test", pii_only=False)
        assert state2.pii_only is False
        
        # Test default (should be False)
        state3 = LangGraphState(domain="test")
        assert state3.pii_only is False
    
    def test_langraph_state_allowed_types_customization(self):
        """Test LangGraphState with custom allowed_types"""
        custom_types = ["string", "float", "boolean", "datetime"]
        
        state = LangGraphState(
            domain="test_domain",
            allowed_types=custom_types
        )
        
        assert state.allowed_types == custom_types
    
    def test_langraph_state_allowed_types_default(self):
        """Test LangGraphState default allowed_types"""
        state = LangGraphState(domain="test_domain")
        
        assert state.allowed_types == ["string", "integer", "date"]
    
    def test_langraph_state_csv_ready_flag(self):
        """Test LangGraphState csv_ready flag variations"""
        # Test explicit True
        state1 = LangGraphState(domain="test", csv_ready=True)
        assert state1.csv_ready is True
        
        # Test explicit False
        state2 = LangGraphState(domain="test", csv_ready=False)
        assert state2.csv_ready is False
        
        # Test default (should be False)
        state3 = LangGraphState(domain="test")
        assert state3.csv_ready is False
    
    def test_langraph_state_json_serialization(self):
        """Test LangGraphState can be serialized to JSON"""
        state = LangGraphState(
            domain="test_domain",
            rules=["rule1"],
            pii_only=True,
            csv_ready=True
        )
        
        json_data = state.model_dump()
        expected = {
            "domain": "test_domain",
            "schema": None,
            "rules": ["rule1"],
            "query_embedding": None,
            "results": None,
            "filtered_columns": None,
            "pii_only": True,
            "allowed_types": ["string", "integer", "date"],
            "csv_ready": True
        }
        
        assert json_data == expected
    
    def test_langraph_state_json_serialization_with_schema(self):
        """Test LangGraphState JSON serialization with schema"""
        column = ColumnInfo(dtype="string", sample_values=["a", "b"])
        state = LangGraphState(
            domain="test_domain",
            schema={"col1": column}
        )
        
        json_data = state.model_dump()
        
        assert json_data["domain"] == "test_domain"
        assert json_data["schema"]["col1"]["dtype"] == "string"
        assert json_data["schema"]["col1"]["sample_values"] == ["a", "b"]


class TestIntegrationScenarios:
    """Test integration scenarios for state management"""
    
    def test_state_workflow_simulation(self):
        """Test a complete state workflow simulation"""
        # Step 1: Initial state creation
        state = LangGraphState(domain="finance")
        assert state.csv_ready is False
        
        # Step 2: Add schema
        schema = {
            "account_id": ColumnInfo(dtype="string", sample_values=["ACC001", "ACC002"]),
            "balance": ColumnInfo(dtype="float", sample_values=["100.50", "200.75"]),
            "email": ColumnInfo(dtype="string", sample_values=["user@example.com"])
        }
        state.schema = schema
        
        # Step 3: Add rules
        state.rules = ["No PII in reports", "Balance must be positive"]
        
        # Step 4: Add query embedding
        state.query_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        # Step 5: Add results
        state.results = [
            {"rule_id": 1, "confidence": 0.95},
            {"rule_id": 2, "confidence": 0.87}
        ]
        
        # Step 6: Filter columns (PII detection)
        state.filtered_columns = [
            {"name": "account_id", "pii": False},
            {"name": "email", "pii": True}
        ]
        state.pii_only = True
        
        # Step 7: Mark as ready for CSV export
        state.csv_ready = True
        
        # Verify final state
        assert state.domain == "finance"
        assert len(state.schema) == 3
        assert len(state.rules) == 2
        assert len(state.query_embedding) == 5
        assert len(state.results) == 2
        assert len(state.filtered_columns) == 2
        assert state.pii_only is True
        assert state.csv_ready is True
    
    def test_state_partial_updates(self):
        """Test partial state updates maintain consistency"""
        state = LangGraphState(domain="healthcare")
        
        # Add schema first
        state.schema = {
            "patient_id": ColumnInfo(dtype="string", sample_values=["P001"]),
            "diagnosis": ColumnInfo(dtype="string", sample_values=["Condition A"])
        }
        
        # Update allowed types
        state.allowed_types = ["string", "integer"]
        
        # Add filtered columns
        state.filtered_columns = [{"name": "patient_id", "pii": True}]
        
        # Verify state consistency
        assert state.domain == "healthcare"
        assert "patient_id" in state.schema
        assert "string" in state.allowed_types
        assert state.filtered_columns[0]["pii"] is True
        assert state.csv_ready is False  # Default should remain
    
    def test_state_reset_simulation(self):
        """Test state can be reset to initial conditions"""
        # Create fully populated state
        state = LangGraphState(
            domain="retail",
            schema={"product": ColumnInfo(dtype="string", sample_values=["A"])},
            rules=["Rule 1"],
            query_embedding=[0.1],
            results=[{"test": "value"}],
            filtered_columns=[{"col": "info"}],
            pii_only=True,
            allowed_types=["string"],
            csv_ready=True
        )
        
        # Reset to minimal state
        state.schema = None
        state.rules = None
        state.query_embedding = None
        state.results = None
        state.filtered_columns = None
        state.pii_only = False
        state.allowed_types = ["string", "integer", "date"]
        state.csv_ready = False
        
        # Verify reset state
        assert state.domain == "retail"  # Domain should remain
        assert state.schema is None
        assert state.rules is None
        assert state.query_embedding is None
        assert state.results is None
        assert state.filtered_columns is None
        assert state.pii_only is False
        assert state.allowed_types == ["string", "integer", "date"]
        assert state.csv_ready is False


class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_column_info_very_long_sample_values(self):
        """Test ColumnInfo with very long sample values list"""
        long_samples = [f"value_{i}" for i in range(1000)]
        
        column = ColumnInfo(
            dtype="string",
            sample_values=long_samples
        )
        
        assert len(column.sample_values) == 1000
        assert column.sample_values[0] == "value_0"
        assert column.sample_values[-1] == "value_999"
    
    def test_langraph_state_very_long_embedding(self):
        """Test LangGraphState with very long embedding vector"""
        long_embedding = [float(i) for i in range(1536)]  # Common embedding size
        
        state = LangGraphState(
            domain="test",
            query_embedding=long_embedding
        )
        
        assert len(state.query_embedding) == 1536
        assert state.query_embedding[0] == 0.0
        assert state.query_embedding[-1] == 1535.0
    
    def test_langraph_state_empty_allowed_types(self):
        """Test LangGraphState with empty allowed_types"""
        state = LangGraphState(
            domain="test",
            allowed_types=[]
        )
        
        assert state.allowed_types == []
    
    def test_column_info_unicode_samples(self):
        """Test ColumnInfo with unicode sample values"""
        unicode_samples = ["测试", "тест", "🚀", "café"]
        
        column = ColumnInfo(
            dtype="string",
            sample_values=unicode_samples
        )
        
        assert column.sample_values == unicode_samples
        assert "测试" in column.sample_values
        assert "🚀" in column.sample_values


if __name__ == "__main__":
    pytest.main([__file__, "-v"])