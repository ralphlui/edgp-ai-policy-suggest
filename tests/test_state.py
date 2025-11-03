#!/usr/bin/env python3
"""
Unit Tests for State Management Module
Tests AgentState, AgentStep, AgentPlan, and ColumnInfo Pydantic models
"""

import pytest
from typing import List, Dict
from pydantic import ValidationError
import time

from app.state.state import ColumnInfo, AgentState, AgentStep, AgentPlan


class TestColumnInfo:
    """Test ColumnInfo model"""
    
    def test_column_info_basic_creation(self):
        """Test basic ColumnInfo creation"""
        column = ColumnInfo(dtype="string", sample_values=["a", "b", "c"])
        
        assert column.dtype == "string"
        assert column.sample_values == ["a", "b", "c"]
    
    def test_column_info_different_types(self):
        """Test ColumnInfo with different data types"""
        # String type
        str_column = ColumnInfo(dtype="string", sample_values=["value1", "value2"])
        assert str_column.dtype == "string"
        
        # Integer type
        int_column = ColumnInfo(dtype="integer", sample_values=["1", "2", "3"])
        assert int_column.dtype == "integer"
        
        # Date type
        date_column = ColumnInfo(dtype="date", sample_values=["2023-01-01", "2023-12-31"])
        assert date_column.dtype == "date"
    
    def test_column_info_empty_samples(self):
        """Test ColumnInfo with empty sample values"""
        column = ColumnInfo(dtype="string", sample_values=[])
        
        assert column.dtype == "string"
        assert column.sample_values == []
    
    def test_column_info_large_samples(self):
        """Test ColumnInfo with many sample values"""
        samples = [f"value_{i}" for i in range(100)]
        column = ColumnInfo(dtype="string", sample_values=samples)
        
        assert len(column.sample_values) == 100
        assert column.sample_values[0] == "value_0"
        assert column.sample_values[-1] == "value_99"


class TestAgentStep:
    """Test AgentStep model"""
    
    def test_agent_step_basic_creation(self):
        """Test basic AgentStep creation"""
        step = AgentStep(
            step_id="step_1",
            action="fetch",
            thought="Need to fetch rules",
            observation="Rules fetched successfully"
        )
        
        assert step.step_id == "step_1"
        assert step.action == "fetch"
        assert step.thought == "Need to fetch rules"
        assert step.observation == "Rules fetched successfully"
        assert step.reflection is None
        assert isinstance(step.timestamp, float)
        assert step.metadata == {}
    
    def test_agent_step_with_optional_fields(self):
        """Test AgentStep with all fields"""
        metadata = {"duration": 1.5, "success": True}
        step = AgentStep(
            step_id="step_2",
            action="suggest",
            thought="Generate suggestions",
            observation="Suggestions generated",
            reflection="Good quality suggestions",
            metadata=metadata
        )
        
        assert step.reflection == "Good quality suggestions"
        assert step.metadata == metadata


class TestAgentPlan:
    """Test AgentPlan model"""
    
    def test_agent_plan_basic_creation(self):
        """Test basic AgentPlan creation"""
        plan = AgentPlan(
            goal="Generate validation rules",
            steps=["fetch", "suggest", "format"]
        )
        
        assert plan.goal == "Generate validation rules"
        assert plan.steps == ["fetch", "suggest", "format"]
        assert plan.current_step == 0
        assert plan.context == {}
        assert plan.constraints == []
    
    def test_agent_plan_with_optional_fields(self):
        """Test AgentPlan with all fields"""
        context = {"domain": "finance", "complexity": "high"}
        constraints = ["no_pii", "fast_execution"]
        
        plan = AgentPlan(
            goal="Generate complex rules",
            steps=["analyze", "fetch", "suggest", "validate"],
            current_step=1,
            context=context,
            constraints=constraints
        )
        
        assert plan.current_step == 1
        assert plan.context == context
        assert plan.constraints == constraints


class TestAgentState:
    """Test AgentState model"""
    
    def test_agent_state_minimal_creation(self):
        """Test creating AgentState with only required field"""
        schema = {"column1": {"dtype": "string", "sample_values": ["a", "b"]}}
        state = AgentState(data_schema=schema)
        
        assert state.data_schema == schema
        assert state.gx_rules is None
        assert state.raw_suggestions is None
        assert state.rule_suggestions is None
        assert state.thoughts == []
        assert state.observations == []
        assert state.reflections == []
        assert state.errors == []
        assert state.retry_count == 0
        assert state.max_retries == 2
    
    def test_agent_state_with_plan(self):
        """Test AgentState with AgentPlan"""
        schema = {"column1": {"dtype": "string"}}
        plan = AgentPlan(goal="test", steps=["step1", "step2"])
        
        state = AgentState(data_schema=schema, plan=plan)
        
        assert state.plan == plan
        assert state.plan.goal == "test"
    
    def test_agent_state_with_steps(self):
        """Test AgentState with step history"""
        schema = {"column1": {"dtype": "string"}}
        step = AgentStep(
            step_id="test_step",
            action="test_action",
            thought="test thought",
            observation="test observation"
        )
        
        state = AgentState(data_schema=schema, step_history=[step])
        
        assert len(state.step_history) == 1
        assert state.step_history[0].step_id == "test_step"
    
    def test_agent_state_missing_schema(self):
        """Test AgentState creation fails without data_schema"""
        with pytest.raises(ValidationError) as exc_info:
            AgentState()
        
        assert "data_schema" in str(exc_info.value)
    
    def test_agent_state_with_suggestions(self):
        """Test AgentState with rule suggestions"""
        schema = {"column1": {"dtype": "string"}}
        suggestions = [
            {"rule_name": "ExpectColumnValuesToNotBeNull", "column_name": "column1"},
            {"rule_name": "ExpectColumnValuesToBeInTypeList", "column_name": "column1"}
        ]
        
        state = AgentState(data_schema=schema, rule_suggestions=suggestions)
        
        assert state.rule_suggestions == suggestions
        assert len(state.rule_suggestions) == 2
    
    def test_agent_state_execution_metrics(self):
        """Test AgentState with execution metrics"""
        schema = {"column1": {"dtype": "string"}}
        metrics = {"total_execution_time": 5.2, "steps_completed": 4}
        
        state = AgentState(data_schema=schema, execution_metrics=metrics)
        
        assert state.execution_metrics == metrics
        assert isinstance(state.execution_start_time, float)


class TestIntegrationScenarios:
    """Test integration scenarios for state management"""
    
    def test_full_agent_workflow_simulation(self):
        """Test a complete agent workflow simulation"""
        # Create initial state
        schema = {
            "user_id": {"dtype": "string", "sample_values": ["U001", "U002"]},
            "balance": {"dtype": "float", "sample_values": ["100.50", "200.75"]}
        }
        
        plan = AgentPlan(
            goal="Generate validation rules for financial data",
            steps=["fetch", "suggest", "format", "convert"]
        )
        
        state = AgentState(data_schema=schema, plan=plan)
        
        # Simulate workflow steps
        step1 = AgentStep(
            step_id="fetch_1",
            action="fetch",
            thought="Fetching GX rules",
            observation="Rules fetched successfully"
        )
        
        state.step_history.append(step1)
        state.thoughts.append("Starting rule generation process")
        
        # Add suggestions
        suggestions = [
            {"rule_name": "ExpectColumnValuesToNotBeNull", "column_name": "user_id"},
            {"rule_name": "ExpectColumnValuesToBeOfType", "column_name": "balance"}
        ]
        state.rule_suggestions = suggestions
        
        # Verify final state
        assert len(state.step_history) == 1
        assert len(state.thoughts) == 1
        assert len(state.rule_suggestions) == 2
        assert state.plan.goal == "Generate validation rules for financial data"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])