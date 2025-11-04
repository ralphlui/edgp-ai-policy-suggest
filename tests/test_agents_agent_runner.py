"""
Tests for app/agents/agent_runner.py module
Tests updated to match LangGraph-based implementation
"""
import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import time
import json


@pytest.fixture(autouse=True)
def _disable_validation_context(monkeypatch):
    import app.agents.agent_runner as _ar

    class _DummyCtx:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(_ar, '_setup_validation_context', lambda *a, **k: _DummyCtx())
    yield


class TestAgentRunnerModule:
    """Test agent_runner.py module functionality"""
    
    def test_agent_runner_module_import(self):
        """Test agent_runner module can be imported"""
        from app.agents import agent_runner
        assert hasattr(agent_runner, '__name__')

    def test_agent_runner_functions_exist(self):
        """Test agent runner functions exist"""
        from app.agents.agent_runner import run_agent, build_graph
        from app.state.state import AgentState
        assert callable(run_agent)
        assert callable(build_graph)
        assert AgentState is not None

    @patch('app.agents.agent_runner.fetch_gx_rules')
    @patch('app.agents.agent_runner.suggest_column_rules')
    @patch('app.agents.agent_runner.format_gx_rules')
    @patch('app.agents.agent_runner.normalize_rule_suggestions')
    @patch('app.agents.agent_runner.convert_to_rule_ms_format')
    def test_run_agent_basic_mock(self, mock_convert, mock_normalize, mock_format, mock_suggest, mock_fetch):
        """Test basic agent running with mocks"""
        # Setup mocks
        mock_fetch.invoke.return_value = ["mock_rule"]
        mock_suggest.invoke.return_value = "raw suggestions"
        mock_format.invoke.return_value = ["formatted rules"]
        mock_normalize.invoke.return_value = {"col1": {"expectations": ["rule1"]}}
        mock_convert.invoke.return_value = [{"rule": "test"}]
        
        from app.agents.agent_runner import run_agent
        
        result = run_agent({"col1": {"type": "string"}})
        assert isinstance(result, list)

    @patch('app.agents.agent_runner.build_graph')
    def test_run_agent_with_tools(self, mock_build_graph):
        """Test agent running with graph execution"""
        mock_graph = Mock()
        mock_graph.invoke.return_value = Mock(rule_suggestions=[{"rule": "test"}])
        mock_build_graph.return_value = mock_graph
        
        from app.agents.agent_runner import run_agent
        
        result = run_agent({"col1": {"type": "string"}})
        assert isinstance(result, list)
        mock_build_graph.assert_called_once()

    @patch('app.agents.agent_runner.build_graph')
    def test_run_agent_error_handling(self, mock_build_graph):
        """Test agent error handling"""
        mock_graph = Mock()
        mock_graph.invoke.side_effect = Exception("Graph execution failed")
        mock_build_graph.return_value = mock_graph
        
        from app.agents.agent_runner import run_agent
        
        result = run_agent({"col1": {"type": "string"}})
        assert result == []

    @patch('app.agents.agent_runner.build_graph')
    @patch('app.agents.agent_runner._setup_validation_context')
    def test_run_agent_complex_query(self, mock_setup_validation, mock_build_graph):
        """Test agent with complex schema"""
        from app.state.state import AgentState
        
        # Mock validation context to return None (no validation)
        mock_setup_validation.return_value = None
        
        # Create a proper AgentState mock result
        mock_result = AgentState(data_schema={"test": "schema"})
        mock_result.rule_suggestions = [
            {"rule": "test1", "column": "col1"},
            {"rule": "test2", "column": "col2"}
        ]
        mock_result.thoughts = ["thought1", "thought2"]
        mock_result.observations = ["obs1", "obs2"]
        mock_result.reflections = ["reflection1"]
        mock_result.execution_metrics = {"total_execution_time": 1.5}
        mock_result.step_history = []
        
        mock_graph = Mock()
        mock_graph.invoke.return_value = mock_result
        mock_build_graph.return_value = mock_graph
        
        from app.agents.agent_runner import run_agent
        
        complex_schema = {
            "domain": "test_domain",
            "col1": {"type": "string", "nullable": False},
            "col2": {"type": "integer", "nullable": True},
            "col3": {"type": "date", "nullable": False}
        }
        
        result = run_agent(complex_schema)
        assert isinstance(result, list)
        assert len(result) == 2


class TestAgentRunnerUtilities:
    """Test utility functions and configurations"""
    
    def test_module_level_constants(self):
        """Test module level constants exist"""
        from app.agents import agent_runner
        assert hasattr(agent_runner, 'logger')

    def test_logging_functionality(self):
        """Test logging configuration"""
        from app.agents.agent_runner import logger
        assert logger.name == 'app.agents.agent_runner'


class TestAgentModels:
    """Test the Pydantic models used in the agent runner"""
    
    def test_agent_step_model(self):
        """Test AgentStep model initialization and validation"""
        from app.agents.agent_runner import AgentStep
        
        # Test basic initialization
        step = AgentStep(
            step_id="test_1",
            action="fetch_rules",
            thought="I need to get rules",
            observation="Found some rules"
        )
        assert step.step_id == "test_1"
        assert step.action == "fetch_rules"
        assert step.thought == "I need to get rules"
        assert step.observation == "Found some rules"
        assert step.reflection is None
        assert isinstance(step.timestamp, float)
        assert isinstance(step.metadata, dict)
        
        # Test with all fields
        step = AgentStep(
            step_id="test_2",
            action="analyze",
            thought="Analyzing data",
            observation="Found patterns",
            reflection="This was useful",
            metadata={"key": "value"}
        )
        assert step.reflection == "This was useful"
        assert step.metadata["key"] == "value"
    
    def test_agent_plan_model(self):
        """Test AgentPlan model initialization and validation"""
        from app.agents.agent_runner import AgentPlan
        
        # Test basic initialization
        plan = AgentPlan(
            goal="Generate rules",
            steps=["step1", "step2"]
        )
        assert plan.goal == "Generate rules"
        assert len(plan.steps) == 2
        assert plan.current_step == 0
        assert isinstance(plan.context, dict)
        assert isinstance(plan.constraints, list)
        
        # Test with all fields
        plan = AgentPlan(
            goal="Complex analysis",
            steps=["analyze", "process", "validate"],
            current_step=1,
            context={"mode": "advanced"},
            constraints=["time_limit", "resource_limit"]
        )
        assert plan.current_step == 1
        assert plan.context["mode"] == "advanced"
        assert len(plan.constraints) == 2


class TestAgentReasoning:
    """Test the agent's reasoning capabilities"""
    
    def test_reason_before_action_fetch(self):
        """Test reasoning before fetch action"""
        from app.agents.agent_runner import reason_before_action
        from app.state.state import AgentState
        
        state = AgentState(
            data_schema={
                "domain": "customer",
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            thoughts=[],
            step_history=[]
        )
        
        result = reason_before_action(state, "fetch_rules")
        assert isinstance(result["thoughts"], list)
        assert len(result["thoughts"]) == 1
        assert "customer" in result["thoughts"][0]
        assert "2 columns" in result["thoughts"][0]
        assert len(result["step_history"]) == 1
        assert result["step_history"][0].action == "fetch_rules"
    
    def test_reason_before_action_suggest(self):
        """Test reasoning before suggest action"""
        from app.agents.agent_runner import reason_before_action
        from app.state.state import AgentState
        
        state = AgentState(
            data_schema={"col1": {"type": "string"}},
            thoughts=[],
            step_history=[],
            gx_rules=["rule1", "rule2"]
        )
        
        result = reason_before_action(state, "suggest")
        assert len(result["thoughts"]) == 1
        assert "2 rule templates" in result["thoughts"][0]
        assert result["current_step"] == "suggest"


class TestAgentReflection:
    """Test the agent's reflection capabilities"""
    
    def test_reflect_on_progress_early_stage(self):
        """Test reflection during early execution stage"""
        from app.agents.agent_runner import reflect_on_progress, AgentState, AgentPlan, AgentStep
        
        state = AgentState(
            data_schema={"col1": {"type": "string"}},
            reflections=[],
            step_history=[
                AgentStep(
                    step_id="step_1",
                    action="fetch",
                    thought="Initial step",
                    observation="Started"
                )
            ],
            plan=AgentPlan(
                goal="Generate rules",
                steps=["fetch", "analyze", "suggest", "validate"]
            ),
            errors=[],
            quality_metrics={}
        )
        
        result = reflect_on_progress(state)
        assert len(result["reflections"]) == 1
        assert "Early stage" in result["reflections"][0]
        assert result["quality_metrics"]["progress"] == 0.25
        assert result["quality_metrics"]["error_rate"] == 0
    
    def test_reflect_on_progress_with_errors(self):
        """Test reflection when errors are encountered"""
        from app.agents.agent_runner import reflect_on_progress, AgentState, AgentPlan
        
        state = AgentState(
            data_schema={"col1": {"type": "string"}},
            reflections=[],
            step_history=[],
            plan=AgentPlan(
                goal="Generate rules",
                steps=["fetch", "analyze", "suggest"]
            ),
            errors=["Error 1", "Error 2", "Error 3"],
            quality_metrics={}
        )
        
        result = reflect_on_progress(state)
        assert len(result["reflections"]) == 1
        assert "High error rate" in result["reflections"][0]
        assert result["quality_metrics"]["error_rate"] > 0


class TestAgentErrorHandling:
    """Test the agent's error handling capabilities"""
    
    def test_handle_agent_error(self):
        """Test error handling function"""
        from app.agents.agent_runner import _handle_agent_error
        from app.validation.validation_base import ValidationSeverity, ValidationIssue, ValidationResult
        from app.validation.metrics import ValidationMetricsCollector
        
        # Create a mock collector and set it as the global collector
        mock_collector = MagicMock(spec=ValidationMetricsCollector)
        
        with patch('app.validation.metrics._metrics_collector', mock_collector):
            start_time = time.time()
            _handle_agent_error(Exception("Test error"), "test_domain", start_time)
            
            # Verify collector's record_validation was called
            mock_collector.record_validation.assert_called_once()
            
            # Get the call arguments
            call_args = mock_collector.record_validation.call_args[0]
            
            # Check domain and response type
            assert call_args[0] == "test_domain"
            assert call_args[1] == "rule"
            
            # Check validation result
            validation_result = call_args[2]
            assert isinstance(validation_result, ValidationResult)
            assert not validation_result.is_valid
            assert len(validation_result.issues) == 1
            assert validation_result.issues[0].severity == ValidationSeverity.CRITICAL


class TestAgentReporting:
    """Test the agent's reporting capabilities"""
    
    def test_generate_agent_report_basic(self):
        """Test basic report generation"""
        from app.agents.agent_runner import generate_agent_report, AgentState
        
        state = AgentState(
            data_schema={
                "domain": "customer",
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            rule_suggestions=[{"rule": "test"}],
            thoughts=["Thought 1", "Thought 2"],
            observations=["Obs 1"],
            reflections=["Reflection 1"],
            execution_metrics={"total_execution_time": 1.5},
            quality_metrics={"accuracy": 0.95}
        )
        
        report = generate_agent_report(state)
        assert "# Enhanced Agent Execution Report" in report
        assert "**Domain**: customer" in report
        assert "**Columns Analyzed**: 2" in report
        assert "**Rules Generated**: 1" in report
        assert "**Execution Time**: 1.50s" in report
    
    def test_generate_agent_report_with_errors(self):
        """Test report generation with error information"""
        from app.agents.agent_runner import generate_agent_report, AgentState
        
        state = AgentState(
            data_schema={"col1": {"type": "string"}},
            errors=["Error 1", "Error 2"],
            thoughts=[],
            observations=[],
            reflections=[],
            execution_metrics={},
            quality_metrics={}
        )
        
        report = generate_agent_report(state)
        assert "##  Issues Encountered" in report
        assert "Error 1" in report
        assert "Error 2" in report


class TestAgentIntegration:
    """Test integration scenarios for the agent runner"""
    
    @patch('app.agents.agent_runner._setup_validation_context')
    @patch('app.agents.agent_runner._validate_with_enhanced_context')
    def test_run_agent_with_validation(self, mock_validate_context, mock_setup_validation):
        """Test agent execution with validation enabled"""
        from app.agents.agent_runner import run_agent
        from app.state.state import AgentState
        
        # Setup validation context mock
        mock_context = Mock()
        mock_context.__enter__ = Mock(return_value=mock_context)
        mock_context.__exit__ = Mock(return_value=False)
        mock_context.get_metrics.return_value = {"validation_count": 1}
        mock_setup_validation.return_value = mock_context
        
        # Mock the validation function to return the rule suggestions
        mock_validate_context.return_value = [{"rule": "test"}]
        
        # Run agent with basic schema
        with patch('app.agents.agent_runner.build_graph') as mock_build_graph:
            mock_graph = Mock()
            mock_result = AgentState(data_schema={"test": "schema"})
            mock_result.rule_suggestions = [{"rule": "test"}]
            mock_result.thoughts = ["thought1"]
            mock_result.observations = ["obs1"]
            mock_result.reflections = ["ref1"]
            mock_result.execution_metrics = {"total_execution_time": 1.0}
            mock_result.step_history = []
            mock_graph.invoke.return_value = mock_result
            mock_build_graph.return_value = mock_graph
            
            result = run_agent({"domain": "test", "col1": {"type": "string"}})
            
            assert len(result) == 1
            mock_validate_context.assert_called_once()
    
    @patch('app.agents.agent_runner._setup_validation_context')
    def test_run_agent_complex_workflow(self, mock_setup_validation):
        """Test agent with complex schema and workflow"""
        from app.agents.agent_runner import run_agent
        from app.state.state import AgentState
        
        # Mock validation context to return None (no validation)
        mock_setup_validation.return_value = None
        
        # Complex schema with various data types
        complex_schema = {
            "domain": "finance",
            "transaction_id": {"type": "string", "nullable": False},
            "amount": {"type": "float", "nullable": False},
            "date": {"type": "date", "nullable": False},
            "status": {"type": "string", "nullable": True},
            "customer_id": {"type": "integer", "nullable": False}
        }
        
        with patch('app.agents.agent_runner.build_graph') as mock_build_graph:
            # Setup complex mock workflow
            mock_graph = Mock()
            mock_result = AgentState(data_schema=complex_schema)
            mock_result.rule_suggestions = [
                {"rule": "rule1", "column": "transaction_id"},
                {"rule": "rule2", "column": "amount"},
                {"rule": "rule3", "column": "date"}
            ]
            mock_result.thoughts = ["thought1", "thought2", "thought3"]
            mock_result.observations = ["obs1", "obs2"]
            mock_result.reflections = ["ref1"]
            mock_result.execution_metrics = {
                "total_execution_time": 2.5,
                "steps_completed": 3
            }
            mock_result.step_history = []
            mock_graph.invoke.return_value = mock_result
            mock_build_graph.return_value = mock_graph
            
            result = run_agent(complex_schema)
            
            assert len(result) == 3
            assert any(r["column"] == "transaction_id" for r in result)
            assert any(r["column"] == "amount" for r in result)
            assert any(r["column"] == "date" for r in result)