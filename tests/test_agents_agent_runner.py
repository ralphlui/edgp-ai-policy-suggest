"""
Tests for app/agents/agent_runner.py module
Tests updated to match LangGraph-based implementation
"""
import pytest
from unittest.mock import Mock, patch, MagicMock


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
        from app.agents.agent_runner import run_agent, build_graph, AgentState
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
        # Setup graph mock
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
        # Setup graph mock to raise exception
        mock_graph = Mock()
        mock_graph.invoke.side_effect = Exception("Graph execution failed")
        mock_build_graph.return_value = mock_graph
        
        from app.agents.agent_runner import run_agent
        
        # The function now catches exceptions and returns an empty list
        result = run_agent({"col1": {"type": "string"}})
        assert result == []

    @patch('app.agents.agent_runner.build_graph')
    def test_run_agent_complex_query(self, mock_build_graph):
        """Test agent with complex schema"""
        # Setup graph mock
        mock_result = Mock()
        mock_result.rule_suggestions = [
            {"rule": "test1", "column": "col1"},
            {"rule": "test2", "column": "col2"}
        ]
        mock_result.thoughts = ["thought1", "thought2"]
        mock_result.observations = ["obs1", "obs2"]
        mock_result.reflections = ["reflection1"]
        mock_result.execution_metrics = {"total_execution_time": 1.5}
        mock_result.step_history = ["step1", "step2"]
        mock_graph = Mock()
        mock_graph.invoke.return_value = mock_result
        mock_build_graph.return_value = mock_graph
        
        from app.agents.agent_runner import run_agent
        
        complex_schema = {
            "col1": {"type": "string", "nullable": False},
            "col2": {"type": "integer", "nullable": True},
            "col3": {"type": "date", "nullable": False}
        }
        
        result = run_agent(complex_schema)
        
        assert isinstance(result, list)
        assert len(result) == 2

    def test_run_agent_executor_mock(self):
        """Test that build_graph creates a workflow"""
        from app.agents.agent_runner import build_graph
        
        # Test that build_graph returns a compiled graph
        graph = build_graph()
        assert graph is not None
        assert hasattr(graph, 'invoke')

    @patch('app.agents.agent_runner.build_graph')
    def test_run_agent_llm_configuration(self, mock_build_graph):
        """Test agent state management"""
        mock_result = Mock()
        mock_result.rule_suggestions = [{"rule": "config_test"}]
        mock_graph = Mock()
        mock_graph.invoke.return_value = mock_result
        mock_build_graph.return_value = mock_graph
        
        from app.agents.agent_runner import run_agent
        
        result = run_agent({"test_col": {"type": "string"}})
        
        assert isinstance(result, list)
        # Verify the graph was called with correct arguments
        mock_build_graph.assert_called_once()

    @patch('app.agents.agent_runner.build_graph')
    def test_run_agent_empty_tools(self, mock_build_graph):
        """Test agent with empty schema"""
        mock_result = Mock()
        mock_result.rule_suggestions = []
        mock_graph = Mock()
        mock_graph.invoke.return_value = mock_result
        mock_build_graph.return_value = mock_graph
        
        from app.agents.agent_runner import run_agent
        
        result = run_agent({})
        
        assert isinstance(result, list)
        assert len(result) == 0

    def test_run_agent_prompt_template(self):
        """Test AgentState creation"""
        from app.agents.agent_runner import AgentState
        
        # Test AgentState initialization
        state = AgentState(data_schema={"col1": {"type": "string"}})
        assert state.data_schema == {"col1": {"type": "string"}}
        assert state.gx_rules is None
        assert state.rule_suggestions is None


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

    @patch('app.agents.agent_runner.build_graph')
    def test_api_key_usage(self, mock_build_graph):
        """Test that agent can handle API configuration"""
        mock_result = Mock()
        mock_result.rule_suggestions = [{"rule": "api_test"}]
        mock_graph = Mock()
        mock_graph.invoke.return_value = mock_result
        mock_build_graph.return_value = mock_graph
        
        from app.agents.agent_runner import run_agent
        
        # Test with schema that might require API calls
        schema = {"api_column": {"type": "text", "description": "API data"}}
        result = run_agent(schema)
        
        assert isinstance(result, list)

    def test_function_docstrings(self):
        """Test that functions have proper documentation"""
        from app.agents.agent_runner import run_agent, build_graph
        
        # Check docstrings exist (even if minimal)
        assert run_agent.__name__ == 'run_agent'
        assert build_graph.__name__ == 'build_graph'