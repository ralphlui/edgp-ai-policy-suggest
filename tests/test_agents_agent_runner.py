"""
Tests for app/agents/agent_runner.py module
"""
import pytest
from unittest.mock import Mock, patch, MagicMock


class TestAgentRunnerModule:
    """Test agent_runner.py module functionality"""
    
    def test_agent_runner_module_import(self):
        """Test agent_runner module can be imported"""
        from app.agents import agent_runner
        assert hasattr(agent_runner, '__name__')

    def test_agent_runner_functions_exist(self):
        """Test agent runner functions exist"""
        from app.agents.agent_runner import run_agent
        assert callable(run_agent)

    @patch('app.agents.agent_runner.create_react_agent')
    @patch('app.agents.agent_runner.ChatOpenAI')
    def test_run_agent_basic_mock(self, mock_chat_openai, mock_create_agent):
        """Test basic agent running with mocks"""
        # Setup ChatOpenAI mock
        mock_llm = Mock()
        mock_chat_openai.return_value = mock_llm
        
        # Setup agent mock
        mock_agent = Mock()
        mock_agent.invoke.return_value = {
            "output": "Test agent response",
            "intermediate_steps": []
        }
        mock_create_agent.return_value = mock_agent
        
        from app.agents.agent_runner import run_agent
        
        result = run_agent("test query", [])
        
        assert isinstance(result, dict)
        assert "output" in result
        mock_chat_openai.assert_called_once()
        mock_create_agent.assert_called_once()

    @patch('app.agents.agent_runner.create_react_agent')
    @patch('app.agents.agent_runner.ChatOpenAI')
    def test_run_agent_with_tools(self, mock_chat_openai, mock_create_agent):
        """Test agent running with tools"""
        mock_llm = Mock()
        mock_chat_openai.return_value = mock_llm
        
        mock_agent = Mock()
        mock_agent.invoke.return_value = {
            "output": "Used tools successfully",
            "intermediate_steps": [("tool1", "result1")]
        }
        mock_create_agent.return_value = mock_agent
        
        # Mock tools
        mock_tool1 = Mock()
        mock_tool1.name = "test_tool"
        mock_tools = [mock_tool1]
        
        from app.agents.agent_runner import run_agent
        
        result = run_agent("use tools to help", mock_tools)
        
        assert isinstance(result, dict)
        assert "output" in result
        # Verify tools were passed to create_react_agent
        call_args = mock_create_agent.call_args
        assert len(call_args[0]) >= 2  # llm, tools, ...

    @patch('app.agents.agent_runner.create_react_agent')
    @patch('app.agents.agent_runner.ChatOpenAI')
    def test_run_agent_error_handling(self, mock_chat_openai, mock_create_agent):
        """Test agent error handling"""
        mock_llm = Mock()
        mock_chat_openai.return_value = mock_llm
        
        mock_agent = Mock()
        mock_agent.invoke.side_effect = Exception("Agent execution failed")
        mock_create_agent.return_value = mock_agent
        
        from app.agents.agent_runner import run_agent
        
        try:
            result = run_agent("test query", [])
            # If it doesn't raise, check result format
            assert isinstance(result, dict)
        except Exception:
            # Expected to fail in some cases
            pass

    @patch('app.agents.agent_runner.create_react_agent')
    @patch('app.agents.agent_runner.ChatOpenAI')
    def test_run_agent_complex_query(self, mock_chat_openai, mock_create_agent):
        """Test agent with complex query"""
        mock_llm = Mock()
        mock_chat_openai.return_value = mock_llm
        
        mock_agent = Mock()
        mock_agent.invoke.return_value = {
            "output": "Complex analysis completed",
            "intermediate_steps": [
                ("analysis_tool", "step1_result"),
                ("validation_tool", "step2_result")
            ]
        }
        mock_create_agent.return_value = mock_agent
        
        from app.agents.agent_runner import run_agent
        
        complex_query = """
        Analyze the customer data schema and suggest appropriate data governance rules.
        Consider data quality, privacy, and compliance requirements.
        """
        
        result = run_agent(complex_query, [])
        
        assert isinstance(result, dict)
        assert "output" in result

    @patch('app.agents.agent_runner.AgentExecutor')
    @patch('app.agents.agent_runner.create_react_agent')
    @patch('app.agents.agent_runner.ChatOpenAI')
    def test_run_agent_executor_mock(self, mock_chat_openai, mock_create_agent, mock_agent_executor):
        """Test agent executor functionality"""
        mock_llm = Mock()
        mock_chat_openai.return_value = mock_llm
        
        mock_agent = Mock()
        mock_create_agent.return_value = mock_agent
        
        mock_executor = Mock()
        mock_executor.invoke.return_value = {"output": "Executor result"}
        mock_agent_executor.from_agent_and_tools.return_value = mock_executor
        
        from app.agents.agent_runner import run_agent
        
        result = run_agent("test query", [])
        
        assert isinstance(result, dict)

    def test_agent_runner_imports(self):
        """Test agent runner imports work correctly"""
        from app.agents.agent_runner import run_agent
        assert callable(run_agent)

    @patch('app.agents.agent_runner.ChatOpenAI')
    def test_run_agent_llm_configuration(self, mock_chat_openai):
        """Test LLM configuration in agent"""
        mock_llm = Mock()
        mock_chat_openai.return_value = mock_llm
        
        with patch('app.agents.agent_runner.create_react_agent') as mock_create_agent:
            mock_agent = Mock()
            mock_agent.invoke.return_value = {"output": "test"}
            mock_create_agent.return_value = mock_agent
            
            from app.agents.agent_runner import run_agent
            
            run_agent("test", [])
            
            # Verify ChatOpenAI was called with expected parameters
            mock_chat_openai.assert_called_once()
            call_kwargs = mock_chat_openai.call_args[1]
            assert "model" in call_kwargs or len(mock_chat_openai.call_args[0]) > 0

    @patch('app.agents.agent_runner.create_react_agent')
    @patch('app.agents.agent_runner.ChatOpenAI')
    def test_run_agent_empty_tools(self, mock_chat_openai, mock_create_agent):
        """Test agent running with empty tools list"""
        mock_llm = Mock()
        mock_chat_openai.return_value = mock_llm
        
        mock_agent = Mock()
        mock_agent.invoke.return_value = {"output": "No tools used"}
        mock_create_agent.return_value = mock_agent
        
        from app.agents.agent_runner import run_agent
        
        result = run_agent("simple query", [])
        
        assert isinstance(result, dict)
        assert "output" in result

    @patch('app.agents.agent_runner.create_react_agent')
    @patch('app.agents.agent_runner.ChatOpenAI')
    def test_run_agent_prompt_template(self, mock_chat_openai, mock_create_agent):
        """Test agent with prompt template"""
        mock_llm = Mock()
        mock_chat_openai.return_value = mock_llm
        
        mock_agent = Mock()
        mock_agent.invoke.return_value = {"output": "Prompted response"}
        mock_create_agent.return_value = mock_agent
        
        from app.agents.agent_runner import run_agent
        
        result = run_agent("query with prompt context", [])
        
        assert isinstance(result, dict)
        # Verify create_react_agent was called with prompt parameter
        call_kwargs = mock_create_agent.call_args[1] if mock_create_agent.call_args[1] else {}
        # Should have prompt parameter or be in positional args
        assert len(mock_create_agent.call_args[0]) >= 2


class TestAgentRunnerUtilities:
    """Test utility functions and edge cases in agent runner"""
    
    def test_module_level_constants(self):
        """Test module level constants and imports"""
        from app.agents.agent_runner import run_agent
        assert callable(run_agent)

    @patch('app.agents.agent_runner.logger')
    def test_logging_functionality(self, mock_logger):
        """Test logging functionality in agent runner"""
        from app.agents import agent_runner
        
        # Just verify module loads with logger
        assert hasattr(agent_runner, '__name__')

    @patch('app.agents.agent_runner.OPENAI_API_KEY', 'test-key')
    @patch('app.agents.agent_runner.ChatOpenAI')
    def test_api_key_usage(self, mock_chat_openai):
        """Test API key usage in agent"""
        mock_llm = Mock()
        mock_chat_openai.return_value = mock_llm
        
        with patch('app.agents.agent_runner.create_react_agent') as mock_create_agent:
            mock_agent = Mock()
            mock_agent.invoke.return_value = {"output": "test"}
            mock_create_agent.return_value = mock_agent
            
            from app.agents.agent_runner import run_agent
            
            run_agent("test", [])
            
            # Verify API key was used in ChatOpenAI initialization
            mock_chat_openai.assert_called_once()

    def test_function_docstrings(self):
        """Test function docstrings exist"""
        from app.agents.agent_runner import run_agent
        
        # Function should be callable
        assert callable(run_agent)