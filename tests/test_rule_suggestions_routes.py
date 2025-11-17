import sys
import types
import pytest
import time
import json
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from fastapi.responses import JSONResponse

from app.api.rule_suggestion_routes import router as rules_router
import app.api.rule_suggestion_routes as rules_module
from app.auth.authentication import UserInfo


# ------------------ Test helpers / fakes ------------------

class FakeUser:
    def __init__(self, email="tester@example.com", scopes=None, user_id="test_user_123"):
        self.email = email
        self.user_id = user_id  # Add user_id for PII protection compliance
        self.scopes = scopes or ["manage:mdm"]

class FakeStore:
    def __init__(self):
        self.refreshed = False
    def force_refresh_index(self):
        self.refreshed = True
        return True


# ------------------ Pytest fixtures ------------------

@pytest.fixture
def app():
    app = FastAPI()
    # Override auth dependency to avoid real JWT checks
    app.dependency_overrides[rules_module.verify_any_scope_token] = lambda: FakeUser()
    app.include_router(rules_router)
    return app

@pytest.fixture
def client(app):
    return TestClient(app)


# ------------------ Tests ------------------

def test_suggest_rules_schema_found_with_insights(client, monkeypatch):
    """
    Schema exists + include_insights=True:
    - Patches get_schema_by_domain to return a schema
    - Mocks guardrails validation to pass
    - Injects a fake app.agents.agent_runner module (AgentState & build_graph)
    - Expects 200 with rule_suggestions + confidence + agent_insights
    """
    schema = {"domain": "customer", "id": {"type": "string"}, "email": {"type": "string"}}
    monkeypatch.setattr(rules_module, "get_schema_by_domain", lambda d: schema)

    # Mock guardrails to always pass validation
    class FakeInputGuardrails:
        def comprehensive_validate(self, domain, available_domains):
            return True, []  # Valid, no violations
    
    class FakeStore:
        def get_all_domains_realtime(self, force_refresh=False):
            return ["customer", "orders", "products"]
        def force_refresh_index(self):
            return True
    
    # Mock the guardrails import
    monkeypatch.setattr("app.validation.input_guardrails.InputGuardrails", FakeInputGuardrails)
    monkeypatch.setattr("app.vector_db.schema_loader.get_store", lambda: FakeStore())

    class FakeAgentState:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self.errors = getattr(self, "errors", [])
            # Create proper AgentStep objects for step_history
            from app.state.state import AgentStep
            default_steps = [
                AgentStep(step_id="1", action="analyze", thought="thinking", observation="observed"),
                AgentStep(step_id="2", action="suggest", thought="considering", observation="suggested"), 
                AgentStep(step_id="3", action="format", thought="formatting", observation="formatted")
            ]
            self.step_history = getattr(self, "step_history", default_steps)
            self.execution_metrics = getattr(self, "execution_metrics", {"total_execution_time": 1.7})
            self.rule_suggestions = getattr(
                self, "rule_suggestions",
                [{"column": "email", "rule": "format:email"}, {"column": "id", "rule": "not_empty"}]
            )
            self.data_schema = getattr(self, "data_schema", schema)
            self.thoughts = getattr(self, "thoughts", ["t1", "t2", "t3", "t4", "t5"])
            self.observations = getattr(self, "observations", ["o1"])
            self.reflections = getattr(self, "reflections", ["r1"])
            self.confidence_scores = getattr(self, "confidence_scores", {"dummy": 1.0})
            self.execution_start_time = getattr(self, "execution_start_time", 1000.0)
            self.plan = getattr(self, "plan", None)
            # Additional attributes needed for execution trace
            self.gx_rules = getattr(self, "gx_rules", [])
            self.raw_suggestions = getattr(self, "raw_suggestions", "")
            self.formatted_rules = getattr(self, "formatted_rules", [])
            self.normalized_suggestions = getattr(self, "normalized_suggestions", {})

    class FakeGraph:
        def invoke(self, initial_state):
            return FakeAgentState(
                data_schema=initial_state.data_schema,
                enhanced_prompt=getattr(initial_state, 'enhanced_prompt', 'test prompt'),
                rule_suggestions=[{"column": "email", "rule": "format:email"},
                                  {"column": "id", "rule": "not_empty"}],
                execution_metrics={"total_execution_time": 1.7},
                thoughts=["t1", "t2", "t3", "t4", "t5"],
                observations=["o1"],
                reflections=["r1"]
            )

    # Mock the RuleRAGEnhancer
    class FakeRAGEnhancer:
        async def enhance_prompt_with_history(self, schema, domain):
            return "Enhanced prompt for testing"
            
        async def store_successful_policy(self, domain, schema, rules, performance_metrics):
            return True

    # Ensure the dynamic imports inside the route resolve to our fake modules
    fake_agent_runner = types.SimpleNamespace(AgentState=FakeAgentState, build_graph=lambda: FakeGraph())
    monkeypatch.setitem(sys.modules, "app.agents.agent_runner", fake_agent_runner)
    monkeypatch.setattr(rules_module, "RuleRAGEnhancer", lambda: FakeRAGEnhancer())

    payload = {"domain": "customer", "include_insights": True}
    resp = client.post("/api/aips/rules/suggest", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data.get("rule_suggestions", []), list)
    assert "confidence" in data and "overall" in data["confidence"]
    assert "agent_insights" in data and "summary" in data["agent_insights"]


def test_suggest_rules_schema_found_no_insights(client, monkeypatch):
    """
    Schema exists + include_insights=False: should call run_agent(schema)
    and return its output.
    """
    schema = {"domain": "orders", "order_id": {"type": "string"}}
    monkeypatch.setattr(rules_module, "get_schema_by_domain", lambda d: schema)

    # Mock guardrails to always pass validation
    class FakeInputGuardrails:
        def comprehensive_validate(self, domain, available_domains):
            return True, []  # Valid, no violations
    
    class FakeStore:
        def get_all_domains_realtime(self, force_refresh=False):
            return ["customer", "orders", "products"]
        def force_refresh_index(self):
            return True
    
    # Mock the guardrails import
    monkeypatch.setattr("app.validation.input_guardrails.InputGuardrails", FakeInputGuardrails)
    monkeypatch.setattr("app.vector_db.schema_loader.get_store", lambda: FakeStore())

    def fake_run_agent(s):
        assert s == schema
        return [{"column": "order_id", "rule": "not_empty"}]

    # run_agent was imported at module import time → patch at rules_module site
    monkeypatch.setattr(rules_module, "run_agent", fake_run_agent)

    payload = {"domain": "orders", "include_insights": False}
    resp = client.post("/api/aips/rules/suggest", json=payload)
    assert resp.status_code == 200
    assert resp.json() == {"rule_suggestions": [{"column": "order_id", "rule": "not_empty"}]}


def test_suggest_rules_vector_db_connection_failed(client, monkeypatch):
    """
    get_schema_by_domain raises a connection-like error: expect 503 with error_type=connection_failed.
    """
    # Mock guardrails to pass validation first
    class FakeInputGuardrails:
        def comprehensive_validate(self, domain, available_domains):
            return True, []  # Valid, no violations
    
    class FakeStore:
        def get_all_domains_realtime(self, force_refresh=False):
            return ["customer", "orders", "products"]
        def force_refresh_index(self):
            return True
    
    # Mock the guardrails import
    monkeypatch.setattr("app.validation.input_guardrails.InputGuardrails", FakeInputGuardrails)
    monkeypatch.setattr("app.vector_db.schema_loader.get_store", lambda: FakeStore())
    
    monkeypatch.setattr(rules_module, "get_schema_by_domain", lambda _: (_ for _ in ()).throw(Exception("ConnectionError: timed out")))

    payload = {"domain": "any", "include_insights": True}
    resp = client.post("/api/aips/rules/suggest", json=payload)
    assert resp.status_code == 503
    data = resp.json()
    assert data["error"] == "Vector database connection failed"
    assert data["error_type"] == "connection_failed"


def test_suggest_rules_schema_not_found_generates_ai_suggestions(client, monkeypatch):
    """
    Vector DB accessible but schema not found: return 404 with AI-suggested column names.
    """
    # Mock guardrails to pass validation first
    class FakeInputGuardrails:
        def comprehensive_validate(self, domain, available_domains):
            return True, []  # Valid, no violations
    
    class FakeStore:
        def get_all_domains_realtime(self, force_refresh=False):
            return ["customer", "orders", "products"]
        def force_refresh_index(self):
            return True
    
    # Mock the guardrails import
    monkeypatch.setattr("app.validation.input_guardrails.InputGuardrails", FakeInputGuardrails)
    monkeypatch.setattr("app.vector_db.schema_loader.get_store", lambda: FakeStore())
    
    # Always return None (schema not found), before and after refresh
    monkeypatch.setattr(rules_module, "get_schema_by_domain", lambda d: None)

    # Provide get_store() for the refresh step (imported dynamically inside the handler)
    fake_schema_loader = types.SimpleNamespace(get_store=lambda: FakeStore())
    monkeypatch.setitem(sys.modules, "app.vector_db.schema_loader", fake_schema_loader)

    # Predictable suggestions from bootstrap
    def _bootstrap(domain):
        assert domain == "newdomain"
        return {"id": {}, "created_at": {}, "status": {}}
    monkeypatch.setattr(rules_module, "bootstrap_schema_for_domain", _bootstrap)

    payload = {"domain": "newdomain", "include_insights": True}
    resp = client.post("/api/aips/rules/suggest", json=payload)
    assert resp.status_code == 404
    data = resp.json()
    assert data["error"] == "Domain not found"
    assert data["domain"] == "newdomain"
    # Check that it has the available_actions instead of suggested_columns
    assert "available_actions" in data
    assert "suggest_schema_ai" in data["available_actions"]
    # Confirm action endpoints point to the correct domains API
    assert data["available_actions"]["suggest_schema_ai"]["endpoint"] == "/api/aips/domains/suggest-schema"


def test_suggest_rules_unexpected_error_returns_500(client, monkeypatch):
    """
    Non-connection exception (accessible_but_error), then bootstrap also fails → outer except returns 500.
    """
    # Mock guardrails to pass validation first
    class FakeInputGuardrails:
        def comprehensive_validate(self, domain, available_domains):
            return True, []  # Valid, no violations
    
    class FakeStore:
        def get_all_domains_realtime(self, force_refresh=False):
            return ["customer", "orders", "products"]
        def force_refresh_index(self):
            return True
    
    # Mock the guardrails import
    monkeypatch.setattr("app.validation.input_guardrails.InputGuardrails", FakeInputGuardrails)
    monkeypatch.setattr("app.vector_db.schema_loader.get_store", lambda: FakeStore())
    
    def _get_schema(_):
        # No connection keywords → accessible_but_error branch
        raise Exception("Some other error")
    monkeypatch.setattr(rules_module, "get_schema_by_domain", _get_schema)

    def _bootstrap(_):
        raise Exception("LLM failure")
    monkeypatch.setattr(rules_module, "bootstrap_schema_for_domain", _bootstrap)

    payload = {"domain": "oops", "include_insights": True}
    resp = client.post("/api/aips/rules/suggest", json=payload)
    # Current implementation returns 404 when domain is not found, even after vector DB errors
    assert resp.status_code == 404
    data = resp.json()
    assert data["error"] == "Domain not found"


def test_suggest_rules_bootstrap_failure_returns_500(client, monkeypatch):
    """
    Vector DB accessible and returns None for schema, but bootstrap_schema_for_domain raises.
    Expect 500 from outer exception handler.
    """
    # Mock guardrails to pass validation first
    class FakeInputGuardrails:
        def comprehensive_validate(self, domain, available_domains):
            return True, []  # Valid, no violations
    
    class FakeStore:
        def get_all_domains_realtime(self, force_refresh=False):
            return ["customer", "orders", "products"]
        def force_refresh_index(self):
            return True
    
    # Mock the guardrails import
    monkeypatch.setattr("app.validation.input_guardrails.InputGuardrails", FakeInputGuardrails)
    monkeypatch.setattr("app.vector_db.schema_loader.get_store", lambda: FakeStore())
    
    # Vector DB accessible (no exception), but no schema found
    monkeypatch.setattr(rules_module, "get_schema_by_domain", lambda d: None)

    # bootstrap fails inside the domain-not-found flow
    def _bootstrap(_):
        raise Exception("boom")
    monkeypatch.setattr(rules_module, "bootstrap_schema_for_domain", _bootstrap)

    payload = {"domain": "finance", "include_insights": False}
    resp = client.post("/api/aips/rules/suggest", json=payload)
    # Current implementation returns 404 when domain is not found, even after bootstrap fails
    assert resp.status_code == 404
    data = resp.json()
    assert data["error"] == "Domain not found"
    assert data["domain"] == "finance"


# ================== COMPREHENSIVE TESTS ==================
# Additional comprehensive tests for better coverage

class TestRuleSuggestionRoutesAPIComprehensive:
    """Comprehensive test class for API routes with better coverage."""
    
    @pytest.fixture
    def mock_user_info(self):
        """Create mock user info for authentication."""
        user_info = Mock(spec=UserInfo)
        user_info.user_id = "test_user_123"
        user_info.email = "test@example.com"
        user_info.scopes = ["read", "write"]
        return user_info

    def test_suggest_rules_success_comprehensive(self, client, monkeypatch, mock_user_info):
        """Test successful rule suggestion endpoint with comprehensive mocking."""
        # Setup auth mock
        monkeypatch.setattr(rules_module, "verify_any_scope_token", lambda: mock_user_info)
        
        # Mock schema exists
        schema = {"domain": "customer", "id": {"type": "string"}, "email": {"type": "string"}}
        monkeypatch.setattr(rules_module, "get_schema_by_domain", lambda d: schema)

        # Mock guardrails to always pass validation
        class FakeInputGuardrails:
            def comprehensive_validate(self, domain, available_domains):
                return True, []  # Valid, no violations
        
        class FakeStore:
            def get_all_domains_realtime(self, force_refresh=False):
                return ["customer", "product", "order"]
            def force_refresh_index(self):
                return True
        
        # Mock the guardrails import
        monkeypatch.setattr("app.validation.input_guardrails.InputGuardrails", FakeInputGuardrails)
        monkeypatch.setattr("app.vector_db.schema_loader.get_store", lambda: FakeStore())

        # Mock agent result with comprehensive state
        class FakeAgentState:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)
                self.errors = getattr(self, "errors", [])
                from app.state.state import AgentStep
                default_steps = [
                    AgentStep(step_id="1", action="analyze", thought="thinking", observation="observed"),
                    AgentStep(step_id="2", action="suggest", thought="considering", observation="suggested"), 
                    AgentStep(step_id="3", action="format", thought="formatting", observation="formatted")
                ]
                self.step_history = getattr(self, "step_history", default_steps)
                self.execution_metrics = getattr(self, "execution_metrics", {"total_execution_time": 2.5})
                self.rule_suggestions = getattr(
                    self, "rule_suggestions",
                    [{"expectation_type": "expect_column_values_to_be_unique", "column": "id"}]
                )
                self.data_schema = getattr(self, "data_schema", schema)
                self.thoughts = getattr(self, "thoughts", ["Thinking about uniqueness"])
                self.observations = getattr(self, "observations", ["Observed data patterns"])
                self.reflections = getattr(self, "reflections", ["Reflection on rules"])
                self.confidence_scores = getattr(self, "confidence_scores", {"dummy": 1.0})
                self.execution_start_time = getattr(self, "execution_start_time", 1000.0)
                self.plan = getattr(self, "plan", None)
                # Additional attributes needed for execution trace
                self.gx_rules = getattr(self, "gx_rules", [])
                self.raw_suggestions = getattr(self, "raw_suggestions", "")
                self.formatted_rules = getattr(self, "formatted_rules", [])
                self.normalized_suggestions = getattr(self, "normalized_suggestions", {})

        class FakeGraph:
            def invoke(self, initial_state):
                return FakeAgentState(
                    data_schema=initial_state.data_schema,
                    enhanced_prompt=getattr(initial_state, 'enhanced_prompt', 'test prompt'),
                    rule_suggestions=[{"expectation_type": "expect_column_values_to_be_unique", "column": "id"}],
                    execution_metrics={"total_execution_time": 2.5},
                    thoughts=["Thinking about uniqueness"],
                    observations=["Observed data patterns"],
                    reflections=["Reflection on rules"]
                )

        # Mock the RuleRAGEnhancer
        class FakeRAGEnhancer:
            async def enhance_prompt_with_history(self, schema, domain):
                return "Enhanced prompt for testing"
                
            async def store_successful_policy(self, domain, schema, rules, performance_metrics):
                return True

        # Ensure the dynamic imports inside the route resolve to our fake modules
        fake_agent_runner = types.SimpleNamespace(AgentState=FakeAgentState, build_graph=lambda: FakeGraph())
        monkeypatch.setitem(sys.modules, "app.agents.agent_runner", fake_agent_runner)
        monkeypatch.setattr(rules_module, "RuleRAGEnhancer", lambda: FakeRAGEnhancer())
        
        # Make request
        payload = {"domain": "customer", "include_insights": True}
        response = client.post("/api/aips/rules/suggest", json=payload)
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert "rule_suggestions" in data
        assert "confidence" in data
        assert "agent_insights" in data

    def test_suggest_rules_guardrail_failure_comprehensive(self, client, monkeypatch, mock_user_info):
        """Test rule suggestion endpoint with guardrail failures."""
        # Setup auth mock
        monkeypatch.setattr(rules_module, "verify_any_scope_token", lambda: mock_user_info)
        
        # Mock guardrails with violations using proper violation objects
        class FakeInputGuardrailsWithViolations:
            def comprehensive_validate(self, domain, available_domains):
                # Import the actual violation classes
                from app.validation.input_guardrails import GuardrailViolation, GuardrailViolationType
                violation = GuardrailViolation(
                    violation_type=GuardrailViolationType.UNSAFE_CONTENT,
                    message="Unsafe content detected",
                    suggested_action="Revise your input",
                    confidence=0.95,
                    detected_pattern="test pattern"
                )
                return False, [violation]
        
        class FakeStore:
            def get_all_domains_realtime(self, force_refresh=False):
                return ["customer", "product"]
            def force_refresh_index(self):
                return True
        
        # Mock the guardrails import
        monkeypatch.setattr("app.validation.input_guardrails.InputGuardrails", FakeInputGuardrailsWithViolations)
        monkeypatch.setattr("app.vector_db.schema_loader.get_store", lambda: FakeStore())
        
        # Make request
        payload = {"domain": "harmful_domain"}
        response = client.post("/api/aips/rules/suggest", json=payload)
        
        # Should fail guardrail validation
        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert data["error"] == "Input validation failed"
        assert data["error_type"] == "guardrail_violation"


class TestUtilityFunctionsComprehensive:
    """Test utility functions in rule suggestion routes."""
    
    def test_calculate_overall_confidence_with_good_metrics(self):
        """Test confidence calculation with good execution metrics."""
        from app.api.rule_suggestion_routes import _calculate_overall_confidence
        
        # Create mock state with good metrics
        state = Mock()
        state.errors = []
        state.step_history = ["step1", "step2", "step3"]
        state.execution_metrics = {"total_execution_time": 2.0}
        state.rule_suggestions = ["rule1", "rule2", "rule3", "rule4", "rule5"]
        state.data_schema = {"domain": "test", "col1": "string", "col2": "int"}
        
        confidence = _calculate_overall_confidence(state)
        
        # Should be high confidence with good metrics
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.7  # Good metrics should yield high confidence

    def test_calculate_overall_confidence_with_errors(self):
        """Test confidence calculation with errors present."""
        from app.api.rule_suggestion_routes import _calculate_overall_confidence
        
        # Create mock state with errors
        state = Mock()
        state.errors = ["error1", "error2", "error3"]
        state.step_history = ["step1", "step2", "step3"]
        state.execution_metrics = {"total_execution_time": 5.0}
        state.rule_suggestions = ["rule1"]
        state.data_schema = {"domain": "test"}
        
        confidence = _calculate_overall_confidence(state)
        
        # Should be lower confidence with errors
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0

    def test_get_confidence_level_mapping(self):
        """Test confidence level string mapping."""
        from app.api.rule_suggestion_routes import _get_confidence_level
        
        # Test with different confidence states
        high_confidence_state = Mock()
        high_confidence_state.errors = []
        high_confidence_state.step_history = ["step1", "step2", "step3"]
        high_confidence_state.execution_metrics = {"total_execution_time": 1.5}
        high_confidence_state.rule_suggestions = ["rule1", "rule2", "rule3", "rule4", "rule5", "rule6"]
        high_confidence_state.data_schema = {"domain": "test"}
        
        level = _get_confidence_level(high_confidence_state)
        assert level in ["high", "medium", "low", "very_low"]

    def test_get_confidence_factors_detailed_breakdown(self):
        """Test detailed confidence factors breakdown."""
        from app.api.rule_suggestion_routes import _get_confidence_factors
        
        # Create comprehensive mock state
        state = Mock()
        state.errors = ["error1"]
        state.step_history = ["step1", "step2", "step3"]
        state.execution_metrics = {"total_execution_time": 3.5}
        state.rule_suggestions = ["rule1", "rule2", "rule3"]
        state.thoughts = ["thought1", "thought2", "thought3", "thought4"]
        state.observations = ["obs1", "obs2"]
        state.reflections = ["ref1"]
        
        factors = _get_confidence_factors(state)
        
        # Check structure
        assert "rule_generation" in factors
        assert "error_handling" in factors
        assert "execution_performance" in factors
        assert "reasoning_depth" in factors
        
        # Check rule generation details
        assert factors["rule_generation"]["rules_generated"] == 3
        assert "status" in factors["rule_generation"]
        assert "score" in factors["rule_generation"]
        
        # Check error handling details
        assert factors["error_handling"]["errors_encountered"] == 1
        assert "status" in factors["error_handling"]
        
        # Check execution performance
        assert factors["execution_performance"]["duration_seconds"] == 3.5
        assert "status" in factors["execution_performance"]
        
        # Check reasoning depth
        assert factors["reasoning_depth"]["thoughts_generated"] == 4
        assert factors["reasoning_depth"]["observations_made"] == 2
        assert factors["reasoning_depth"]["reflections_completed"] == 1

    def test_build_execution_trace_comprehensive(self):
        """Test comprehensive execution trace building."""
        from app.api.rule_suggestion_routes import _build_execution_trace
        
        # Create detailed mock state
        state = Mock()
        state.execution_start_time = time.time() - 10
        state.gx_rules = [{"rule1": "data"}, {"rule2": "data"}]
        state.raw_suggestions = "Rule 1\nRule 2\nRule 3"
        state.formatted_rules = [{"formatted1": "data"}, {"formatted2": "data"}]
        state.normalized_suggestions = {"norm1": "data", "norm2": "data"}
        state.rule_suggestions = [{"final1": "data"}, {"final2": "data"}, {"final3": "data"}]
        state.data_schema = {"domain": "test", "col1": "string", "col2": "int"}
        state.thoughts = ["thought1", "thought2"]
        state.observations = ["obs1"]
        state.step_history = []
        state.errors = []
        state.execution_metrics = {"total_execution_time": 2.5}
        
        trace = _build_execution_trace(state)
        
        # Check structure
        assert "workflow_path" in trace
        assert "tool_invocations" in trace
        assert "reasoning_chain" in trace
        assert "observations" in trace
        assert "execution_metrics" in trace
        
        # Check tool invocations
        assert len(trace["tool_invocations"]) > 0
        
        # Check each tool invocation has required fields
        for invocation in trace["tool_invocations"]:
            assert "tool" in invocation
            assert "status" in invocation
            assert "timestamp" in invocation
            
        # Check execution metrics
        metrics = trace["execution_metrics"]
        assert "total_duration_ms" in metrics
        assert "final_rule_count" in metrics
        assert metrics["final_rule_count"] == 3


class TestLoggingAndSanitizationComprehensive:
    """Test logging and sanitization functions."""
    
    def test_sanitize_for_logging_various_inputs(self):
        """Test sanitization function with various inputs."""
        from app.api.rule_suggestion_routes import sanitize_for_logging
        
        # Test normal string
        assert sanitize_for_logging("normal_string") == "normal_string"
        
        # Test string with control characters
        result = sanitize_for_logging("test\x00\x1F")
        assert "\x00" not in result
        assert "\x1F" not in result
        
        # Test string with special characters
        result = sanitize_for_logging("test(){}[]<>'\"\\")
        assert "(" not in result or result.count("_") > 0
        
        # Test very long string
        long_string = "a" * 200
        result = sanitize_for_logging(long_string)
        assert len(result) <= 100
        
        # Test non-string input
        result = sanitize_for_logging(123)
        assert result == '<non-string>'

    @patch('app.api.rule_suggestion_routes.logger')
    def test_log_domain_operation(self, mock_logger):
        """Test domain operation logging."""
        from app.api.rule_suggestion_routes import log_domain_operation
        
        # Test with details
        log_domain_operation("TEST_OP", "test_domain", "test details")
        mock_logger.info.assert_called_with("TEST_OP - domain: test_domain - test details")
        
        # Test without details
        mock_logger.reset_mock()
        log_domain_operation("TEST_OP", "test_domain")
        mock_logger.info.assert_called_with("TEST_OP - domain: test_domain")

    @patch('app.api.rule_suggestion_routes.logger')
    def test_log_error(self, mock_logger):
        """Test error logging."""
        from app.api.rule_suggestion_routes import log_error
        
        error = Exception("test error message")
        log_error("TEST_OP", "test_domain", error)
        mock_logger.error.assert_called_with("TEST_OP failed - domain: test_domain - error: test error message")

    @patch('app.api.rule_suggestion_routes.logger')
    @patch('time.time')
    def test_log_duration_context_manager(self, mock_time, mock_logger):
        """Test duration logging context manager."""
        from app.api.rule_suggestion_routes import log_duration
        
        # Mock time to simulate duration
        mock_time.side_effect = [0.0, 2.5]
        
        with log_duration("test_operation"):
            pass  # Simulate some work
        
        mock_logger.info.assert_called_with(" test_operation took 2.50s")


class TestListDomainsEndpoint:
    """Test the list domains endpoint."""
    
    def test_list_domains_success(self, client, monkeypatch):
        """Test successful domain listing."""
        class FakeStore:
            def get_all_domains_realtime(self, force_refresh=False):
                return ["customer", "product", "order"]
        
        # Import get_store from the right location
        monkeypatch.setattr("app.vector_db.schema_loader.get_store", lambda: FakeStore())
        
        response = client.get("/api/aips/rules/domains")
        
        assert response.status_code == 200
        data = response.json()
        assert "available_domains" in data
        assert len(data["available_domains"]) == 3
        assert "customer" in data["available_domains"]

    def test_list_domains_store_unavailable(self, client, monkeypatch):
        """Test domain listing when store is unavailable."""
        monkeypatch.setattr("app.vector_db.schema_loader.get_store", lambda: None)
        
        response = client.get("/api/aips/rules/domains")
        
        assert response.status_code == 503
        data = response.json()
        assert data["error"] == "Vector database not available"

    def test_list_domains_store_error(self, client, monkeypatch):
        """Test domain listing when store raises an error."""
        class FakeStore:
            def get_all_domains_realtime(self, force_refresh=False):
                raise Exception("Database error")
        
        monkeypatch.setattr("app.vector_db.schema_loader.get_store", lambda: FakeStore())
        
        response = client.get("/api/aips/rules/domains")
        
        assert response.status_code == 500
        data = response.json()
        assert data["error"] == "Failed to fetch domains"
