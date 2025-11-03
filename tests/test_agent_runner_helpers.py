from types import SimpleNamespace
from app.agents import agent_runner as ar
from app.state.state import AgentState


def make_state(**overrides):
    base = {
        "data_schema": {"domain": "test", "a": {}, "b": {}},
        "gx_rules": [{"rule_name": "ExpectX"}],
        "thoughts": [],
        "observations": [],
        "reflections": [],
        "step_history": [],
        "errors": [],
        "plan": ar.AgentPlan(goal="g", steps=["s1", "s2", "s3"]),
        "quality_metrics": {},
        "execution_metrics": {},
    }
    base.update(overrides)
    # Build a real AgentState instance to ensure Pydantic behavior
    return AgentState(**base)


def test_create_agent_plan_variants():
    p1 = ar.create_agent_plan({"domain": "d", "a": {}, "b": {}})
    assert "simple" in p1.context["plan_type"] and len(p1.steps) == 4

    p2 = ar.create_agent_plan({"domain": "d", **{str(i): {} for i in range(5)}})
    assert "moderate" in p2.context["plan_type"] and len(p2.steps) > 4

    p3 = ar.create_agent_plan({"domain": "d", **{str(i): {} for i in range(20)}})
    assert "complex" in p3.context["plan_type"] and len(p3.steps) > 6


def test_reason_before_action_thoughts_and_history():
    s = make_state()
    for action in ["fetch_rules", "suggest", "format", "normalize", "other"]:
        updates = ar.reason_before_action(s, action)
        assert updates["thoughts"]
        assert updates["step_history"][-1].action == action


def test_observe_after_action_updates_observations():
    s = make_state()
    s = s.copy(update=ar.reason_before_action(s, "fetch_rules"))
    out = ar.observe_after_action(s, "fetch_rules", [{"rule_name": "X"}])
    assert "retrieved" in out["observations"][0]

    s = s.copy(update=ar.reason_before_action(s, "suggest"))
    out = ar.observe_after_action(s, "suggest", "line1\nline2")
    assert "LLM generated" in out["observations"][-1]

    s = s.copy(update=ar.reason_before_action(s, "format"))
    out = ar.observe_after_action(s, "format", [1, 2, 3])
    assert "formatted" in out["observations"][-1]

    s = s.copy(update=ar.reason_before_action(s, "normalize"))
    out = ar.observe_after_action(s, "normalize", {"a": {"expectations": [1]}, "b": {"expectations": []}})
    assert "Normalization completed" in out["observations"][-1]


def test_reflect_on_progress_metrics_and_reflection():
    s = make_state()
    # Simulate some completed steps and errors
    s = s.copy(update=ar.reason_before_action(s, "fetch_rules"))
    s = s.copy(update=ar.observe_after_action(s, "fetch_rules", []))
    s = s.copy(update={"errors": ["e1", "e2", "e3"]})

    out = ar.reflect_on_progress(s)
    assert out["reflections"]
    qm = out["quality_metrics"]
    assert 0.0 <= qm["progress"] <= 1.0
    assert "error_rate" in qm
