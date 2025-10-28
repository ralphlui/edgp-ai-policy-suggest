from types import SimpleNamespace

from app.api import rule_suggestion_routes as rs


def make_state(**overrides):
    base = {
        "errors": [],
        "step_history": ["start", "end"],
        "execution_metrics": {"total_execution_time": 2.5},
        "rule_suggestions": [{}, {}, {}, {}, {}],
        "data_schema": {"domain": "x", "col1": {}, "col2": {}},
        "thoughts": [1, 2, 3, 4, 5],
        "observations": [1, 2],
        "reflections": [1],
        "confidence_scores": {"dummy": 1.0},
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_sanitize_for_logging_strips_controls_and_specials():
    raw = "hello\nworld<>`"  # includes newline and special chars
    cleaned = rs.sanitize_for_logging(raw)
    assert "\n" not in cleaned
    assert "<" not in cleaned and ">" not in cleaned and "`" not in cleaned


def test_confidence_helpers_happy_path():
    state = make_state()
    overall = rs._calculate_overall_confidence(state)
    assert 0.0 <= overall <= 1.0

    level = rs._get_confidence_level(state)
    assert level in {"high", "medium", "low", "very_low"}

    factors = rs._get_confidence_factors(state)
    assert set(factors.keys()) == {"rule_generation", "error_handling", "execution_performance", "reasoning_depth"}
    assert factors["rule_generation"]["rules_generated"] == 5


def test_confidence_helpers_edge_cases():
    # Lots of errors and long time reduce confidence
    state = make_state(errors=[1, 2, 3, 4], execution_metrics={"total_execution_time": 20.0}, rule_suggestions=[])
    level = rs._get_confidence_level(state)
    assert level in {"low", "very_low"}

    # No data_schema
    state2 = make_state(data_schema=None)
    overall2 = rs._calculate_overall_confidence(state2)
    assert 0.0 <= overall2 <= 1.0
