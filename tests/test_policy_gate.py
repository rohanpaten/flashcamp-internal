import pytest
from flashcamp.backend.analysis import predict_success

def make_metrics(scores, strict=False):
    # Map to expected keys for pillar_scores
    return {
        "capital_score": scores.get("Capital", 0),
        "market_score": scores.get("Market", 0),
        "people_score": scores.get("People", 0),
        "advantage_score": scores.get("Advantage", 0),
        "strict_mode": strict
    }

@pytest.mark.parametrize("scores,strict,exp", [
    ({"Capital":0.49,"Market":0.9,"People":0.9,"Advantage":0.9}, True, "fail"),
    ({"Capital":0.21,"Market":0.21,"People":0.6,"Advantage":0.6}, False, "pass"),
    ({"Capital":0.51,"Market":0.51,"People":0.51,"Advantage":0.51}, True, "pass"),
    ({"Capital":0.19,"Market":0.9,"People":0.9,"Advantage":0.9}, False, "pass"),
])
def test_policy_gate(scores, strict, exp):
    metrics = make_metrics(scores, strict)
    result = predict_success(metrics)
    assert result["label"] == exp
    if strict and exp == "fail":
        assert any(v < 0.5 for v in scores.values()) 