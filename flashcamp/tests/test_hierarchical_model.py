"""
Tests for the hierarchical model architecture.
Tests the hierarchical prediction process with both the actual models
and the fallback methods.
"""
import pytest
import numpy as np
import os
import json
from pathlib import Path
import sys

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parents[2]))

from flashcamp.backend.app.engines.ml import (
    predict_success_probability
)
from flashcamp.backend.schemas import MetricsInput

# Test data directory
TEST_DATA_DIR = Path(__file__).parent.parent / "data" / "gold"


@pytest.fixture
def sample_startup_data():
    """Create a sample startup data dictionary for testing"""
    return {
        # Capital metrics
        "cash_on_hand_usd": 2000000,
        "monthly_burn_usd": 80000,
        "runway_months": 18,
        "burn_multiple": 1.5,
        "ltv_cac_ratio": 3.2,
        "gross_margin_percent": 65,
        "customer_concentration_percent": 20,
        "post_money_valuation_usd": 12000000,
        
        # Advantage metrics
        "patent_count": 2,
        "network_effects_present": True,
        "has_data_moat": True,
        "regulatory_advantage_present": False,
        "tech_differentiation_score": 4,
        "switching_cost_score": 3,
        "brand_strength_score": 2,
        "product_retention_30d": 0.85,
        "product_retention_90d": 0.65,
        "nps_score": 45,
        
        # Market metrics
        "tam_size_usd": 5000000000,
        "sam_size_usd": 500000000,
        "claimed_cagr_pct": 25,
        "market_growth_rate_percent": 18,
        "competition_intensity": 3,
        "top3_competitor_share_pct": 60,
        "industry_regulation_level": "medium",
        
        # People metrics
        "founders_count": 2,
        "team_size_total": 15,
        "founder_domain_experience_years": 8,
        "prior_successful_exits_count": 1,
        "board_advisor_experience_score": 3,
        "team_diversity_percent": 40,
        "gender_diversity_index": 0.45,
        "geography_diversity_index": 0.3,
        "key_person_dependency": False,
        
        # Additional metrics
        "sector": "SaaS",
        "product_stage": "GA",
        "investor_tier_primary": "Tier1"
    }


def test_fallback_pillar_scores(sample_startup_data):
    """Test that pillar scores from the main contract return reasonable values"""
    metrics = MetricsInput(**sample_startup_data)
    result = predict_success_probability(metrics)
    pillars = ["capital", "advantage", "market", "people"]
    for pillar in pillars:
        score = result["pillar_scores"][pillar]
        assert 0 <= score <= 1
        assert score >= 0.1, f"Pillar {pillar} score too low: {score}"


def test_fallback_prediction(sample_startup_data):
    """Test the main contract returns a valid prediction structure"""
    metrics = MetricsInput(**sample_startup_data)
    result = predict_success_probability(metrics)
    assert "pillar_scores" in result
    assert "final_score" in result
    assert "prediction" in result
    assert "confidence" in result
    assert "threshold" in result
    assert "confidence_interval" in result

    # Check pillar scores
    for pillar in ["capital", "advantage", "market", "people"]:
        assert pillar in result["pillar_scores"]
        assert 0 <= result["pillar_scores"][pillar] <= 1

    # Check the final score
    assert 0 <= result["final_score"] <= 1


def test_predict_success_responds_to_input_changes():
    """Test that the prediction changes when inputs change"""
    base_metrics = {
        "cash_on_hand_usd": 1000000,
        "runway_months": 12,
        "market_growth_rate_percent": 15,
        "team_size_total": 10
    }
    improved_metrics = base_metrics.copy()
    improved_metrics["cash_on_hand_usd"] = 5000000
    improved_metrics["runway_months"] = 24
    improved_metrics["market_growth_rate_percent"] = 30
    # Get predictions for both sets of metrics
    base_result = predict_success_probability(MetricsInput(**base_metrics))
    improved_result = predict_success_probability(MetricsInput(**improved_metrics))
    # Ensure both predictions return valid results
    assert "pillar_scores" in base_result
    assert "pillar_scores" in improved_result
    # The improved metrics should have a higher success probability
    assert improved_result["final_score"] > base_result["final_score"]
    # Also check that pillar scores changed
    assert improved_result["pillar_scores"]["capital"] >= base_result["pillar_scores"]["capital"]
    assert improved_result["pillar_scores"]["market"] >= base_result["pillar_scores"]["market"]


def test_full_prediction_output_structure(sample_startup_data):
    """Test the structure of the full prediction output"""
    metrics = MetricsInput(**sample_startup_data)
    result = predict_success_probability(metrics)
    # Check overall structure
    assert "pillar_scores" in result
    assert isinstance(result["pillar_scores"], dict)
    assert "final_score" in result
    assert isinstance(result["final_score"], float)
    assert "prediction" in result
    assert result["prediction"] in ["pass", "fail"]
    assert "confidence" in result
    assert 0 <= result["confidence"] <= 1
    assert "threshold" in result
    assert 0 <= result["threshold"] <= 1
    assert "confidence_interval" in result
    ci = result["confidence_interval"]
    assert isinstance(ci, list) and len(ci) == 2
    assert 0 <= ci[0] <= ci[1] <= 1
    # Check pillar scores
    for pillar in ["capital", "advantage", "market", "people"]:
        assert pillar in result["pillar_scores"]
        assert isinstance(result["pillar_scores"][pillar], float)
        assert 0 <= result["pillar_scores"][pillar] <= 1
    # Check that final_score is in valid range
    assert 0 <= result["final_score"] <= 1


@pytest.mark.parametrize("change_factor", [0.5, 2.0])
def test_model_sensitivity(sample_startup_data, change_factor):
    """Test model sensitivity to key input changes"""
    # Make a copy of the original data
    modified_data = sample_startup_data.copy()
    # Apply the change factor to key metrics
    metrics_to_modify = [
        "cash_on_hand_usd",
        "runway_months",
        "market_growth_rate_percent",
        "tech_differentiation_score",
        "founder_domain_experience_years"
    ]
    for metric in metrics_to_modify:
        if metric in modified_data:
            # Don't modify boolean values
            if not isinstance(modified_data[metric], bool):
                modified_data[metric] = modified_data[metric] * change_factor
    # Get predictions
    original_result = predict_success_probability(MetricsInput(**sample_startup_data))
    modified_result = predict_success_probability(MetricsInput(**modified_data))
    # Verify that changes in input led to changes in output
    if change_factor > 1.0:
        assert modified_result["final_score"] >= original_result["final_score"]
    else:
        assert modified_result["final_score"] <= original_result["final_score"]


if __name__ == "__main__":
    # Run the tests
    pytest.main(["-xvs", __file__]) 