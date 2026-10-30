import json
from pathlib import Path

C = json.loads((Path(__file__).resolve().parents[1] / "docs" / "check.json").read_text())


def test_demo_results_are_reported_honestly():
    assert C["random_forest"]["accuracy"] < 0.7
    assert C["planted_rule"]["churn_if_rule"] > C["planted_rule"]["churn_if_not"]
