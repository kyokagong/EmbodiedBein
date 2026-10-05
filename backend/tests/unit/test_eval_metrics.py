import pytest
from eb.eval.wilson import wilson_ci, is_statistically_significant
from eb.eval.cross_domain import compute_observation_drift


def test_wilson_ci_standard():
    # 52 successes in 64 trials (81.25%)
    ci = wilson_ci(52, 64, confidence=0.95)
    assert ci["n"] == 64
    assert 0.70 <= ci["lower"] <= 0.73
    assert 0.88 <= ci["upper"] <= 0.91
    assert "81.2%" in ci["label"]


def test_wilson_ci_boundary_cases():
    ci_zero = wilson_ci(0, 50, confidence=0.95)
    assert ci_zero["point"] == 0.0
    assert ci_zero["lower"] == 0.0
    assert ci_zero["upper"] > 0.0  # Wilson upper bound is > 0 for 0 successes!

    ci_full = wilson_ci(50, 50, confidence=0.95)
    assert ci_full["point"] == 1.0
    assert ci_full["lower"] < 1.0
    assert ci_full["upper"] == 1.0


def test_statistical_significance_overlapping():
    ci_a = wilson_ci(50, 100) # [0.40, 0.60]
    ci_b = wilson_ci(55, 100) # [0.45, 0.65] (overlaps)
    assert not is_statistically_significant(ci_a, ci_b)

    ci_c = wilson_ci(90, 100) # [0.82, 0.95] (does not overlap with ci_a)
    assert is_statistically_significant(ci_a, ci_c)


def test_cross_domain_drift_computation():
    layout = [
        {"name": "base_ang_vel", "dim": 3},
        {"name": "projected_gravity", "dim": 3},
        {"name": "joint_pos_rel", "dim": 12},
    ]
    report = compute_observation_drift("ee_bhl_biped_12dof_v1", layout)
    assert report["embodiment_env_id"] == "ee_bhl_biped_12dof_v1"
    assert report["terms"] == []
    assert report["divergence_step"] is None
    assert report["passed_sim2sim_gate"] is None
    assert "sim2sim" in report["summary"]

    stats = {item["name"]: {"mean": 0.0, "std": 1.0} for item in layout}
    measured = compute_observation_drift("ee", layout, stats, stats)
    assert len(measured["terms"]) == 3
    assert measured["max_drift_value"] == 0.0
    assert measured["passed_sim2sim_gate"] is None


def test_report_interval_never_invents_sample_count():
    from eb.eval.wilson import report_interval
    assert report_interval({"success_rate": 0.5}) is None
    assert report_interval({"success_rate": 0.5, "n_trials": 0}) is None
    assert report_interval({"success_rate": float("nan"), "n_trials": 4}) is None
    assert report_interval({"success_rate": 0.5, "n_trials": 4})["n"] == 4
