from pathlib import Path

from eb.eval.parse_play import parse_play

SAMPLE = Path(
    "/data/kk/Berkeley-Humanoid-Lite/logs/rsl_rl/wbc/"
    "2026-08-08_11-49-53_wbc_rsl_final_entropy001_seed42_resume399/"
    "eval_model_1700_seed20260808.json"
)


def test_parse_existing_eval_json():
    r = parse_play(SAMPLE)
    assert 0 <= r["success_rate"] <= 1
    assert r["per_seed"]
    assert r["episodes"] == 344


def test_explicit_success_and_seeds():
    r = parse_play({"success_rate": 0.5, "seeds": [0, 1, 2, 3], "metrics": {}})
    assert r["success_rate"] == 0.5
    assert r["per_seed"] == [0.5, 0.5, 0.5, 0.5]


def test_per_seed_list_preserved():
    r = parse_play({"success_rate": 0.25, "per_seed": [0.0, 0.5, 0.0, 0.5]})
    assert r["per_seed"] == [0.0, 0.5, 0.0, 0.5]
