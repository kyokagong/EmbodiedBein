import pytest

from eb.runners.ufo_training_analysis import snapshot_csv, summarize


def test_training_analysis_preserves_native_windows_and_missing_values(tmp_path):
    path = tmp_path / "train_log.txt"
    path.write_text(
        "distributed/optimizer_steps,fb_loss,timestep\n16,4,11264\n32,,12288\n48,2,13312\n64,NaN,14336\n80,"
    )
    raw, rows = snapshot_csv(path)
    assert raw.endswith(b"64,NaN,14336\n")
    result = summarize(rows)
    assert result["fb_loss"]["points"] == 2
    assert result["fb_loss"]["missing_or_nonnumeric"] == 2
    assert result["fb_loss"]["first_window_mean"] == 4
    assert result["fb_loss"]["last_window_mean"] == 2
    assert result["timestep"]["last_update"] == 64


def test_analysis_requires_actual_update_axis():
    with pytest.raises(ValueError, match="agent-update"):
        summarize([{"timestep": "100", "fb_loss": "1"}])
