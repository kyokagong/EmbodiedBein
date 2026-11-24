import pytest

from eb.motiondecode_preview import frame_selection, preview_file


def test_video_selection_is_bounded_and_preserves_source_indices():
    assert frame_selection(12, 4) == (12, [4, 8])
    end, indices = frame_selection(20000, 0)
    assert end == 7200 and len(indices) == 1800
    assert indices[0] == 0 and indices[-1] == 7196
    assert frame_selection(7201, 7200) == (7201, [7200])
    for start in [-1, 12]:
        with pytest.raises(ValueError):
            frame_selection(12, start)


@pytest.mark.parametrize("path", ["../receipt.json", "table/../receipt.json", "table/-1.json", "other.mp4"])
def test_preview_files_cannot_escape_the_frozen_receipt(path):
    with pytest.raises(ValueError):
        preview_file(path, {"table_pages": 2})


def test_preview_table_pages_have_bounds():
    assert preview_file("table/000001.json", {"table_pages": 2}) == "table/000001.json"
    assert preview_file("replay.mp4", {"table_pages": 2}) == "replay.mp4"
    with pytest.raises(ValueError):
        preview_file("table/000002.json", {"table_pages": 2})
