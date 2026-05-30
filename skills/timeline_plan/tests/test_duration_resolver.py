"""Tests for Duration Resolver."""
from skills.timeline_plan.duration_resolver import (
    resolve_duration,
    _try_pad_strategy,
)


def _rl(line_id, start_s, end_s, original="x", rewritten="y", speaker="S"):
    return {
        "line_id": line_id,
        "start_seconds": start_s,
        "end_seconds": end_s,
        "original": original,
        "rewritten": rewritten,
        "speaker": speaker,
        "shot_number": 1,
        "shot_scene": "",
    }


def _nl(line_id, dialogue, start_s, end_s, speaker="S"):
    return {
        "line_id": line_id,
        "dialogue": dialogue,
        "speaker": speaker,
        "start_seconds": start_s,
        "end_seconds": end_s,
    }


class TestTryPadStrategy:
    def test_pad_after_when_close_to_min(self):
        group = [_rl("l1", 2.83, 3.5), _rl("l2", 5.3, 6.43)]
        result = _try_pad_strategy(group, "pad_after", min_duration=4.0)
        assert result is not None
        new_min = min(r["start_seconds"] for r in result)
        new_max = max(r["end_seconds"] for r in result)
        assert new_max - new_min >= 4.0
        assert new_min == 2.83

    def test_pad_before(self):
        group = [_rl("l1", 3.5, 7.0)]
        result = _try_pad_strategy(group, "pad_before", min_duration=4.0)
        assert result is not None
        new_min = min(r["start_seconds"] for r in result)
        new_max = max(r["end_seconds"] for r in result)
        assert new_max - new_min >= 4.0
        assert new_max == 7.0

    def test_direct_when_already_long_enough(self):
        group = [_rl("l1", 1.0, 6.0)]
        result = _try_pad_strategy(group, "direct", min_duration=4.0)
        assert result is not None
        assert result == group


class TestResolveDuration:
    def test_pads_short_group(self):
        group = [_rl("l1", 2.83, 3.5), _rl("l2", 5.3, 6.43)]
        all_lines = {
            "l1": _nl("l1", "hello", 2.83, 3.5),
            "l2": _nl("l2", "world", 5.3, 6.43),
        }
        line_to_node = {"l1": "n1", "l2": "n1"}

        extended, strategy, duration = resolve_duration(
            group, all_lines, line_to_node, min_duration=4.0
        )
        assert duration >= 4.0
        assert strategy in ("pad_after", "pad_before")
        extended_ids = {r["line_id"] for r in extended}
        assert "l1" in extended_ids
        assert "l2" in extended_ids

    def test_already_long_enough_unchanged(self):
        group = [_rl("l1", 17.47, 18.27), _rl("l2", 18.83, 20.03), _rl("l3", 21.15, 29.55)]
        all_lines = {r["line_id"]: _nl(r["line_id"], r["original"], r["start_seconds"], r["end_seconds"]) for r in group}
        line_to_node = {r["line_id"]: "n1" for r in group}

        extended, strategy, duration = resolve_duration(
            group, all_lines, line_to_node, min_duration=4.0
        )
        assert strategy == "direct"
        assert duration >= 4.0
        assert len(extended) == 3

    def test_single_short_line_padded(self):
        group = [_rl("l1", 17.47, 18.27)]
        all_lines = {"l1": _nl("l1", "test", 17.47, 18.27)}
        line_to_node = {"l1": "n1"}

        extended, strategy, duration = resolve_duration(
            group, all_lines, line_to_node, min_duration=4.0
        )
        assert duration >= 4.0
        assert "l1" in {r["line_id"] for r in extended}

    def test_empty_group(self):
        extended, strategy, duration = resolve_duration(
            [], {}, {}, min_duration=4.0
        )
        assert len(extended) == 0
        assert strategy == "direct"
        assert duration == 0.0
