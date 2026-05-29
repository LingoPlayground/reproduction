"""Tests for canvas node matcher."""
from types import SimpleNamespace
from skills.timeline_plan.models import CanvasNode
from skills.timeline_plan.canvas_matcher import (
    match_lines_to_nodes,
    _score_mapping,
    _compute_consistency,
)


class TestScoreMapping:
    def test_perfect_mapping_no_penalty(self):
        lines = [
            SimpleNamespace(line_id="l1", shot_number=1),
            SimpleNamespace(line_id="l2", shot_number=1),
            SimpleNamespace(line_id="l3", shot_number=2),
        ]
        mapping = {"l1": "n1", "l2": "n1", "l3": "n2"}
        score = _score_mapping(mapping, lines)
        assert score == 3.0  # 3 matches, no cross-shot splits within same shot

    def test_contiguity_penalty(self):
        """Consecutive lines in same shot → different nodes incurs penalty."""
        lines = [
            SimpleNamespace(line_id="l1", shot_number=1),
            SimpleNamespace(line_id="l2", shot_number=1),
        ]
        mapping = {"l1": "n1", "l2": "n2"}
        score = _score_mapping(mapping, lines)
        assert score == 1.5  # 2 matches - 0.5 penalty

    def test_empty_mapping(self):
        score = _score_mapping({}, [])
        assert score == 0.0

    def test_different_shots_no_penalty(self):
        """Lines in different shots → different nodes: no penalty."""
        lines = [
            SimpleNamespace(line_id="l1", shot_number=1),
            SimpleNamespace(line_id="l2", shot_number=2),
        ]
        mapping = {"l1": "n1", "l2": "n2"}
        score = _score_mapping(mapping, lines)
        assert score == 2.0  # 2 matches, different shots, no penalty


class TestComputeConsistency:
    def test_full_agreement(self):
        results = [
            {"l1": "n1", "l2": "n2"},
            {"l1": "n1", "l2": "n2"},
            {"l1": "n1", "l2": "n2"},
        ]
        conf = _compute_consistency(results)
        assert conf["l1"] == 1.0
        assert conf["l2"] == 1.0

    def test_partial_agreement(self):
        results = [
            {"l1": "n1", "l2": "n2"},
            {"l1": "n1", "l2": "n1"},  # l2 disagrees
            {"l1": "n1", "l2": "n2"},
        ]
        conf = _compute_consistency(results)
        assert conf["l1"] == 1.0
        assert conf["l2"] == 2.0 / 3.0

    def test_empty_results(self):
        conf = _compute_consistency([])
        assert conf == {}

    def test_line_only_in_some_runs(self):
        results = [
            {"l1": "n1"},
            {"l1": "n1", "l2": "n2"},
            {"l2": "n2"},
        ]
        conf = _compute_consistency(results)
        assert conf["l1"] == 1.0
        assert conf["l2"] == 1.0


class TestMatchLinesToNodes:
    def test_empty_returns_empty(self):
        node_groups, confidences = match_lines_to_nodes([], [])
        assert node_groups == {}
        assert confidences == {}

    def test_no_nodes_returns_empty(self):
        lines = [SimpleNamespace(line_id="l1", dialogue="hello")]
        node_groups, confidences = match_lines_to_nodes(lines, [])
        assert node_groups == {}
        assert confidences == {}

    def test_no_api_key_returns_empty(self):
        lines = [SimpleNamespace(line_id="l1", dialogue="hello", speaker="", shot_number=0, shot_scene="")]
        nodes = [CanvasNode(node_id="n1", prompt="test prompt", video_url="")]
        node_groups, confidences = match_lines_to_nodes(lines, nodes)
        assert node_groups == {}
        assert confidences == {}
