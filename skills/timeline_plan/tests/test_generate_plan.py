"""Tests for timeline plan generator orchestrator."""
import json
from dataclasses import asdict
from skills.timeline_plan.models import TimelinePlan, CanvasNode, Stage3Input
from skills.timeline_plan.generate_plan import generate_timeline_plan, _classify_operation_type, _fuzzy_word_match


class FakeLine:
    def __init__(self, line_id, dialogue, start_s=0.0, end_s=1.0):
        self.line_id = line_id
        self.dialogue = dialogue
        self.start_seconds = start_s
        self.end_seconds = end_s


class FakeShot:
    def __init__(self, shot_number, start, end, scene_desc="", lines=None):
        self.shot_number = shot_number
        self.start_seconds = start
        self.end_seconds = end
        self.scene_description = scene_desc
        self.lines = lines or []


class FakeScript:
    def __init__(self, shots):
        self.shots = shots


class FakeScriptOutput:
    def __init__(self, shots):
        self.script = FakeScript(shots)


def make_rewrite(line_id, original, rewritten, shot_num, start_s, end_s):
    return {"line_id": line_id, "original": original, "rewritten": rewritten,
            "shot_number": shot_num, "start_seconds": start_s, "end_seconds": end_s,
            "shot_scene": "", "speaker": "Speaker"}


class TestGenerateTimelinePlan:
    def test_produces_original_when_no_rewrite(self):
        shots = [FakeShot(1, 0.0, 10.0, "Opening", [FakeLine("p1_l1", "hello", 1.0, 2.0)])]
        script = FakeScriptOutput(shots)
        rewrite = {"level": "B2", "lines": [make_rewrite("p1_l1", "hello", "hello", 1, 1.0, 2.0)]}
        inp = Stage3Input(script_output=script, rewrite_json=rewrite, level="B2")
        plan = generate_timeline_plan(inp)
        assert len(plan.items) >= 1
        assert plan.items[0].source == "original"

    def test_produces_seedance_when_rewritten(self):
        shots = [FakeShot(1, 0.0, 10.0, "Opening", [FakeLine("p1_l1", "hello", 1.0, 6.0)])]
        script = FakeScriptOutput(shots)
        rewrite = {"level": "B2", "lines": [make_rewrite("p1_l1", "hello", "hi there", 1, 1.0, 6.0)]}
        nodes = [CanvasNode(node_id="n1", prompt='He says "hello"', video_url="http://x.com/v.mp4", reference_images=["http://x.com/r.png"])]
        inp = Stage3Input(script_output=script, rewrite_json=rewrite, canvas_nodes=nodes, level="B2")
        plan = generate_timeline_plan(inp)
        seedance_items = [i for i in plan.items if i.source == "seedance"]
        assert len(seedance_items) >= 1
        item = seedance_items[0]
        assert item.rewritten_prompt is not None
        assert "hi there" in item.rewritten_prompt

    def test_mixed_original_and_seedance(self):
        shots = [
            FakeShot(1, 0.0, 5.0, "Scene A", [FakeLine("p1_l1", "hello", 1.0, 2.0)]),
            FakeShot(2, 5.0, 15.0, "Scene B", [FakeLine("p2_l1", "goodbye", 6.0, 7.0)]),
        ]
        script = FakeScriptOutput(shots)
        rewrite = {"level": "B2", "lines": [
            make_rewrite("p1_l1", "hello", "hello", 1, 1.0, 2.0),
            make_rewrite("p2_l1", "goodbye", "farewell", 2, 6.0, 11.0),
        ]}
        nodes = [CanvasNode(node_id="n2", prompt='She says "goodbye"', video_url="http://x.com/v2.mp4", reference_images=["http://x.com/r2.png"])]
        inp = Stage3Input(script_output=script, rewrite_json=rewrite, canvas_nodes=nodes, level="B2")
        plan = generate_timeline_plan(inp)
        seedance_items = [i for i in plan.items if i.source == "seedance"]
        assert len(seedance_items) >= 1
        assert plan.items[0].source == "original"
        assert seedance_items[0].source == "seedance"

    def test_degradation_level_tracking(self):
        shots = [FakeShot(1, 0.0, 10.0, "Scene", [FakeLine("p1_l1", "unique text", 1.0, 6.0)])]
        script = FakeScriptOutput(shots)
        rewrite = {"level": "B2", "lines": [make_rewrite("p1_l1", "unique text", "rewritten unique", 1, 1.0, 6.0)]}
        inp = Stage3Input(script_output=script, rewrite_json=rewrite, canvas_nodes=[], level="B2")
        plan = generate_timeline_plan(inp)
        seedance_items = [i for i in plan.items if i.source == "seedance"]
        assert len(seedance_items) >= 1
        assert seedance_items[0].degradation_level > 0

    def test_json_serializable_output(self):
        shots = [FakeShot(1, 0.0, 10.0, "Scene", [FakeLine("p1_l1", "hi", 1.0, 6.0)])]
        script = FakeScriptOutput(shots)
        rewrite = {"level": "B2", "lines": [make_rewrite("p1_l1", "hi", "hello", 1, 1.0, 6.0)]}
        inp = Stage3Input(script_output=script, rewrite_json=rewrite, level="B2")
        plan = generate_timeline_plan(inp)
        json_str = json.dumps(asdict(plan), indent=2)
        parsed = json.loads(json_str)
        assert parsed["pipeline_version"] == "2.0"
        assert len(parsed["items"]) >= 1

    def test_empty_shots(self):
        script = FakeScriptOutput([])
        rewrite = {"level": "B2", "lines": []}
        inp = Stage3Input(script_output=script, rewrite_json=rewrite, level="B2")
        plan = generate_timeline_plan(inp)
        assert len(plan.items) >= 1  # gap-filling creates a full-length original segment

    def test_short_rewritten_line_not_dropped(self):
        shots = [FakeShot(1, 0.0, 10.0, "Opening", [FakeLine("p1_l1", "hello", 1.0, 2.0)])]
        script = FakeScriptOutput(shots)
        rewrite = {"level": "B2", "lines": [make_rewrite("p1_l1", "hello", "hi there", 1, 1.0, 2.0)]}
        nodes = [CanvasNode(node_id="n1", prompt='He says "hello"', video_url="http://x.com/v.mp4", reference_images=["http://x.com/r.png"])]
        inp = Stage3Input(script_output=script, rewrite_json=rewrite, canvas_nodes=nodes, level="B2")
        plan = generate_timeline_plan(inp)
        seedance_items = [i for i in plan.items if i.source == "seedance"]
        assert len(seedance_items) >= 1, "Short group was silently dropped!"
        item = seedance_items[0]
        assert item.rewritten_prompt is not None
        assert "hi there" in item.rewritten_prompt
        assert item.duration_strategy is not None
        assert item.duration_strategy in ("pad_after", "pad_before", "forced_min_duration")


def _rl(line_id, original, rewritten, start_s, end_s):
    return {"line_id": line_id, "original": original, "rewritten": rewritten,
            "start_seconds": start_s, "end_seconds": end_s}


class TestClassifyOperationType:
    def test_literal_replace_when_dialogue_in_prompt(self):
        prompt = 'He says "hello" and "world"'
        lines = [_rl("l1", "hello", "hi", 1.0, 2.0)]
        result = _classify_operation_type(prompt, lines)
        assert result == "literal_replace"

    def test_semantic_insert_when_dialogue_missing(self):
        prompt = "美式情景喜剧，真实的破防（面部特写）"
        lines = [_rl("l1", "no no no", "No, no, no, this can't be.", 17.0, 18.0)]
        result = _classify_operation_type(prompt, lines)
        assert result == "semantic_insert"

    def test_mixed_lines_prefer_literal(self):
        prompt = 'He says "hello" during a breakdown scene'
        lines = [
            _rl("l1", "hello", "hi", 1.0, 2.0),
            _rl("l2", "goodbye", "farewell", 3.0, 4.0),
        ]
        result = _classify_operation_type(prompt, lines)
        assert result == "literal_replace"

    def test_full_fallback_when_no_prompt(self):
        lines = [_rl("l1", "hello", "hi", 1.0, 2.0)]
        result = _classify_operation_type("", lines)
        assert result == "full_fallback"

    def test_unchanged_lines_ignored(self):
        prompt = "美式情景喜剧"
        lines = [_rl("l1", "hello", "hello", 1.0, 2.0)]
        result = _classify_operation_type(prompt, lines)
        assert result == "literal_replace"

    def test_fuzzy_replace_with_asr_drift(self):
        prompt = 'Donny says: "no, no, no!" '
        lines = [_rl("l1", "no no no", "No, no, no, this can't be.", 1.0, 2.0)]
        result = _classify_operation_type(prompt, lines)
        assert result == "fuzzy_replace"

    def test_fuzzy_replace_has_lower_priority_than_literal(self):
        prompt = 'Donny says: "hello" and "no, no, no!"'
        lines = [
            _rl("l1", "hello", "hi", 1.0, 2.0),
            _rl("l2", "no no no", "no", 3.0, 4.0),
        ]
        result = _classify_operation_type(prompt, lines)
        assert result == "literal_replace"


class TestFuzzyWordMatch:
    def test_exact_match(self):
        assert _fuzzy_word_match("hello world", "Donny says: hello world")

    def test_punctuation_drift(self):
        assert _fuzzy_word_match("no no no", 'Donny says: "no, no, no!"')

    def test_partial_match_below_threshold(self):
        assert not _fuzzy_word_match("hello world", "Donny says: goodbye")

    def test_short_words_ignored(self):
        assert not _fuzzy_word_match("no no", "nothing here")


class TestEpisode1Scenario:
    """End-to-end simulation of Episode 1's real failure modes.

    Node 13126e5a covers p001_l001-l005:
    - l001-l002: dialogue IS in prompt as quoted English → literal_replace
    - l003-l005: dialogue NOT in prompt, described in Chinese → semantic_insert
    """

    def test_short_group_padded_not_dropped(self):
        shots = [FakeShot(1, 0.0, 30.0, "Graduation celebration", [
            FakeLine("p001_l001", "this ceremony is boring", 2.83, 4.19),
            FakeLine("p001_l002", "let's see who wants me", 5.31, 6.43),
            FakeLine("p001_l003", "no no no", 17.47, 18.27),
            FakeLine("p001_l004", "they all refused me", 18.83, 20.03),
            FakeLine("p001_l005", "are they going crazy Donnie", 21.15, 29.55),
        ])]
        script = FakeScriptOutput(shots)
        rewrite = {"level": "B2", "lines": [
            make_rewrite("p001_l001", "this ceremony is boring",
                         "This ceremony is not entertaining at all.", 1, 2.83, 4.19),
            make_rewrite("p001_l002", "let's see who wants me",
                         "Let's see who is desperate to hire me.", 1, 5.31, 6.43),
            make_rewrite("p001_l003", "no no no",
                         "No, no, no, this can't be.", 1, 17.47, 18.27),
            make_rewrite("p001_l004", "they all refused me",
                         "Every single one of them rejected me.", 1, 18.83, 20.03),
            make_rewrite("p001_l005", "are they going crazy Donnie",
                         "Have they gone completely crazy, Donny?", 1, 21.15, 29.55),
        ]}
        nodes = [CanvasNode(
            node_id="13126e5a",
            prompt=(
                '美式情景喜剧，真实短剧，柔光雾化，画面通透，8k，超高清，电影级布光。'
                '镜头 1：三层景深与霸总转身'
                '..."This ceremony is boring."（这派对太无聊了。）...'
                '"Let\'s see who wants me"...'
                '镜头 2：真实的破防（面部特写）'
                '画面与动作：面部特写，固定机位。男主刚 Wink 完后表情瞬间崩塌...'
                '镜头3：眼部超大特写...'
            ),
            video_url="http://x.com/v.mp4",
            reference_images=["http://x.com/r.png"],
        )]
        inp = Stage3Input(script_output=script, rewrite_json=rewrite, canvas_nodes=nodes, level="B2")
        plan = generate_timeline_plan(inp)
        seedance_items = [i for i in plan.items if i.source == "seedance"]
        assert len(seedance_items) >= 1, f"Expected seedance items, got {len(seedance_items)}"

    def test_operation_type_is_semantic_insert_for_implicit_dialogue(self):
        shots = [FakeShot(1, 0.0, 30.0, "Graduation celebration", [
            FakeLine("p001_l003", "no no no", 17.47, 18.27),
            FakeLine("p001_l004", "they all refused me", 18.83, 20.03),
            FakeLine("p001_l005", "are they going crazy Donnie", 21.15, 29.55),
        ])]
        script = FakeScriptOutput(shots)
        rewrite = {"level": "B2", "lines": [
            make_rewrite("p001_l003", "no no no",
                         "No, no, no, this can't be.", 1, 17.47, 18.27),
            make_rewrite("p001_l004", "they all refused me",
                         "Every single one of them rejected me.", 1, 18.83, 20.03),
            make_rewrite("p001_l005", "are they going crazy Donnie",
                         "Have they gone completely crazy, Donny?", 1, 21.15, 29.55),
        ]}
        nodes = [CanvasNode(
            node_id="13126e5a",
            prompt="美式情景喜剧，8k，电影级布光。镜头 2：Donny says: \"no, no, no!\" 真实的破防（面部特写）。",
            video_url="http://x.com/v.mp4",
            reference_images=["http://x.com/r.png"],
        )]
        inp = Stage3Input(script_output=script, rewrite_json=rewrite, canvas_nodes=nodes, level="B2")
        plan = generate_timeline_plan(inp)
        seedance_items = [i for i in plan.items if i.source == "seedance"]
        assert len(seedance_items) >= 1, "Lines should produce seedance items"
        group_b = seedance_items[0]
        assert group_b.operation_type in ("semantic_insert", "fuzzy_replace"), (
            f"Expected semantic_insert or fuzzy_replace, got {group_b.operation_type}"
        )

    def test_style_preserved_in_fallback(self):
        shots = [FakeShot(1, 0.0, 30.0, "Graduation celebration", [
            FakeLine("p001_l003", "no no no", 17.47, 18.27),
            FakeLine("p001_l004", "they all refused me", 18.83, 20.03),
            FakeLine("p001_l005", "are they going crazy Donnie", 21.15, 29.55),
        ])]
        script = FakeScriptOutput(shots)
        rewrite = {"level": "B2", "lines": [
            make_rewrite("p001_l003", "no no no",
                         "No, no, no, this can't be.", 1, 17.47, 18.27),
            make_rewrite("p001_l004", "they all refused me",
                         "Every single one of them rejected me.", 1, 18.83, 20.03),
            make_rewrite("p001_l005", "are they going crazy Donnie",
                         "Have they gone completely crazy, Donny?", 1, 21.15, 29.55),
        ]}
        nodes = [CanvasNode(
            node_id="13126e5a",
            prompt="美式情景喜剧，真实短剧，柔光雾化，画面通透，8k，超高清，电影级布光。镜头 2：...Donny says: \"no no no\"...",
            video_url="http://x.com/v.mp4",
            reference_images=["http://x.com/r.png"],
        )]
        inp = Stage3Input(script_output=script, rewrite_json=rewrite, canvas_nodes=nodes, level="B2")
        plan = generate_timeline_plan(inp)
        seedance_items = [i for i in plan.items if i.source == "seedance"]
        assert len(seedance_items) >= 1
        prompt = seedance_items[0].rewritten_prompt
        assert prompt is not None and len(prompt) > 0
        # Rewritten dialogue must appear in the output
        assert "No, no, no, this can't be." in prompt
        # operation_type and degradation are tracked
        assert seedance_items[0].operation_type is not None
        assert seedance_items[0].degradation_level is not None
