"""Unit tests of the arena's rules handling (referee.make_rules, RulesState)."""
import pytest

from quoridor_arena.referee import STANDARD_RULES, RulesError, RulesState, make_rules, rules_tag


class FakeEngine:
    def __init__(self, reject=()):
        self.starts = 1
        self.sent = []
        self.reject = reject

    def send(self, cmd):
        self.sent.append(cmd)
        return (not any(cmd.startswith(r) for r in self.reject)), ""


def test_make_rules():
    assert make_rules() == STANDARD_RULES
    assert make_rules({"komi": 1.5}, {"whiteInitialWalls": 7}, {"komi": -2.5}) == {
        "komi": -2.5, "blackInitialWalls": 10, "whiteInitialWalls": 7, "repetitionDrawCount": 0}
    for bad in ({"komi": 0}, {"komi": 1.25}, {"komi": 21.5}, {"blackInitialWalls": -1}, {"whiteInitialWalls": 11},
                {"whiteInitialWalls": 9.5}, {"maxPlies": 300}, {"repetitionDrawCount": 1},
                {"repetitionDrawCount": -1}, {"repetitionDrawCount": 3.0}, {"repetitionDrawCount": True}):
        with pytest.raises(ValueError):
            make_rules(bad)
    assert rules_tag(STANDARD_RULES) == "" and rules_tag(None) == ""
    assert rules_tag(make_rules({"komi": 1.5, "whiteInitialWalls": 9})) == "_k+1.5_w10-9"
    assert make_rules({"repetitionDrawCount": 3})["repetitionDrawCount"] == 3
    assert rules_tag(make_rules({"repetitionDrawCount": 3})) == "_k-0.5_w10-10_r3"


def test_rules_state_sends_only_changes_and_resets_on_restart():
    eng, state = FakeEngine(), RulesState()
    # A fresh process has the standard rules: nothing to send.
    state.sync(eng, STANDARD_RULES, "e")
    assert eng.sent == []
    r = make_rules({"komi": 1.5, "blackInitialWalls": 8})
    state.sync(eng, r, "e")
    assert eng.sent == ["komi 1.5", "kata-set-rule blackInitialWalls 8"]
    eng.sent.clear()
    state.sync(eng, r, "e")
    assert eng.sent == []
    # Back to the standard game: the rules persist across clear_board, so they are reset.
    state.sync(eng, STANDARD_RULES, "e")
    assert eng.sent == ["komi -0.5", "kata-set-rule blackInitialWalls 10"]
    eng.sent.clear()
    state.sync(eng, make_rules({"repetitionDrawCount": 3}), "e")
    assert eng.sent == ["kata-set-rule repetitionDrawCount 3"]
    eng.sent.clear()
    state.sync(eng, STANDARD_RULES, "e")
    assert eng.sent == ["kata-set-rule repetitionDrawCount 0"]
    # After a restart the process has the standard rules again.
    state.sync(eng, r, "e")
    eng.sent.clear()
    eng.starts += 1
    state.sync(eng, STANDARD_RULES, "e")
    assert eng.sent == []


def test_rules_state_rejection():
    eng, state = FakeEngine(reject=("komi",)), RulesState()
    with pytest.raises(RulesError):
        state.sync(eng, make_rules({"komi": 1.5}), "e")
    # Unknown state after a failure: the next sync sends everything that differs from the standard game.
    eng.reject = ()
    eng.sent.clear()
    state.sync(eng, make_rules({"komi": 1.5}), "e")
    assert eng.sent == ["komi 1.5"]
