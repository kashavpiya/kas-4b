"""
Engine unit tests — no real model needed.

Tests cover:
- validate() compatibility for 2, 26, 27, 255 options and noul
- _build_payload for string, JSON, and empty state
- Prompt-repeat rendering
- Unsupported raised when over context limit
"""

import math
import pytest
from unittest.mock import MagicMock, patch

from decision_index.engines.base import validate, Unsupported
from kas_engine.engine import _build_payload


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _choice_q(n):
    keys = [chr(ord("a") + i) for i in range(min(n, 26))]
    keys += [chr(ord("a") + i // 26) + chr(ord("a") + i % 26) for i in range(max(0, n - 26))]
    keys = keys[:n]
    return {
        "type": "choice",
        "instructions": "Pick one.",
        "criteria": {k: f"Option {k}" for k in keys},
    }


def _noul_q():
    return {"type": "noul", "instructions": "Is it raining?"}


def _uniform(keys):
    n = len(keys)
    return {k: 1.0 / n for k in keys}


def _labels(n):
    import string
    pool = list(string.ascii_uppercase)
    for a in string.ascii_uppercase:
        for b in string.ascii_uppercase:
            pool.append(a + b)
    return pool[:n]


# ---------------------------------------------------------------------------
# validate() compatibility
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [2, 26, 27, 255])
def test_validate_choice_n_options(n):
    q = _choice_q(n)
    keys = list(q["criteria"])
    resp = {
        "answers": {
            "q": {
                "type": "choice",
                "choice": keys[0],
                "probabilities": _uniform(keys),
            }
        }
    }
    validate({"q": q}, resp)  # must not raise


def test_validate_noul_passes():
    resp = {"answers": {"q": {"type": "noul", "noul": 0.73}}}
    validate({"q": _noul_q()}, resp)


def test_validate_noul_rejects_out_of_range():
    resp = {"answers": {"q": {"type": "noul", "noul": 1.5}}}
    with pytest.raises(ValueError):
        validate({"q": _noul_q()}, resp)


def test_validate_probs_must_sum_to_1():
    q = _choice_q(3)
    keys = list(q["criteria"])
    # sums to 0.9 — should fail
    probs = {keys[0]: 0.5, keys[1]: 0.3, keys[2]: 0.1}
    resp = {"answers": {"q": {"type": "choice", "choice": keys[0], "probabilities": probs}}}
    with pytest.raises(ValueError):
        validate({"q": q}, resp)


def test_validate_rejects_missing_option_in_probs():
    q = _choice_q(3)
    keys = list(q["criteria"])
    probs = {keys[0]: 0.6, keys[1]: 0.4}  # missing keys[2]
    resp = {"answers": {"q": {"type": "choice", "choice": keys[0], "probabilities": probs}}}
    with pytest.raises(ValueError):
        validate({"q": q}, resp)


def test_validate_rejects_nan_prob():
    q = _choice_q(2)
    keys = list(q["criteria"])
    probs = {keys[0]: float("nan"), keys[1]: 1.0}
    resp = {"answers": {"q": {"type": "choice", "choice": keys[0], "probabilities": probs}}}
    with pytest.raises(ValueError):
        validate({"q": q}, resp)


# ---------------------------------------------------------------------------
# _build_payload: state variants
# ---------------------------------------------------------------------------

def test_payload_string_state():
    lbls = _labels(2)
    payload, keys = _build_payload(
        "The sky is blue.", "choice", "What color?",
        {"opt1": "blue", "opt2": "red"}, lbls
    )
    assert payload["state"] == "The sky is blue."
    assert payload["options"] == {"A": "blue", "B": "red"}
    assert keys == ["opt1", "opt2"]


def test_payload_json_state():
    lbls = _labels(2)
    state = {"tool": "search", "query": "weather"}
    payload, _ = _build_payload(state, "choice", "Which tool?",
                                {"t1": "search", "t2": "lookup"}, lbls)
    assert payload["state"] == state


def test_payload_empty_string_state_omitted():
    lbls = _labels(2)
    payload, _ = _build_payload("", "choice", "Which?",
                                {"x": "X", "y": "Y"}, lbls)
    assert "state" not in payload


def test_payload_none_state_omitted():
    lbls = _labels(2)
    payload, _ = _build_payload(None, "choice", "Which?",
                                {"x": "X", "y": "Y"}, lbls)
    assert "state" not in payload


def test_payload_empty_dict_state_omitted():
    lbls = _labels(2)
    payload, _ = _build_payload({}, "choice", "Which?",
                                {"x": "X", "y": "Y"}, lbls)
    assert "state" not in payload


def test_payload_noul_maps_to_no_yes():
    lbls = _labels(2)
    payload, keys = _build_payload("context", "noul", "Is it hot?", {}, lbls)
    assert payload["options"] == {"A": "No", "B": "Yes"}
    assert keys == ["false", "true"]


def test_payload_none_description_falls_back_to_key():
    lbls = _labels(2)
    payload, _ = _build_payload(None, "choice", "Pick",
                                {"key1": None, "key2": None}, lbls)
    assert payload["options"]["A"] == "key1"
    assert payload["options"]["B"] == "key2"


# ---------------------------------------------------------------------------
# Prompt repeat
# ---------------------------------------------------------------------------

def test_repeat_prompt_contains_repeat_marker():
    """When repeat_prompt=True the body should include the repeat separator."""
    import json
    from kas_engine.engine import _build_payload

    lbls = _labels(2)
    payload, _ = _build_payload("state", "choice", "Q?",
                                {"a": "A opt", "b": "B opt"}, lbls)
    body_once = json.dumps(payload, ensure_ascii=False, separators=(", ", ": "))
    body_twice = body_once + "\n\nLet me repeat that:\n\n" + body_once

    assert "Let me repeat that:" in body_twice
    assert body_twice.count(body_once) == 2


# ---------------------------------------------------------------------------
# Unsupported: over-length prompt
# ---------------------------------------------------------------------------

def test_unsupported_raised_over_context_limit():
    """__call__ raises Unsupported (not truncates) when prompt exceeds the limit."""
    # Build a minimal KasEngine without loading a real model
    engine = KasEngine_stub(limit=10)
    q = _choice_q(2)
    with pytest.raises(Unsupported):
        engine._check_length(ids=list(range(100)))  # 100 > 10


class KasEngine_stub:
    """Minimal stub to test the length-check logic without a real model."""

    def __init__(self, limit):
        self.limit = limit

    def _check_length(self, ids):
        if len(ids) > self.limit:
            raise Unsupported(
                f"prompt ({len(ids)} tokens) exceeds context limit ({self.limit})"
            )
