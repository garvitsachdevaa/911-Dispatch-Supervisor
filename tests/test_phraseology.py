"""Tests for dispatch phraseology renderer/judge."""

from __future__ import annotations

from src.models import Action, DispatchAction
from src.phraseology import PhraseologyJudge, PhraseologyRenderer


def test_dispatch_renders() -> None:
    renderer = PhraseologyRenderer()
    action = Action(action_type=DispatchAction.DISPATCH, unit_id="MED-1", incident_id="INC-001")
    assert renderer.render(action) == "DISPATCH MED-1 -> INC-001"


def test_cancel_renders() -> None:
    renderer = PhraseologyRenderer()
    action = Action(action_type=DispatchAction.CANCEL, unit_id="MED-1", incident_id="INC-001")
    assert renderer.render(action) == "CANCEL MED-1 FROM INC-001"


def test_exact_match_scores_one() -> None:
    judge = PhraseologyJudge()
    action = Action(action_type=DispatchAction.DISPATCH, unit_id="MED-1", incident_id="INC-001")
    candidate = "DISPATCH MED-1 -> INC-001"
    assert judge.score(action, candidate) == 1.0


def test_readback_needs_unit_and_incident() -> None:
    judge = PhraseologyJudge()
    action = Action(action_type=DispatchAction.DISPATCH, unit_id="MED-1", incident_id="INC-001")
    assert judge.check_readback("DISPATCH MED-1 -> INC-001", action) is True
    assert judge.check_readback("DISPATCH MED-1", action) is False
