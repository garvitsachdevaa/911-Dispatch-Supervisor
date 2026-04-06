"""Tests for the dispatch state machine."""

from __future__ import annotations

from src.city_schema import CitySchemaLoader
from src.models import Action, DispatchAction
from src.state_machine import DispatchStateMachine


def test_reset_sets_ids_and_has_entities() -> None:
    schema = CitySchemaLoader.load("metro_city")
    sm = DispatchStateMachine(schema=schema, seed=42)
    state = sm.reset(task_id="single_incident", episode_id="ep-1")
    assert state.task_id == "single_incident"
    assert state.episode_id == "ep-1"
    assert state.step_count == 0
    assert state.units
    assert state.incidents


def test_legal_actions_non_empty_initially() -> None:
    schema = CitySchemaLoader.load("metro_city")
    sm = DispatchStateMachine(schema=schema, seed=42)
    state = sm.reset(task_id="single_incident", episode_id="ep-1")
    legal = sm.get_legal_actions(state)
    assert legal
    assert any(a.action_type == DispatchAction.DISPATCH for a in legal)


def test_additional_actions_become_reachable() -> None:
    schema = CitySchemaLoader.load("metro_city")
    sm = DispatchStateMachine(schema=schema, seed=42)

    # Multi-incident is a better reachability surface (multiple incidents + P2 incident).
    state = sm.reset(task_id="multi_incident", episode_id="ep-1")
    legal = sm.get_legal_actions(state)

    assert any(a.action_type == DispatchAction.STAGE for a in legal)
    assert any(a.action_type == DispatchAction.UPGRADE for a in legal)
    assert any(a.action_type == DispatchAction.DOWNGRADE for a in legal)

    # After a dispatch, REASSIGN should be legal to the other active incident.
    dispatch = next(a for a in legal if a.action_type == DispatchAction.DISPATCH)
    state, _ = sm.step(state, dispatch)
    legal2 = sm.get_legal_actions(state)
    assert any(a.action_type == DispatchAction.CANCEL for a in legal2)
    assert any(a.action_type == DispatchAction.REASSIGN for a in legal2)


def test_mutual_aid_appears_when_type_exhausted() -> None:
    schema = CitySchemaLoader.load("metro_city")
    sm = DispatchStateMachine(schema=schema, seed=42)
    state = sm.reset(task_id="multi_incident", episode_id="ep-1")

    # Exhaust all MEDIC availability.
    from src.models import UnitStatus

    for unit in state.units.values():
        if unit.unit_type.value == "MEDIC":
            unit.status = UnitStatus.DISPATCHED
            unit.assigned_incident_id = "INC-001"

    legal = sm.get_legal_actions(state)
    assert any(a.action_type == DispatchAction.MUTUAL_AID for a in legal)


def test_invalid_action_yields_protocol_ok_false() -> None:
    schema = CitySchemaLoader.load("metro_city")
    sm = DispatchStateMachine(schema=schema, seed=42)
    state = sm.reset(task_id="single_incident", episode_id="ep-1")
    bad = Action(action_type=DispatchAction.DISPATCH, unit_id="NOPE", incident_id="INC-001")
    state2, obs = sm.step(state, bad)
    assert obs.protocol_ok is False
    assert obs.result == "invalid action"
    assert state2.step_count == 1


def test_dispatch_progresses_incident() -> None:
    schema = CitySchemaLoader.load("metro_city")
    sm = DispatchStateMachine(schema=schema, seed=42)
    state = sm.reset(task_id="single_incident", episode_id="ep-1")
    legal = sm.get_legal_actions(state)
    state, obs = sm.step(state, legal[0])
    assert obs.protocol_ok is True
    assert any(u.status.value != "AVAILABLE" for u in state.units.values())
