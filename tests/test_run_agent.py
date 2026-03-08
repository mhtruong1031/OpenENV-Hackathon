"""Tests for run_agent parser and fallback helpers."""

from models import ActionType, ExperimentAction
from run_agent import fallback_action, parse_action
from server.hackathon_environment import BioExperimentEnvironment


def test_parse_action_accepts_reasoning_variant():
    action = parse_action(
        '{"action_type":"run_qc","parameters":{},"Reasoning":"check quality","confidence":0.8}'
    )
    assert action is not None
    assert action.action_type == ActionType.RUN_QC
    assert action.justification == "check quality"


def test_parse_action_accepts_justifyement_typo():
    action = parse_action(
        '{"action_type":"collect_sample","parameters":{},"justifyement":"typo key","confidence":0.7}'
    )
    assert action is not None
    assert action.action_type == ActionType.COLLECT_SAMPLE
    assert action.justification == "typo key"


def test_fallback_uses_observation_progress_not_step_index():
    env = BioExperimentEnvironment(scenario_name="cardiac_disease_de", domain_randomise=False)
    obs = env.reset(seed=0)
    for action_type in (
        ActionType.COLLECT_SAMPLE,
        ActionType.PREPARE_LIBRARY,
        ActionType.SEQUENCE_CELLS,
    ):
        obs = env.step(ExperimentAction(action_type=action_type))
    action = fallback_action(obs)
    assert action.action_type == ActionType.RUN_QC
