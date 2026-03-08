"""Collect trajectories for training and benchmarking this environment."""

from __future__ import annotations

import argparse
import random
import uuid
from pathlib import Path
from typing import Dict, List, Sequence

from models import ActionType, ExperimentAction
from training.evaluation import EvaluationSuite
from training.trajectory import Trajectory, TrajectoryDataset
from training.gym_wrapper import BioExperimentGymEnv


HEURISTIC_SEQUENCE = [
    ActionType.COLLECT_SAMPLE,
    ActionType.PREPARE_LIBRARY,
    ActionType.SEQUENCE_CELLS,
    ActionType.RUN_QC,
    ActionType.FILTER_DATA,
    ActionType.NORMALIZE_DATA,
    ActionType.CLUSTER_CELLS,
    ActionType.TRAJECTORY_ANALYSIS,
    ActionType.MARKER_SELECTION,
    ActionType.SYNTHESIZE_CONCLUSION,
]


def action_space_index(action_type: ActionType) -> int:
    return list(ActionType).index(action_type)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run rollout episodes and persist trajectories."
    )
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes.")
    parser.add_argument(
        "--policy",
        choices=["random", "heuristic"],
        default="heuristic",
        help="Policy to use for rollouts.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional hard cutoff per episode (defaults to env limit).",
    )
    parser.add_argument(
        "--output-dir",
        default="training/rollouts",
        help="Directory for JSON trajectory outputs.",
    )
    parser.add_argument("--seed", type=int, default=None, help="RNG seed.")
    return parser.parse_args()


def heuristic_next_action(history: Sequence[ActionType], step_index: int) -> ActionType:
    seen = set(history)
    for action in HEURISTIC_SEQUENCE:
        if action not in seen:
            return action
    if step_index >= 2 and ActionType.VALIDATE_MARKER not in seen:
        return ActionType.VALIDATE_MARKER
    if ActionType.SYNTHESIZE_CONCLUSION in seen:
        return ActionType.SYNTHESIZE_CONCLUSION
    return ActionType.SYNTHESIZE_CONCLUSION


def pick_action(policy: str, step_index: int, history: Sequence[ActionType]) -> ActionType:
    if policy == "random":
        return random.choice(list(ActionType))
    return heuristic_next_action(history, step_index)


def run_episode(
    env: BioExperimentGymEnv,
    episode_id: str,
    policy: str,
    max_steps: int | None = None,
) -> Trajectory:
    observation, info = env.reset()
    structured_obs = info["structured_obs"]
    traj = Trajectory(
        episode_id=episode_id,
        task=structured_obs.task.model_dump(),
        metadata={
            "task_problem": structured_obs.task.problem_statement,
            "policy": policy,
        },
    )

    done = False
    step_num = 0
    while not done:
        if max_steps is not None and step_num >= max_steps:
            break

        history = [rec.action_type for rec in structured_obs.pipeline_history]
        action_type = pick_action(policy, step_num, history)
        action_idx = action_space_index(action_type)
        action = {
            "action_type": action_idx,
            "confidence": 0.75,
        }
        experiment_action = ExperimentAction(action_type=action_type, confidence=0.75)

        observation, reward, terminated, truncated, info = env.step(action)
        structured_obs = info["structured_obs"]
        done = terminated or truncated
        step_num += 1

        traj.add_step(
            action=experiment_action,
            observation=structured_obs,
            reward=reward,
            done=done,
            reward_breakdown=structured_obs.step_reward_breakdown,
        )

        print(
            f"  step={structured_obs.step_index:02d} "
            f"action={action_type.value:>28} "
            f"reward={reward:+.3f}"
        )

    return traj


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = BioExperimentGymEnv()
    trajectories: List[Trajectory] = []

    print(
        f"Starting rollout training script: episodes={args.episodes}, policy={args.policy}"
    )
    for ep in range(args.episodes):
        print(f"Episode {ep + 1}/{args.episodes}")
        traj = run_episode(
            env=env,
            episode_id=str(uuid.uuid4()),
            policy=args.policy,
            max_steps=args.max_steps,
        )
        traj.save(out_dir / f"{traj.episode_id}.json")
        trajectories.append(traj)

    dataset = TrajectoryDataset(trajectories)
    stats = EvaluationSuite.online_metrics(trajectories)

    print("\nRun complete.")
    print(f"Saved trajectories to: {out_dir}")
    print("Online metrics:")
    for metric in stats:
        print(f"  - {metric.name}: {metric.value:.4f}")

    print(f"Summary: {dataset.summary()}")


if __name__ == "__main__":
    main()
