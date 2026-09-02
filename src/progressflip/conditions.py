from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

ViewMode = Literal[
    "true",
    "full_old",
    "robot_old",
    "nonrobot_old",
    "eef_old",
    "arm_old",
    "robot_mask",
    "robot_random",
    "robot_phase25",
    "robot_phase50",
    "robot_phase75",
]
InstructionMode = Literal["original", "remaining", "explicit_progress", "incorrect"]
ActionSource = Literal["query", "true", "stale"]


@dataclass(frozen=True)
class Condition:
    name: str
    view: ViewMode = "true"
    instruction: InstructionMode = "original"
    action_source: ActionSource = "query"
    execute_steps: int = 1
    action_scale: float = 1.0
    reset_after_first: bool = False
    requery_true_after_first: bool = True

    def validate(self) -> None:
        if self.execute_steps < 0:
            raise ValueError(f"{self.name}: execute_steps must be non-negative")
        if not 0.0 <= self.action_scale <= 1.0:
            raise ValueError(f"{self.name}: action_scale must be in [0, 1]")
        if self.action_source in {"true", "stale"} and self.execute_steps == 0:
            raise ValueError(f"{self.name}: a forced action source requires execute_steps > 0")


def registry() -> dict[str, Condition]:
    conditions: list[Condition] = [
        Condition("TRUE_K1"),
        Condition("TRUE_RECOMPOSED_K1", view="true"),
        Condition("FULL_OLD_K1", view="full_old"),
        Condition("ROBOT_OLD_K1", view="robot_old"),
        Condition("NONROBOT_OLD_K1", view="nonrobot_old"),
        Condition("EEF_OLD_K1", view="eef_old"),
        Condition("ARM_OLD_K1", view="arm_old"),
        Condition("ROBOT_MASK_K1", view="robot_mask"),
        Condition("ROBOT_RANDOM_K1", view="robot_random"),
        Condition("ROBOT_PHASE25_K1", view="robot_phase25"),
        Condition("ROBOT_PHASE50_K1", view="robot_phase50"),
        Condition("ROBOT_PHASE75_K1", view="robot_phase75"),
        Condition("STALE_K0", view="full_old", execute_steps=0),
        Condition("STALE_K1", view="full_old"),
        Condition("STALE_TRUE_ACTION", view="full_old", action_source="true"),
        Condition("TRUE_STALE_ACTION", view="true", action_source="stale"),
        Condition("STALE_K1_RESET", view="full_old", reset_after_first=True),
        Condition("STALE_SCALE_010", view="full_old", action_source="stale", action_scale=0.10),
        Condition("STALE_SCALE_025", view="full_old", action_source="stale", action_scale=0.25),
        Condition("STALE_SCALE_050", view="full_old", action_source="stale", action_scale=0.50),
        Condition("STALE_SCALE_075", view="full_old", action_source="stale", action_scale=0.75),
        Condition("STALE_SCALE_100", view="full_old", action_source="stale", action_scale=1.00),
        Condition("ORIGINAL_INSTRUCTION", view="full_old", instruction="original"),
        Condition("REMAINING_INSTRUCTION", view="full_old", instruction="remaining"),
        Condition("EXPLICIT_PROGRESS", view="full_old", instruction="explicit_progress"),
        Condition("INCORRECT_SUBGOAL", view="full_old", instruction="incorrect"),
    ]
    output = {condition.name: condition for condition in conditions}
    for condition in output.values():
        condition.validate()
    return output


def resolve(names: list[str]) -> list[Condition]:
    available = registry()
    missing = [name for name in names if name not in available]
    if missing:
        raise KeyError(f"Unknown conditions: {missing}; available={sorted(available)}")
    return [available[name] for name in names]


def with_execute_steps(condition: Condition, steps: int) -> Condition:
    updated = replace(condition, execute_steps=steps)
    updated.validate()
    return updated
