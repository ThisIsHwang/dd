import json
from pathlib import Path

import numpy as np

from progressflip.actions import QueryBank, scale_action, select_chunk
from progressflip.analysis import analyze
from progressflip.conditions import registry
from progressflip.records import PairRecord, validate_pair, write_pair
from progressflip.vision import Capture, GeomGroups, build_composites, geom_mask, image_mae


def _capture(robot_slice: slice, robot_id: int = 7):
    rgb = np.zeros((16, 16, 3), dtype=np.uint8)
    rgb[..., 1] = 80
    rgb[4:12, robot_slice, :] = np.asarray([220, 20, 20], dtype=np.uint8)
    seg = np.zeros((16, 16, 2), dtype=np.int32)
    seg[4:12, robot_slice, 0] = 5
    seg[4:12, robot_slice, 1] = robot_id
    return Capture(rgb=rgb, segmentation=seg)


def test_condition_registry_contains_causal_controls():
    required = {
        "TRUE_RECOMPOSED_K1",
        "ROBOT_OLD_K1",
        "NONROBOT_OLD_K1",
        "STALE_K0",
        "STALE_TRUE_ACTION",
        "TRUE_STALE_ACTION",
        "STALE_K1_RESET",
        "REMAINING_INSTRUCTION",
    }
    assert required.issubset(registry())


def test_action_scaling_and_crossover():
    action = np.asarray([1, 2, 3, 4, 5, 6, -1], dtype=np.float32)
    scaled = scale_action(action, 0.25)
    np.testing.assert_allclose(scaled[:6], action[:6] * 0.25)
    assert scaled[6] == -1
    bank = QueryBank(
        true_raw=np.zeros((2, 7)),
        stale_raw=np.ones((2, 7)),
        true_env=np.full((2, 7), 2.0),
        stale_env=np.full((2, 7), 3.0),
    )
    np.testing.assert_array_equal(select_chunk(bank, "true", np.full((2, 7), 4.0)), 2.0)
    np.testing.assert_array_equal(select_chunk(bank, "stale", np.full((2, 7), 4.0)), 3.0)


def test_segmentation_composite_has_exact_true_control():
    current = _capture(slice(10, 13))
    old = _capture(slice(2, 5))
    groups = GeomGroups.from_ids([7], [7])
    assert int(geom_mask(current.segmentation, [7]).sum()) == 24
    bundle = build_composites(current, old, groups, dilation_px=1)
    assert image_mae(bundle.true, bundle.true_recomposed) == 0.0
    assert not np.array_equal(bundle.robot_old, current.rgb)


def test_pair_roundtrip(tmp_path: Path):
    root = tmp_path / "pair"
    metadata = {
        "pair_id": "task--init000",
        "task_key": "task",
        "task_id": 1,
        "instruction": "do the task",
        "remaining_instruction": "finish it",
        "explicit_progress_instruction": "the first step is done; finish it",
        "incorrect_instruction": "repeat the first step",
        "target_object": "object_1",
        "target_predicate": ["in", "object_1", "region_1"],
    }
    write_pair(
        root,
        metadata,
        initial_state=np.zeros(9),
        prefix_actions=np.zeros((3, 7)),
        trigger_state=np.ones(9),
        object_advanced_state=np.full(9, 2),
        endpoint_state=np.full(9, 3),
        phase_states={25: np.full(9, 4), 50: np.full(9, 5), 75: np.full(9, 6)},
    )
    validate_pair(root)
    pair = PairRecord.load(root)
    assert pair.instruction_for("remaining") == "finish it"
    np.testing.assert_array_equal(pair.phase_states[50], 5)


def test_paired_analysis(tmp_path: Path):
    output = tmp_path / "run"
    results = output / "results"
    results.mkdir(parents=True)
    manifest = []
    records = []
    for index in range(4):
        for condition in ("A", "B"):
            pair_id = f"p{index}"
            job_id = f"{pair_id}--{condition}"
            manifest.append({"job_id": job_id})
            records.append(
                {
                    "job_id": job_id,
                    "pair_id": pair_id,
                    "condition": condition,
                    "completed": True,
                    "valid": True,
                    "success": condition == "A" or index == 3,
                    "timeout": False,
                    "target_progress_preserved": True,
                    "any_new_goal_achieved": condition == "A",
                    "query_to_true_chunk_mae": 0.1,
                    "query_to_stale_chunk_mae": 0.2,
                }
            )
    (output / "manifest.jsonl").write_text("".join(json.dumps(row) + "\n" for row in manifest))
    (results / "rank000.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
    cfg = {
        "data": {"output_root": str(output)},
        "experiment": {
            "conditions": ["A", "B"],
            "confirmatory_contrasts": [{"a": "A", "b": "B"}],
            "bootstrap_samples": 200,
            "analysis_seed": 1,
        },
    }
    summary = analyze(cfg)
    assert summary["confirmatory_contrasts"][0]["difference_a_minus_b"] == 0.75
    assert summary["inference_gate_passed"]
