from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


class PolicyError(RuntimeError):
    """Raised when an OpenVLA checkpoint cannot be initialized."""


def _single_file(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        raise PolicyError(f"Expected exactly one {pattern} in {root}, found {len(matches)}")
    return matches[0]


def _state_dict(path: Path):
    import torch

    state = torch.load(path, map_location="cpu", weights_only=True)
    return {key.removeprefix("module."): value for key, value in state.items()}


class OpenVLAOFTPolicy:
    """Read-only local loader for the public OpenVLA-OFT LIBERO checkpoint."""

    def __init__(self, model_cfg: dict[str, Any], suite_name: str) -> None:
        import torch
        from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
        from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
        from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
        from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
        from prismatic.models.action_heads import L1RegressionActionHead
        from prismatic.models.projectors import ProprioProjector

        if not torch.cuda.is_available():
            raise PolicyError("CUDA is required")
        torch.cuda.set_device(0)
        self.torch = torch
        self.checkpoint = Path(model_cfg["checkpoint"]).expanduser().resolve()
        if not self.checkpoint.is_dir():
            raise PolicyError(
                "Use a complete local checkpoint snapshot. Run scripts/prefetch_checkpoint.sh first."
            )
        registrations = (
            (AutoConfig.register, ("openvla", OpenVLAConfig)),
            (AutoImageProcessor.register, (OpenVLAConfig, PrismaticImageProcessor)),
            (AutoProcessor.register, (OpenVLAConfig, PrismaticProcessor)),
            (AutoModelForVision2Seq.register, (OpenVLAConfig, OpenVLAForActionPrediction)),
        )
        for function, arguments in registrations:
            try:
                function(*arguments)
            except ValueError:
                pass

        model = AutoModelForVision2Seq.from_pretrained(
            str(self.checkpoint),
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=False,
            local_files_only=True,
        )
        adapter_path = self.checkpoint / "adapter_config.json"
        if adapter_path.is_file():
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, str(self.checkpoint), is_trainable=False)
            model = model.merge_and_unload()
        model.vision_backbone.set_num_images_in_input(int(model_cfg.get("num_images", 2)))
        model.eval().to("cuda:0")
        self.model = model
        self.processor = AutoProcessor.from_pretrained(
            str(self.checkpoint), trust_remote_code=False, local_files_only=True
        )

        action_head = L1RegressionActionHead(
            input_dim=model.llm_dim,
            hidden_dim=model.llm_dim,
            action_dim=7,
        ).to(torch.bfloat16).to("cuda:0")
        action_head.load_state_dict(
            _state_dict(_single_file(self.checkpoint, "*action_head*checkpoint*.pt"))
        )
        action_head.eval()
        self.action_head = action_head

        proprio = ProprioProjector(llm_dim=model.llm_dim, proprio_dim=8)
        proprio.load_state_dict(
            _state_dict(_single_file(self.checkpoint, "*proprio_projector*checkpoint*.pt"))
        )
        proprio = proprio.to(torch.bfloat16).to("cuda:0").eval()
        self.proprio_projector = proprio

        stats = json.loads((self.checkpoint / "dataset_statistics.json").read_text())
        self.model.norm_stats = stats
        unnorm_key = suite_name if suite_name in stats else f"{suite_name}_no_noops"
        if unnorm_key not in stats:
            raise PolicyError(f"No normalization stats for {suite_name}; keys={sorted(stats)}")
        self.cfg = SimpleNamespace(
            model_family="openvla",
            pretrained_checkpoint=str(self.checkpoint),
            use_l1_regression=True,
            use_diffusion=False,
            use_proprio=True,
            use_film=False,
            num_images_in_input=int(model_cfg.get("num_images", 2)),
            center_crop=bool(model_cfg.get("center_crop", True)),
            unnorm_key=unnorm_key,
        )
        self.resize_size = 224

    @property
    def dummy_action(self) -> np.ndarray:
        return np.asarray([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)

    def prepare(
        self,
        raw_obs: dict[str, np.ndarray],
        *,
        agent_image: np.ndarray | None = None,
        wrist_image: np.ndarray | None = None,
        proprio: np.ndarray | None = None,
    ) -> dict[str, Any]:
        from experiments.robot.libero.libero_utils import (
            get_libero_image,
            get_libero_wrist_image,
            quat2axisangle,
        )
        from experiments.robot.openvla_utils import resize_image_for_policy

        obs = {key: np.asarray(value).copy() for key, value in raw_obs.items()}
        if agent_image is not None:
            obs["agentview_image"] = np.asarray(agent_image, dtype=np.uint8)
        if wrist_image is not None:
            obs["robot0_eye_in_hand_image"] = np.asarray(wrist_image, dtype=np.uint8)
        state = (
            np.asarray(proprio, dtype=np.float32)
            if proprio is not None
            else np.concatenate(
                (
                    obs["robot0_eef_pos"],
                    quat2axisangle(obs["robot0_eef_quat"]),
                    obs["robot0_gripper_qpos"],
                )
            ).astype(np.float32)
        )
        return {
            "full_image": resize_image_for_policy(get_libero_image(obs), self.resize_size),
            "wrist_image": resize_image_for_policy(get_libero_wrist_image(obs), self.resize_size),
            "state": state,
        }

    def query_prepared(self, prepared: dict[str, Any], instruction: str) -> np.ndarray:
        from experiments.robot.robot_utils import get_action

        with self.torch.no_grad():
            actions = get_action(
                self.cfg,
                self.model,
                prepared,
                instruction,
                processor=self.processor,
                action_head=self.action_head,
                proprio_projector=self.proprio_projector,
                noisy_action_projector=None,
                use_film=False,
            )
        array = np.asarray(actions, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != 7:
            raise PolicyError(f"Expected action chunk [T,7], got {array.shape}")
        return array

    def query(
        self,
        raw_obs: dict[str, np.ndarray],
        instruction: str,
        *,
        agent_image: np.ndarray | None = None,
        wrist_image: np.ndarray | None = None,
        proprio: np.ndarray | None = None,
    ) -> np.ndarray:
        return self.query_prepared(
            self.prepare(
                raw_obs,
                agent_image=agent_image,
                wrist_image=wrist_image,
                proprio=proprio,
            ),
            instruction,
        )

    def process_chunk(self, raw: np.ndarray) -> np.ndarray:
        from experiments.robot.robot_utils import invert_gripper_action, normalize_gripper_action

        output = []
        for action in np.asarray(raw, dtype=np.float32):
            converted = normalize_gripper_action(action, binarize=True)
            converted = invert_gripper_action(converted)
            output.append(converted)
        return np.asarray(output, dtype=np.float32)
