"""Piper 的 openpi **JAX** 训练配置（全量 + LoRA，完全对齐官方 Libero）。

本文件提供 Piper 在 JAX 后端上的微调配置，与 openpi-official 原生 Libero 配置保持一致。

使用方法：
    python scripts/train.py pi05_piper              # 全量微调（JAX 后端）
    python scripts/train.py pi05_piper_lora         # LoRA 微调（JAX 后端）
    python scripts/train.py pi0_piper               # Pi0 全量微调
    python scripts/train.py pi0_piper_low_mem       # Pi0 LoRA 微调

前置条件：
- openpi 仓库保持主分支（与官方 Libero 配置兼容）。
- JAX 基座权重：gs://openpi-assets/checkpoints/pi05_base/params（官方 JAX 格式）。
- 数据集在 LeRobot 格式。

数据约定（与本项目 LeRobot 数据集一致）：
- 两路相机：`observation.images.cam_global`（第三视角）、`observation.images.cam_gripper`（腕部）。
- `observation.state` (10,) = `[x,y,z, r00,r10,r20, r01,r11,r21, gripper]`，绝对末端位姿。
- `action` (10,) = 绝对末端位姿；在训练时通过 `DeltaActions` 转换为增量。
"""

import dataclasses
import os

import numpy as np


def _sixd_to_mat(sixd):
    """6D rotation repr -> 3x3 rotation matrix (Zhou et al. 2019, Gram-Schmidt).

    Convention: sixd = [r00,r10,r20, r01,r11,r21] = first two columns of R.
    """
    c1 = np.asarray(sixd[0:3], dtype=np.float64)
    c2 = np.asarray(sixd[3:6], dtype=np.float64)
    b1 = c1 / (np.linalg.norm(c1) + 1e-8)
    c2 = c2 - np.dot(b1, c2) * b1
    b2 = c2 / (np.linalg.norm(c2) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=1)


def _mat_to_sixd(mat):
    """3x3 rotation matrix -> 6D repr (first two columns)."""
    return np.concatenate([mat[:, 0], mat[:, 1]]).astype(np.float32)


@dataclasses.dataclass(frozen=True)
class PiperRelRotDeltaActions:
    """Training input side: absolute EE pose -> delta (relative-rotation variant).

    - position(0:3): linear subtraction action - state.
    - 6D rotation(3:9): relative rotation R_rel = R_state^T @ R_action (body frame).
    - gripper(9): kept absolute.
    Reference state is current obs (first frame of chunk), broadcast over the chunk.
    """

    def __call__(self, data: dict) -> dict:
        if "actions" not in data:
            return data
        state = np.asarray(data["state"])
        actions = np.asarray(data["actions"]).copy()
        actions[..., 0:3] -= state[0:3]
        r_s = _sixd_to_mat(state[3:9])
        flat = actions.reshape(-1, actions.shape[-1])
        for i in range(flat.shape[0]):
            r_rel = r_s.T @ _sixd_to_mat(flat[i, 3:9])
            flat[i, 3:9] = _mat_to_sixd(r_rel)
        data["actions"] = flat.reshape(actions.shape)
        return data


@dataclasses.dataclass(frozen=True)
class PiperRelRotAbsoluteActions:
    """Inference output side: delta -> absolute EE pose (inverse of the above).

    - position(0:3): action + state.
    - 6D rotation(3:9): R_action = R_state @ R_rel.
    - gripper(9): kept absolute.
    """

    def __call__(self, data: dict) -> dict:
        if "actions" not in data:
            return data
        state = np.asarray(data["state"])
        actions = np.asarray(data["actions"]).copy()
        actions[..., 0:3] += state[0:3]
        r_s = _sixd_to_mat(state[3:9])
        flat = actions.reshape(-1, actions.shape[-1])
        for i in range(flat.shape[0]):
            r_a = r_s @ _sixd_to_mat(flat[i, 3:9])
            flat[i, 3:9] = _mat_to_sixd(r_a)
        data["actions"] = flat.reshape(actions.shape)
        return data


def get_piper_configs():
    """返回 Piper 的 JAX 训练配置列表。
    
    四个配置都走 openpi JAX 后端（scripts/train.py）：
    - pi05_piper            ：全量微调（显存需求高）。
    - pi05_piper_lora       ：LoRA 微调（显存节省）。
    - pi0_piper             ：Pi0 全量微调。
    - pi0_piper_low_mem     ：Pi0 LoRA 微调。
    """
    # 延迟导入：避免循环导入 config.py
    from typing_extensions import override
    
    import openpi.models.pi0_config as pi0_config
    import openpi.policies.piper_policy as piper_policy
    import openpi.training.optimizer as _optimizer
    import openpi.transforms as _transforms
    from openpi.training import weight_loaders
    from openpi.training.config import (
        DataConfig,
        DataConfigFactory,
        ModelTransformFactory,
        TrainConfig,
    )

    # 你的 LeRobot 数据集 repo id
    PIPER_REPO_ID = os.environ.get(
        "PIPER_LEROBOT_REPO_ID", os.environ.get("PIPER_REPO_ID", "<your_hf_username>/piper_pickplace")
    )
    PIPER_ACTION_HORIZON = 10
    PI05_ACTION_DIM = 32

    @dataclasses.dataclass(frozen=True)
    class LeRobotPiperDataConfig(DataConfigFactory):
        """Piper LeRobot 数据集到 JAX 训练管线的适配器。
        
        action 在数据集中是绝对末端位姿，在训练时显式转换为增量。
        """

        # action_space: "delta" (default) | "absolute" | "relrot".
        #   delta    : element-wise subtraction on first 9 dims (pos3+6D rot), gripper absolute.
        #   absolute : no transform; model learns absolute 6D EE pose directly.
        #   relrot   : linear position delta + relative rotation R_state^T @ R_action, gripper absolute.
        action_space: str = "delta"

        @override
        def create(self, assets_dirs, model_config):
            # repack：把数据集列名映射成 PiperInputs 期望的键
            repack_transform = _transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "observation/image": "observation.images.cam_global",
                            "observation/wrist_image": "observation.images.cam_gripper",
                            "observation/state": "observation.state",
                            "actions": "action",
                            "prompt": "prompt",
                        }
                    )
                ]
            )

            data_transforms = _transforms.Group(
                inputs=[
                    piper_policy.PiperInputs(
                        action_dim=model_config.action_dim,
                        model_type=model_config.model_type,
                    )
                ],
                outputs=[piper_policy.PiperOutputs()],
            )
            # DeltaActions/AbsoluteActions 必须在 PiperInputs 之后（用 push 追加，与官方 libero 一致）：
            # PiperInputs 先产出扁平的 state/actions 键，DeltaActions 才能取到 data["state"]/["actions"]。
            # 前 9 维（pos3 + 6D 旋转）转增量，第 10 维夹爪保持绝对；输出侧对称做 AbsoluteActions 还原。
            if self.action_space == "delta":
                data_transforms = data_transforms.push(
                    inputs=[_transforms.DeltaActions(_transforms.make_bool_mask(9, -1))],
                    outputs=[_transforms.AbsoluteActions(_transforms.make_bool_mask(9, -1))],
                )
            elif self.action_space == "relrot":
                data_transforms = data_transforms.push(
                    inputs=[PiperRelRotDeltaActions()],
                    outputs=[PiperRelRotAbsoluteActions()],
                )
            elif self.action_space == "absolute":
                pass  # learn absolute 6D EE pose directly, no delta/absolute transform
            else:
                raise ValueError(
                    f"unknown action_space={self.action_space!r}; expected delta/absolute/relrot"
                )

            model_transforms = ModelTransformFactory()(model_config)

            return dataclasses.replace(
                self.create_base_config(assets_dirs, model_config),
                repack_transforms=repack_transform,
                data_transforms=data_transforms,
                model_transforms=model_transforms,
                # 数据集里动作列名是 `action`（单数，LeRobot 标准名）；DataConfig 默认
                # action_sequence_keys=("actions",) 会让 delta_timestamps 去查不存在的
                # `actions` 列而报 KeyError，这里显式对齐成 ("action",)。
                action_sequence_keys=("action",),
            )

    return [
        # ===== Pi05 全量微调 =====
        TrainConfig(
            name="pi05_piper",
            model=pi0_config.Pi0Config(
                pi05=True,
                action_dim=PI05_ACTION_DIM, # 32 means 3D pos + 6D rot + 1D gripper
                action_horizon=PIPER_ACTION_HORIZON, # 10 means 10 steps of actions
            ),
            data=LeRobotPiperDataConfig(
                repo_id=PIPER_REPO_ID,
                base_config=DataConfig(prompt_from_task=True),
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(
                "gs://openpi-assets/checkpoints/pi05_base/params"
            ),
            optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=1_000,
                peak_lr=5e-5,
                decay_steps=1_000_000,
                decay_lr=5e-5,
            ),
            ema_decay=0.999,
            batch_size=256,
            num_train_steps=30_000,
        ),
        # ===== Pi05 LoRA 微调 =====
        TrainConfig(
            name="pi05_piper_lora",
            model=pi0_config.Pi0Config(
                pi05=True,
                action_dim=PI05_ACTION_DIM, # 32 means 3D pos + 6D rot + 1D gripper padding to 32 input dim
                action_horizon=PIPER_ACTION_HORIZON, # 10 means 10 steps of actions
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
            ),
            data=LeRobotPiperDataConfig(
                repo_id=PIPER_REPO_ID,
                base_config=DataConfig(prompt_from_task=True),
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(
                "gs://openpi-assets/checkpoints/pi05_base/params"
            ),
            optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=1_000,
                peak_lr=1e-4,
                decay_steps=1_000_000,
                decay_lr=1e-5,
            ),
            ema_decay=None,
            batch_size=16,
            num_train_steps=100_000,
            freeze_filter=pi0_config.Pi0Config(
                pi05=True,
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
            ).get_freeze_filter(),
        ),
        # ===== Pi05 LoRA - comparison A: absolute 6D (no delta) =====
        TrainConfig(
            name="pi05_piper_lora_absolute",
            model=pi0_config.Pi0Config(
                pi05=True,
                action_dim=PI05_ACTION_DIM,
                action_horizon=PIPER_ACTION_HORIZON,
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
            ),
            data=LeRobotPiperDataConfig(
                repo_id=PIPER_REPO_ID,
                base_config=DataConfig(prompt_from_task=True),
                action_space="absolute",
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(
                "gs://openpi-assets/checkpoints/pi05_base/params"
            ),
            optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=1_000,
                peak_lr=1e-4,
                decay_steps=1_000_000,
                decay_lr=1e-5,
            ),
            ema_decay=None,
            batch_size=16,
            num_train_steps=300_000,
            freeze_filter=pi0_config.Pi0Config(
                pi05=True,
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
            ).get_freeze_filter(),
        ),
        # ===== Pi05 LoRA - comparison B: relative-rotation delta (R_state^T @ R_action) =====
        TrainConfig(
            name="pi05_piper_lora_relrot",
            model=pi0_config.Pi0Config(
                pi05=True,
                action_dim=PI05_ACTION_DIM,
                action_horizon=PIPER_ACTION_HORIZON,
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
            ),
            data=LeRobotPiperDataConfig(
                repo_id=PIPER_REPO_ID,
                base_config=DataConfig(prompt_from_task=True),
                action_space="relrot",
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(
                "gs://openpi-assets/checkpoints/pi05_base/params"
            ),
            optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=1_000,
                peak_lr=1e-4,
                decay_steps=1_000_000,
                decay_lr=1e-5,
            ),
            ema_decay=None,
            batch_size=16,
            num_train_steps=30_000,
            freeze_filter=pi0_config.Pi0Config(
                pi05=True,
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
            ).get_freeze_filter(),
        ),
        # ===== Pi0 全量微调 =====
        TrainConfig(
            name="pi0_piper",
            model=pi0_config.Pi0Config(
                action_dim=PI05_ACTION_DIM,
                action_horizon=PIPER_ACTION_HORIZON,
            ),
            data=LeRobotPiperDataConfig(
                repo_id=PIPER_REPO_ID,
                base_config=DataConfig(prompt_from_task=True),
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(
                "gs://openpi-assets/checkpoints/pi0_base/params"
            ),
            optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=1_000,
                peak_lr=5e-5,
                decay_steps=1_000_000,
                decay_lr=5e-5,
            ),
            ema_decay=0.999,
            batch_size=256,
            num_train_steps=30_000,
        ),
        # ===== Pi0 LoRA 微调 =====
        TrainConfig(
            name="pi0_piper_low_mem",
            model=pi0_config.Pi0Config(
                action_dim=PI05_ACTION_DIM,
                action_horizon=PIPER_ACTION_HORIZON,
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
            ),
            data=LeRobotPiperDataConfig(
                repo_id=PIPER_REPO_ID,
                base_config=DataConfig(prompt_from_task=True),
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(
                "gs://openpi-assets/checkpoints/pi0_base/params"
            ),
            optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=1_000,
                peak_lr=1e-4,
                decay_steps=1_000_000,
                decay_lr=1e-5,
            ),
            ema_decay=None,
            batch_size=64,
            num_train_steps=30_000,
            freeze_filter=pi0_config.Pi0Config(
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
            ).get_freeze_filter(),
        ),
    ]
