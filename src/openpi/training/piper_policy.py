"""Piper 末端位姿空间（EEPose）的 openpi 数据变换（PiperInputs / PiperOutputs）。

放置位置：把本文件拷进 openpi 仓库的 `src/openpi/policies/piper_policy.py`，
然后在 `src/openpi/training/config.py` 里 `import openpi.policies.piper_policy as piper_policy`
（详见 `backends/openpi/README.md` 的「在 openpi 上微调 Piper 模型」一节）。

约定（与本项目 LeRobot 数据集 / 真机部署一致）：
- 两路相机：第三视角 `observation/image`、腕部 `observation/wrist_image`，uint8 (H,W,3)。
- 状态为 **10 维**：`[x,y,z, r00,r10,r20, r01,r11,r21, gripper]`，**绝对末端位姿**（位置 + 6D旋转 + 夹爪）。
- 动作为 **10 维**：`[dx,dy,dz, dr00,dr10,dr20, dr01,dr11,dr21, gripper]`，**增量末端位姿** + 夹爪绝对值。
- pi05 模型内部按 32 维处理，这里 pad/截断在 32 维和 10 维之间转换。

inference 阶段 `deploy_policy.py` 下发的观测键名与下方 PiperInputs 期望完全一致：
`observation/image` / `observation/wrist_image` / `observation/state` / `prompt`。
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

# Piper 单臂：末端位姿 9 维（3 位置 + 6D 旋转）+ 1 夹爪 = 10 维。改本体时同步改这里和 PiperOutputs 的切片。
PIPER_ACTION_DIM = 10


def make_piper_example() -> dict:
    """构造随机输入样例（供 openpi 的 fake/调试管线使用）。"""
    return {
        "observation/state": np.random.rand(PIPER_ACTION_DIM),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "pick up the object and place it",
    }


def _parse_image(image) -> np.ndarray:
    """LeRobot 默认存 float32 (C,H,W)；统一转成 uint8 (H,W,C)。"""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class PiperInputs(transforms.DataTransformFn):
    """把 Piper 数据集 / 真机观测转换成模型期望的输入格式（训练 + 推理共用）。"""

    # 模型动作维度（pi05 为 32）。由 config 注入，请勿写死。
    action_dim: int

    # 决定使用哪个模型（PI0 / PI05 / PI0_FAST），由 config 注入，请勿改。
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # pi0 / pi05 才对 padding 相机做 mask；pi0-FAST 不 mask。
        mask_padding = self.model_type != _model.ModelType.PI0_FAST

        # 状态 pad 到模型动作维度（7 -> 32）。状态 = 绝对末端位姿。
        state = transforms.pad_to_dim(data["observation/state"], self.action_dim)

        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                # Piper 只有第三视角 + 腕部两路；右腕用零图补位。
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # 补位的右腕在 pi0/pi05 下 mask 掉。
                "right_wrist_0_rgb": np.False_ if mask_padding else np.True_,
            },
        }

        # 动作仅训练时存在；pad 到模型动作维度（7 -> 32）。
        if "actions" in data:
            inputs["actions"] = transforms.pad_to_dim(data["actions"], self.action_dim)

        # 语言指令；prompt_from_task=True 时由数据集 task 注入到 "prompt"。
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class PiperOutputs(transforms.DataTransformFn):
    """把模型输出截回 Piper 的 10 维增量末端位姿动作（仅推理用）。"""

    def __call__(self, data: dict) -> dict:
        # 训练时 pad 到 32 维，这里取回前 10 维 = [dx,dy,dz, dr00,dr10,dr20, dr01,dr11,dr21, gripper]。
        return {"actions": np.asarray(data["actions"][..., :PIPER_ACTION_DIM])}
