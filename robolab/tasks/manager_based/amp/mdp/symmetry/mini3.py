# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Symmetry functions for the Mini3 robot.

Mini3 joint order (Isaac Lab, 21 joints):
  0: left_hip_pitch_joint
  1: right_hip_pitch_joint
  2: waist_yaw_joint
  3: left_hip_roll_joint
  4: right_hip_roll_joint
  5: left_shoulder_pitch_joint
  6: right_shoulder_pitch_joint
  7: left_hip_yaw_joint
  8: right_hip_yaw_joint
  9: left_shoulder_roll_joint
 10: right_shoulder_roll_joint
 11: left_knee_pitch_joint
 12: right_knee_pitch_joint
 13: left_shoulder_yaw_joint
 14: right_shoulder_yaw_joint
 15: left_ankle_pitch_joint
 16: right_ankle_pitch_joint
 17: left_elbow_pitch_joint
 18: right_elbow_pitch_joint
 19: left_ankle_roll_joint
 20: right_ankle_roll_joint
"""

from __future__ import annotations

import torch
from tensordict import TensorDict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["compute_symmetric_states"]

# left indices: [0, 3, 5, 7, 9, 11, 13, 15, 17, 19]
# right indices: [1, 4, 6, 8, 10, 12, 14, 16, 18, 20]
_LEFT  = [0, 3, 5, 7, 9, 11, 13, 15, 17, 19]
_RIGHT = [1, 4, 6, 8, 10, 12, 14, 16, 18, 20]
# joints whose sign flips under left-right mirror (yaw + roll + waist_yaw)
_NEGATE = [2, 3, 4, 7, 8, 9, 10, 13, 14, 19, 20]


@torch.no_grad()
def compute_symmetric_states(
    env: ManagerBasedRLEnv,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
):
    if obs is not None:
        batch_size = obs.batch_size[0]
        obs_aug = obs.repeat(2)
        obs_aug["policy"][:batch_size] = obs["policy"][:]
        obs_aug["policy"][batch_size:] = _transform_policy_obs_left_right(env.unwrapped, obs["policy"])
        obs_aug["critic"][:batch_size] = obs["critic"][:]
        obs_aug["critic"][batch_size:] = _transform_critic_obs_left_right(env.unwrapped, obs["critic"])
    else:
        obs_aug = None

    if actions is not None:
        batch_size = actions.shape[0]
        actions_aug = torch.zeros(batch_size * 2, actions.shape[1], device=actions.device)
        actions_aug[:batch_size] = actions[:]
        actions_aug[batch_size:] = _transform_actions_left_right(actions)
    else:
        actions_aug = None

    return obs_aug, actions_aug


def _transform_policy_obs_left_right(env: ManagerBasedRLEnv, obs: torch.Tensor) -> torch.Tensor:
    # policy obs layout (21 joints):
    # [0:3]   ang_vel
    # [3:6]   projected_gravity
    # [6:9]   velocity_commands
    # [9:30]  joint_pos
    # [30:51] joint_vel
    # [51:72] last_actions
    obs = obs.clone()
    device = obs.device
    obs[:, 0:3] = obs[:, 0:3] * torch.tensor([-1, 1, -1], device=device)   # ang_vel
    obs[:, 3:6] = obs[:, 3:6] * torch.tensor([1, -1, 1], device=device)    # gravity
    obs[:, 6:9] = obs[:, 6:9] * torch.tensor([1, -1, -1], device=device)   # vel_cmd
    obs[:, 9:30]  = _switch_joints_left_right(obs[:, 9:30])                 # joint_pos
    obs[:, 30:51] = _switch_joints_left_right(obs[:, 30:51])                # joint_vel
    obs[:, 51:72] = _switch_joints_left_right(obs[:, 51:72])                # last_actions
    return obs


def _transform_critic_obs_left_right(env: ManagerBasedRLEnv, obs: torch.Tensor) -> torch.Tensor:
    # critic obs layout (21 joints):
    # [0:3]   lin_vel
    # [3:6]   ang_vel
    # [6:9]   projected_gravity
    # [9:12]  velocity_commands
    # [12:33] joint_pos
    # [33:54] joint_vel
    # [54:75] last_actions
    obs = obs.clone()
    device = obs.device
    obs[:, 0:3] = obs[:, 0:3] * torch.tensor([1, -1, 1], device=device)    # lin_vel
    obs[:, 3:6] = obs[:, 3:6] * torch.tensor([-1, 1, -1], device=device)   # ang_vel
    obs[:, 6:9] = obs[:, 6:9] * torch.tensor([1, -1, 1], device=device)    # gravity
    obs[:, 9:12] = obs[:, 9:12] * torch.tensor([1, -1, -1], device=device)  # vel_cmd
    obs[:, 12:33] = _switch_joints_left_right(obs[:, 12:33])                # joint_pos
    obs[:, 33:54] = _switch_joints_left_right(obs[:, 33:54])                # joint_vel
    obs[:, 54:75] = _switch_joints_left_right(obs[:, 54:75])                # last_actions
    return obs


def _transform_actions_left_right(actions: torch.Tensor) -> torch.Tensor:
    actions = actions.clone()
    actions[:] = _switch_joints_left_right(actions[:])
    return actions


def _switch_joints_left_right(joint_data: torch.Tensor) -> torch.Tensor:
    """Swap left<->right joints and negate yaw/roll joints."""
    out = joint_data.clone()
    out[..., _LEFT]  = joint_data[..., _RIGHT]
    out[..., _RIGHT] = joint_data[..., _LEFT]
    out[..., _NEGATE] *= -1
    return out
