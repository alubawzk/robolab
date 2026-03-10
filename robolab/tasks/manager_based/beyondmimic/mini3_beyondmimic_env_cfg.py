
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

from robolab.assets.robots import MINI3_CFG
from robolab.tasks.manager_based.beyondmimic.beyondmimic_env_cfg import BeyondMimicEnvCfg

from isaaclab.utils import configclass
from robolab import ROBOLAB_ROOT_DIR

s_body_name = [
    'left_hip_yaw_link',
    'right_hip_yaw_link',
    'left_knee_pitch_link',
    'right_knee_pitch_link',
    'left_elbow_pitch_link',
    'right_elbow_pitch_link',
    'left_ankle_pitch_link',
    'right_ankle_pitch_link',
]

@configclass
class Mini3BeyondMimicEnvCfg(BeyondMimicEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = MINI3_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.commands.motion.motion_file = os.path.join(
            ROBOLAB_ROOT_DIR, "data", "motions", "mini3_lab", "mini3_new_walk.pkl"
        )
        self.commands.motion.anchor_body_name = "waist_yaw_link"
        self.commands.motion.body_names = [
            'left_hip_pitch_link',
            'right_hip_pitch_link',
            'waist_yaw_link',
            'left_hip_roll_link',
            'right_hip_roll_link',
            'left_shoulder_pitch_link',
            'right_shoulder_pitch_link',
            'left_hip_yaw_link',
            'right_hip_yaw_link',
            'left_shoulder_roll_link',
            'right_shoulder_roll_link',
            'left_knee_pitch_link',
            'right_knee_pitch_link',
            'left_shoulder_yaw_link',
            'right_shoulder_yaw_link',
            'left_ankle_pitch_link',
            'right_ankle_pitch_link',
            'left_elbow_pitch_link',
            'right_elbow_pitch_link',
            'left_ankle_roll_link',
            'right_ankle_roll_link',
        ]

        self.episode_length_s = 20.0
