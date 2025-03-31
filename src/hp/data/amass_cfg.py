# Partially taken from https://github.com/cvlab-epfl/adv_param_pose_prior/blob/main/lib/datasets/amass.py

from enum import Enum
from typing import List

AmassDatasets = Enum(
    "AmassDatasets",
    [
        "ACCAD",
        "CMU",
        "KIT",
        "EyesJapanDataset",
        "WEIZMANN",
        "BMLrub",
        "BMLmovi",
        "SFU",
        "TotalCapture",
        "Transitions",
        "EKUT",
        "HDM05",
        "HumanEva",
        "MoSh",
        "SSM",
        "TCDHands",
        "PosePrior",
    ],
    start=0,
)
AmassJoints = Enum(
    "AmassJoints",
    [
        "ROOT",
        "LeftUpLeg",
        "RightUpLeg",
        "Spine0",
        "LeftLeg",
        "RightLeg",
        "Spine1",
        "LeftFoot",
        "RightFoot",
        "Spine2",
        "LeftToeBase",
        "RightToeBase",
        "Neck",
        "LeftShoulder",
        "RightShoulder",
        "Head",
        "LeftArm",
        "RightArm",
        "LeftElbow",
        "RightElbow",
        "LeftHand",
        "RightHand",
    ],
    start=0,
)
AmassJoints4Learning: List[AmassJoints] = [
    aj for aj in AmassJoints if aj.value not in [0, 10, 11]
]

ORIGINAL_AMASS_SPLITS = {
    "valid": ["HumanEva", "HDM05", "SFU", "MoSh"],  # 117679 samples
    "test": ["Transitions", "SSM"],  # 1320126 samples
    "train": [
        "CMU",
        "PosePrior",
        "TotalCapture",
        "EyesJapanDataset",
        "KIT",
        "BMLrub",
        "EKUT",
        "TCDHands",
        "ACCAD",
    ],  # 14779964 samples
}
