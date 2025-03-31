from omegaconf import OmegaConf
import torch
import numpy as np
from typing import List
from hp.data.amass_cfg import AmassJoints
from hp.utils.smpl import SmplSkeleton, relative2absolute_joint_coordinates
from hp.utils.rotation_tools import RotationConverterTorchRoma as RotationConverterTorch


class SkelForwardKinematics:

    def __init__(self) -> None:
        self.skeleton = SmplSkeleton()
        self.parents = list(self.skeleton.parent.values())
        self.children = None

    def forward(self, poses_all: torch.tensor) -> torch.tensor:
        N_poses = poses_all.shape[0]
        step = 1_000_000
        p3d_rel = torch.zeros((N_poses, 52, 3)).to(poses_all.device)
        for i in range(0, N_poses, step):
            p3d_rel[i : i + step] = self.skeleton.poses_2_3d(poses_all[i : i + step])
        p3d = relative2absolute_joint_coordinates(
            p3d_rel, self.parents, get_body_pose_related_joints=True
        )
        return p3d

    def get_full_parent_map(self):
        def get_all_parents(joint, parent_map):
            parents = []
            while joint in parent_map:
                joint = parent_map[joint]
                if joint not in [-1, 0]:
                    parents.append(joint)
            return parents

        parents_dict = {i: p for i, p in enumerate(self.parents)}
        full_parent_map = {
            joint: get_all_parents(joint, parents_dict)
            for joint in parents_dict
            if joint <= 21 and joint != 0
        }
        return full_parent_map


def perform_forward_kinematics(
    poses: torch.tensor, selected_joints: List[AmassJoints] = None
) -> torch.tensor:
    s_fk = SkelForwardKinematics()
    bs = poses.shape[0]
    num_param_for_aa_representation_smplx = 165
    poses_all = torch.zeros((bs, num_param_for_aa_representation_smplx)).to(
        poses.device
    )
    poses = RotationConverterTorch.convert_from_to(poses, "matrix", "axis_angle")
    joints = selected_joints if selected_joints is not None else AmassJoints
    joints_inds = [joint.value for joint in joints]
    joints_inds = np.concatenate(
        [[3 * ind, 3 * ind + 1, 3 * ind + 2] for ind in joints_inds]
    )
    poses_all[:, joints_inds] = poses.reshape(-1, 3 * len(joints))
    return s_fk.forward(poses_all)  # [...,:2]
