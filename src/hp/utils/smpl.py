import numpy as np
from typing import Dict
import logging
import torch
from hp.utils.rotation_tools import RotationConverterTorchRoma as RC
from hp.data.amass_cfg import AmassJoints
from typing import Union, List

device = "cpu"


class SmplSkeleton:

    def __init__(
        self, parents: Dict[int, int] = None, p3d0: torch.tensor = None
    ) -> None:
        # parents = {0: -1, 1: 0, 2: 0, 3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7, 11: 8, 12: 9, 13: 9, 14: 9, 15: 12, 16: 13, 17: 14, 18: 16, 19: 17, 20: 18, 21: 19, 22: 20, 23: 21}
        self.skel = None
        if parents is None:
            self.skel = np.load("../body_visualizer/support_data/smpl_skeleton.npz")
            parents = self.skel["parents"]

        parent = {}
        for i in range(len(parents)):
            parent[i] = parents[i]

        self.parent = parent

        if p3d0 is None and self.skel is not None:
            p3d0 = torch.from_numpy(self.skel["p3d0"]).float().to(device)

        self.p3d0 = p3d0
        if self.p3d0 is None:
            logging.warning(
                "No positions defined for skeleton. Forward kinematics not possible."
            )

    def ang_2_transformations_last_considered_joint_in_rest_pose(
        self, p3d0: torch.tensor, pose: torch.tensor
    ):
        # Derived from https://github.com/wei-mao-2019/HisRepItself/blob/55a4186fd962388f25f21afebfbabdb493fd3451/utils/ang2joint.py
        """
        :param p3d0:[batch_size, joint_num, 3]
        :param pose:[batch_size, joint_num, 3]
        :return: transformations:[batch_size, joint_num, 3]
        """
        batch_num = p3d0.shape[0]
        jnum = pose.shape[-2]  # len(self.parent.keys())
        J = p3d0
        R_cube_big = RC.convert_from_to(
            pose.contiguous().view(-1, 1, 3), "axis_angle", "matrix"
        ).reshape(batch_num, -1, 3, 3)
        results = []
        results.append(
            SmplSkeleton.with_zeros(
                torch.cat(
                    (R_cube_big[:, 0], torch.reshape(J[:, 0, :], (-1, 3, 1))), dim=2
                )
            )
        )
        results_last_in_rest = [results[0]]

        for i in range(1, min(jnum, len(self.parent))):
            if i <= 3:
                B = torch.matmul(
                    results[self.parent[i]],
                    SmplSkeleton.with_zeros(
                        torch.cat(
                            (
                                R_cube_big[:, 0],
                                torch.reshape(
                                    J[:, i, :] - J[:, self.parent[i], :], (-1, 3, 1)
                                ),
                            ),
                            dim=2,
                        )
                    ),
                )
            else:
                A = torch.matmul(
                    results[self.parent[self.parent[i]]],
                    SmplSkeleton.with_zeros(
                        torch.cat(
                            (
                                R_cube_big[:, 0],
                                torch.reshape(
                                    J[:, self.parent[i], :]
                                    - J[:, self.parent[self.parent[i]], :],
                                    (-1, 3, 1),
                                ),
                            ),  # identity
                            dim=2,
                        )
                    ),
                )
                B = torch.matmul(
                    A,
                    SmplSkeleton.with_zeros(
                        torch.cat(
                            (
                                R_cube_big[:, 0],
                                torch.reshape(
                                    J[:, i, :] - J[:, self.parent[i], :], (-1, 3, 1)
                                ),
                            ),  # identity
                            dim=2,
                        )
                    ),
                )
            results_last_in_rest.append(B)

            results.append(
                torch.matmul(
                    results[self.parent[i]],
                    SmplSkeleton.with_zeros(
                        torch.cat(
                            (
                                R_cube_big[:, i],
                                torch.reshape(
                                    J[:, i, :] - J[:, self.parent[i], :], (-1, 3, 1)
                                ),
                            ),
                            dim=2,
                        )
                    ),
                )
            )

        stacked = torch.stack(results, dim=1)
        stacked_last_in_rest = torch.stack(results_last_in_rest, dim=1)

        return stacked, stacked_last_in_rest

    def ang_2_transformations(self, p3d0: torch.tensor, pose: torch.tensor):
        # Derived from https://github.com/wei-mao-2019/HisRepItself/blob/55a4186fd962388f25f21afebfbabdb493fd3451/utils/ang2joint.py
        """
        :param p3d0:[batch_size, joint_num, 3]
        :param pose:[batch_size, joint_num, 3]
        :return: transformations:[batch_size, joint_num, 3]
        """
        batch_num = p3d0.shape[0]
        jnum = pose.shape[-2]  # len(self.parent.keys())
        J = p3d0
        R_cube_big = RC.convert_from_to(
            pose.contiguous().view(-1, 1, 3), "axis_angle", "matrix"
        ).reshape(batch_num, -1, 3, 3)

        results = []
        results.append(
            SmplSkeleton.with_zeros(
                torch.cat(
                    (R_cube_big[:, 0], torch.reshape(J[:, 0, :], (-1, 3, 1))), dim=2
                )
            )
        )
        for i in range(1, min(jnum, len(self.parent))):
            results.append(
                torch.matmul(
                    results[self.parent[i]],
                    SmplSkeleton.with_zeros(
                        torch.cat(
                            (
                                R_cube_big[:, i],
                                torch.reshape(
                                    J[:, i, :] - J[:, self.parent[i], :], (-1, 3, 1)
                                ),
                            ),
                            dim=2,
                        )
                    ),
                )
            )

        stacked = torch.stack(results, dim=1)

        return stacked

    def transformations_2_joint(self, transformations: torch.tensor) -> torch.tensor:
        """
        Select the joint positions from the 4x4 pose matrices.
        :param: transformations:[batch_size, joint_num, 4, 4]
        :return: joints:[batch_size, joint_num, 3]
        """
        return transformations[:, :, :3, 3]

    def ang2joint(self, p3d0: torch.tensor, pose: torch.tensor) -> torch.tensor:
        transformations = self.ang_2_transformations(p3d0, pose)
        J_transformed = transformations[:, :, :3, 3]
        return J_transformed

    def poses_configs_2_transformations(
        self,
        poses: torch.tensor,
        remove_global_rotation: bool = False,
        with_rest: bool = False,
    ) -> Union[torch.tensor, List[torch.tensor]]:
        # Expects aa representation
        if type(poses) is np.ndarray:
            device = "cpu"
            poses = torch.from_numpy(poses).float().to(device)

        if len(poses.shape) > 2 and poses.shape[-2] != 1:
            logging.info(f"Reshaping poses from {poses.shape} to (1,-1).")
            poses = poses.reshape(poses.shape[0], -1)

        fn = poses.shape[0]
        poses = poses.reshape([fn, -1, 3])

        if remove_global_rotation:
            poses[:, 0] = 0

        p3d0_tmp = self.p3d0.repeat([fn, 1, 1])

        if with_rest:
            transformations, transformations_with_rest = (
                self.ang_2_transformations_last_considered_joint_in_rest_pose(
                    p3d0_tmp, poses
                )
            )  # FIXME
            transformations = [transformations, transformations_with_rest]
        else:
            transformations = self.ang_2_transformations(p3d0_tmp, poses)

        return transformations

    def poses_2_3d(
        self, poses: torch.tensor, remove_global_rotation: bool = True
    ) -> torch.tensor:
        # partly taken from https://github.com/wei-mao-2019/HisRepItself/blob/55a4186fd962388f25f21afebfbabdb493fd3451/utils/amass3d.py

        transformations = self.poses_configs_2_transformations(
            poses, remove_global_rotation
        )

        p3d = self.transformations_2_joint(transformations)
        # p3d = self.ang2joint(p3d0_tmp, poses).float()

        return p3d  # .cpu().data.numpy()

    def poses_2_3d_with_rest(
        self, poses: torch.tensor, remove_global_rotation: bool = True
    ) -> torch.tensor:
        # partly taken from https://github.com/wei-mao-2019/HisRepItself/blob/55a4186fd962388f25f21afebfbabdb493fd3451/utils/amass3d.py
        transformations = self.poses_configs_2_transformations(
            poses, remove_global_rotation, with_rest=True
        )
        p3d = self.transformations_2_joint(transformations[0])
        p3d_rest = self.transformations_2_joint(transformations[1])
        return p3d, p3d_rest

    @staticmethod
    def with_zeros(x):
        # Taken from https://github.com/wei-mao-2019/HisRepItself/blob/55a4186fd962388f25f21afebfbabdb493fd3451/utils/ang2joint.py
        """
        Append a [0, 0, 0, 1] tensor to a [3, 4] tensor.

        Parameter:
        ---------
        x: Tensor to be appended.

        Return:
        ------
        Tensor after appending of shape [4,4]

        """
        ones = (
            torch.tensor([[[0.0, 0.0, 0.0, 1.0]]], dtype=torch.float)
            .expand(x.shape[0], -1, -1)
            .to(x.device)
        )
        ret = torch.cat((x, ones), dim=1)
        return ret

    @staticmethod
    def pack(x):
        # Taken from https://github.com/wei-mao-2019/HisRepItself/blob/55a4186fd962388f25f21afebfbabdb493fd3451/utils/ang2joint.py
        """
        Append zero tensors of shape [4, 3] to a batch of [4, 1] shape tensor.

        Parameter:
        ----------
        x: A tensor of shape [batch_size, 4, 1]

        Return:
        ------
        A tensor of shape [batch_size, 4, 4] after appending.

        """
        zeros43 = torch.zeros((x.shape[0], x.shape[1], 4, 3), dtype=torch.float).to(
            x.device
        )
        ret = torch.cat((zeros43, x), dim=3)
        return ret


def absolute2relative(x, parents, invert=False, x0=None):
    """
    x: [bs,..., jn, 3] or [bs,..., jn-1, 3] if invert
    x0: [1,..., jn, 3]
    parents: [-1,0,1 ...]
    """
    if not invert:
        xt = x[..., 1:, :] - x[..., parents[1:], :]
        xt = xt / np.linalg.norm(xt, axis=-1, keepdims=True)
        return xt
    else:
        jn = x0.shape[-2]
        limb_l = np.linalg.norm(
            x0[..., 1:, :] - x0[..., parents[1:], :], axis=-1, keepdims=True
        )
        xt = x * limb_l
        xt0 = np.zeros_like(xt[..., :1, :])
        xt = np.concatenate([xt0, xt], axis=-2)
        for i in range(1, jn):
            xt[..., i, :] = xt[..., parents[i], :] + xt[..., i, :]
        return xt


def absolute2relative_torch(x, parents, invert=False, x0=None):
    """
    x: [bs,..., jn, 3] or [bs,..., jn-1, 3] if invert
    x0: [1,..., jn, 3]
    parents: [-1,0,1 ...]
    """
    if not invert:
        xt = x[..., 1:, :] - x[..., parents[1:], :]
        xt = xt / torch.norm(xt, dim=-1, keepdim=True)
        return xt
    else:
        jn = x0.shape[-2]
        limb_l = torch.norm(
            x0[..., 1:, :] - x0[..., parents[1:], :], dim=-1, keepdim=True
        )
        xt = x * limb_l
        xt0 = torch.zeros_like(xt[..., :1, :])
        xt = torch.cat([xt0, xt], dim=-2)
        for i in range(1, jn):
            xt[..., i, :] = xt[..., parents[i], :] + xt[..., i, :]
        return xt


def relative2absolute_joint_coordinates(
    p3d_rel: Union[np.ndarray, torch.tensor],
    parents: list,
    get_body_pose_related_joints: bool = False,
) -> np.ndarray:
    """
    p3d_rel: [N_joints, 3] or [bs, N_joints, 3]
    parents: [N_joints]
    """
    if p3d_rel.shape[-2] != len(parents):
        logging.info("Reduce the parents to 24 SMPL parameters.")
        parents = parents[: p3d_rel.shape[-2] - 1]

    shape2d = True if len(p3d_rel.shape) == 2 else False
    bn = 1 if shape2d else p3d_rel.shape[0]
    p3d_rel = p3d_rel[None] if shape2d else p3d_rel

    if type(p3d_rel) == torch.Tensor:
        p3d_abs = torch.zeros((bn, len(parents), 3), device=p3d_rel.device)
    elif type(p3d_rel) == np.ndarray:
        p3d_abs = np.zeros((bn, len(parents), 3))
    else:
        print(f"Not supported type {type(p3d_rel)}")
    p3d_abs[:, 1:, :] = p3d_rel[:, 1 : len(parents)] + p3d_abs[:, parents[1:]]

    if get_body_pose_related_joints:
        relevant_inds = list(np.arange(22))
        p3d_abs = p3d_abs[:, relevant_inds]
    if shape2d:
        p3d_abs = p3d_abs[0]
    return p3d_abs


def get_children_from_parents(parents: list) -> dict:
    parents = np.array(parents)
    unique_parents = np.unique(parents)
    children = [tuple(np.where(parents == i)[0]) for i in range(len(unique_parents))]
    return children


def children_dict_from_parents(parents: list) -> dict:
    children = get_children_from_parents(parents)
    d = dict(zip(parents, children))
    return d


LEAF_NAMES = ["Head", "LeftHand", "RightHand", "LeftToeBase", "RightToeBase"]
LEAF_IDX = [i for i in range(len(AmassJoints)) if AmassJoints(i).name in LEAF_NAMES]


def parents_to_children(parents: torch.tensor) -> torch.tensor:
    if type(parents) == list:
        parents = torch.tensor(parents)
    children = torch.ones_like(parents) * -1
    for i in range(len(AmassJoints)):
        if children[parents[i]] < 0:
            children[parents[i]] = i
    for i in LEAF_IDX:
        if i < children.shape[0]:
            children[i] = -1
    children[AmassJoints(9).value] = -3
    children[0] = 3
    children[AmassJoints(9).value] = AmassJoints(12).value

    # Add the hands as children for left and right hand joint
    children[20] = 22
    children[21] = 23
    print("[parents_to_children] Manually added children of 20 and 21.")

    return children
