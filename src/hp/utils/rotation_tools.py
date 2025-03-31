from torch import nn
import torch
import numpy as np
import roma


def batch_geodesic_distance_matrix(R1: np.ndarray, R2: np.ndarray):
    R1_expanded = R1[:, np.newaxis, :, :]
    R2_expanded = R2[np.newaxis, :, :, :]
    R = np.matmul(R1_expanded, np.transpose(R2_expanded, axes=(0, 1, 3, 2)))
    cos = (np.trace(R, axis1=2, axis2=3) - 1) / 2
    cos = np.clip(cos, -1, 1)
    distance_matrix = np.arccos(cos)
    return distance_matrix


def batch_geodesic_distance(R1: np.ndarray, R2: np.ndarray) -> np.ndarray:
    assert R1.shape == R2.shape, "Input shapes must match."
    R = np.matmul(R1, np.transpose(R2, axes=(0, 2, 1)))
    cos = (np.diagonal(R, axis1=1, axis2=2).sum(axis=1) - 1) / 2
    cos = np.minimum(cos, 1)
    cos = np.maximum(cos, -1)
    angles = np.arccos(cos)
    return angles


def batch_geodesic_distance_matrix_torch(R1: np.ndarray, R2: np.ndarray):
    R1_expanded = R1.unsqueeze(1)
    R2_expanded = R2.unsqueeze(0)
    R = torch.matmul(R1_expanded, R2_expanded.transpose(-2, -1))
    cos = (
        torch.sum(R[..., torch.arange(R.size(-2)), torch.arange(R.size(-1))], dim=-1)
        - 1
    ) / 2
    cos = torch.clamp(cos, -1, 1)
    distance_matrix = torch.acos(cos)
    return distance_matrix


def batch_geodesic_distance_torch(R1: np.ndarray, R2: np.ndarray):
    m1 = R1
    m2 = R2.transpose(-1, -2)
    m = torch.matmul(m1, m2)
    cos = (m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2] - 1) / 2
    cos = torch.clamp(cos, min=-1, max=1)
    return torch.acos(cos)


class RotationConverterTorchRoma:

    def __init__(self):
        pass

    @staticmethod
    def convert_from_to(
        to_convert: torch.tensor,
        from_rot_type: str = "quaternion",
        to_rot_type: str = "euler_xyz",
    ) -> torch.tensor:

        if from_rot_type == to_rot_type:
            return to_convert

        input_is_array = False
        if type(to_convert) == np.ndarray:
            to_convert = torch.tensor(to_convert)
            input_is_array = True

        if from_rot_type == to_rot_type:
            return to_convert

        if from_rot_type == "matrix":
            if to_convert.shape[-1] != 3 or to_convert.shape[-2] != 3:
                raise AttributeError("Input shape needs to be (...,3,3).")
            rot_mat = to_convert.clone()
        elif from_rot_type == "axis_angle":
            rot_mat = roma.rotvec_to_rotmat(to_convert)
        elif from_rot_type == "quaternion":
            rot_mat = roma.unitquat_to_rotmat(to_convert[..., [1, 2, 3, 0]])
        elif from_rot_type == "euler_xyz":
            rot_mat = roma.euler_to_rotmat("ZYX", to_convert)
        elif from_rot_type == "euler_XYZ":
            rot_mat = roma.euler_to_rotmat("XYZ", to_convert)
        elif from_rot_type == "6D":
            all_first_dims = to_convert.shape[:-2]
            rot_mat = roma.special_gramschmidt(
                to_convert.reshape(*all_first_dims, 3, 2)
            )
        elif from_rot_type == "hopf_drake":
            raise NotImplementedError(f"from hopf_drake Not implemented yet")

        if to_rot_type == "matrix":
            converted = rot_mat
        elif to_rot_type == "axis_angle":
            converted = roma.rotmat_to_rotvec(rot_mat)
        elif to_rot_type == "quaternion":
            converted = roma.rotmat_to_unitquat(rot_mat)[..., [3, 0, 1, 2]]
        elif to_rot_type == "euler_xyz":
            euler_ = roma.rotmat_to_euler("ZYX", rot_mat)
            converted = euler_
        elif to_rot_type == "euler_XYZ":
            euler_ = roma.rotmat_to_euler("XYZ", rot_mat)
            converted = euler_
        elif to_rot_type == "6D":
            all_first_dims = to_convert.shape[:-2]
            converted = rot_mat[..., :2, :].reshape(*all_first_dims, -1)
        elif to_rot_type == "hopf_drake":
            raise NotImplementedError(f"from hopf_drake Not implemented yet")

        if input_is_array:
            converted = converted.cpu().numpy()

        return converted
