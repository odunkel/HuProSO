import torch
import numpy as np
from matplotlib import pyplot as plt
from hp.utils.skeleton_forward_kinematics import perform_forward_kinematics
from hp.utils.rotation_tools import batch_geodesic_distance_torch
from hp.data.amass_cfg import AmassJoints4Learning
from hp.utils.rotation_tools import RotationConverterTorchRoma as RotationConverterTorch
from hp.utils.smpl import relative2absolute_joint_coordinates

try:
    from hp.visualization.vis_tools import HumanRender
except:
    print("Visualizer not available. Install body_visualizer to use it.")


def get_vertices_of_pose(
    theta_pose,
    render_human=False,
    config=None,
    selected_joints=AmassJoints4Learning,
    fig=None,
):
    vs = []
    render_thetas = theta_pose.detach().cpu().numpy()
    pose_quats = RotationConverterTorch.convert_from_to(
        render_thetas,
        from_rot_type=config.rotation_representation,
        to_rot_type="quaternion",
        config=None,
    )
    pose_bodys = RotationConverterTorch.convert_from_to(
        pose_quats, from_rot_type="quaternion", to_rot_type="axis_angle", config=None
    )
    hr = HumanRender(device="cpu")
    human_render_path = f"{config.save.path_dir}/render"

    for i, _ in enumerate(render_thetas):
        pose_body_render = torch.zeros((22, 3))
        selected_joints_ints = [sj.value for sj in selected_joints]
        pose_body_render[selected_joints_ints] = torch.tensor(
            pose_bodys[i], dtype=torch.float32
        )
        v = hr.get_vertices_from_pose(pose_body_render[1:])
        if render_human:
            hr.render_human(pose_body_render[1:])
        vs.append(v)
    vs = torch.concatenate(vs, dim=0)
    return vs


def get_render_thetas(
    theta_pose, theta_pose_gt, config, selected_joints=AmassJoints4Learning
):
    raise NotImplementedError("Not yet implemented...")


def compute_3d_keypoints_h36m(trainer, theta_hat, theta_raw):
    # theta are in matrix representation
    N_joints = trainer.dataset.skeleton.num_nodes

    qs = RotationConverterTorch.convert_from_to(theta_hat, "matrix", "quaternion")
    qs_hat_all = torch.zeros((theta_hat.shape[0], N_joints, 4)).to(qs.device)
    qs_hat_all[..., 0] = 1
    qs_hat_all[:, trainer.dataset.selected_joints] = qs
    p3d_hat_rel = trainer.dataset.skeleton.forward(qs_hat_all) * 1e-3
    p3d_hat = relative2absolute_joint_coordinates(
        p3d_hat_rel,
        trainer.dataset.skeleton._parents,
        get_body_pose_related_joints=False,
    )
    p3d_hat = p3d_hat.to(theta_raw.device)

    qs = RotationConverterTorch.convert_from_to(theta_raw, "matrix", "quaternion")
    qs_gt_all = torch.zeros((theta_hat.shape[0], N_joints, 4)).to(qs.device)
    qs_gt_all[..., 0] = 1
    qs_gt_all[:, trainer.dataset.selected_joints] = qs
    p3d_gt_rel = trainer.dataset.skeleton.forward(qs_gt_all) * 1e-3
    p3d_gt = relative2absolute_joint_coordinates(
        p3d_gt_rel,
        trainer.dataset.skeleton._parents,
        get_body_pose_related_joints=False,
    )
    p3d_gt = p3d_gt.to(theta_raw.device)

    return p3d_hat, p3d_gt


def compute_3d_keypoints(
    theta: torch.tensor,
    theta_gt: torch.tensor,
    selected_joints: list = AmassJoints4Learning,
):
    if theta_gt is not None:
        p3d = perform_forward_kinematics(theta_gt.cpu(), selected_joints)
    else:
        p3d = p3d.cpu().reshape(theta.shape[0], -1, 3)
    poses = theta.to("cpu")
    p3d_est = perform_forward_kinematics(poses, selected_joints)
    p3d_est = p3d_est.to(theta_gt.device)
    p3d = p3d.to(theta_gt.device)
    return p3d_est, p3d


def compute_mpjpe(
    theta: torch.tensor,
    p3d: torch.tensor,
    selected_joints: list = AmassJoints4Learning,
    p3d_est: torch.tensor = None,
    theta_raw: torch.tensor = None,
    mode_2d: bool = False,
):
    if p3d_est is None:
        p3d_est, p3d = compute_3d_keypoints(theta, theta_raw, selected_joints)
    loss_p3d = torch.abs((p3d_est - p3d))
    lj = loss_p3d.norm(dim=-1).mean(axis=0)
    if mode_2d:
        return lj, torch.abs((p3d_est[..., :2] - p3d[..., :2])).norm(dim=-1).mean(
            axis=0
        )
    return lj  # array of size 19


def compute_geodesic_dist(theta, theta_raw):
    theta_raw = theta_raw.to("cpu")
    theta_hat = theta.reshape(theta.shape[0], -1, 3, 3).detach().to("cpu")
    loss_geod = batch_geodesic_distance_torch(theta_hat, theta_raw).mean(dim=0)
    return loss_geod  # array of size 19


# From https://github.com/GONGJIA0208/Diffpose/blob/main/common/loss.py
def mpjpe(predicted, target):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    return torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1))


def weighted_mpjpe(predicted, target, w):
    """
    Weighted mean per-joint position error (i.e. mean Euclidean distance)
    """
    assert predicted.shape == target.shape
    assert w.shape[0] == predicted.shape[0]
    return torch.mean(w * torch.norm(predicted - target, dim=len(target.shape) - 1))


def p_mpjpe(predicted, target):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    """
    assert predicted.shape == target.shape

    predicted = predicted.cpu().numpy()
    target = target.cpu().numpy()

    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)

    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0**2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0**2, axis=(1, 2), keepdims=True))

    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1))  # Rotation

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY  # Scale
    t = muX - a * np.matmul(muY, R)  # Translation

    # Perform rigid transformation on the input
    predicted_aligned = a * np.matmul(predicted, R) + t

    # Return MPJPE
    return np.mean(
        np.linalg.norm(predicted_aligned - target, axis=len(target.shape) - 1)
    )
