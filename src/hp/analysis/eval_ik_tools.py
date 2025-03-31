import torch
import numpy as np
from scipy.spatial.transform import Rotation
from typing import Tuple

from hp.utils.rotation_tools import batch_geodesic_distance_torch
from hp.utils.skeleton_forward_kinematics import perform_forward_kinematics
from hp.data.amass_cfg import AmassJoints
from hp.metrics.metrics_inverse_kinematic import compute_3d_keypoints_h36m
from hp.metrics.metrics_inverse_kinematic import p_mpjpe

min_over_samples = []
min_mpjpe_over_samples = []

geo_over_mean_samples = []
mpjpe_over_mean_samples = []


def compute_mean_metrics_over_samples(
    trainer,
    ths,
    theta_raw,
    p3d,
    return_mean=False,
    only_return_mean=False,
    database=None,
    selected_joints=None,
    skeleton=None,
) -> Tuple[float, float]:
    ths_np = ths.cpu().numpy()
    mean_rotations = np.zeros((ths_np.shape[1], ths_np.shape[2], 3, 3))
    for i_s in range(ths_np.shape[1]):
        for i_joint in range(ths_np.shape[2]):
            ths_np_i = ths_np[:, i_s, i_joint]
            rot_obj = Rotation.from_matrix(ths_np_i)
            mean_direction = rot_obj.mean().as_matrix()
            mean_rotations[i_s, i_joint] = mean_direction
    mean_rotations = torch.tensor(mean_rotations, dtype=torch.float32)

    with torch.no_grad():
        s = trainer.forward_estimator(
            {
                "theta": mean_rotations.to(trainer.config.device),
                "condition": p3d.to(trainer.config.device)[:, :, :].reshape(
                    -1, 22 * p3d.shape[-1]
                ),
            }
        )
    print(f"LL={s[1].mean().item()}")

    if trainer is not None:
        database = trainer.config.data.database
        selected_joints = trainer.selected_joints
        if database == "H36M":
            skeleton = trainer.dataset.skeleton

    if only_return_mean:
        return mean_rotations

    loss_geod = batch_geodesic_distance_torch(mean_rotations, theta_raw.cpu())
    if database == "H36M":
        p3d_hat = skeleton.forward_kinematics_from_dynamic_nodes(
            mean_rotations, mode="matrix"
        )
    elif database == "AMASS":
        p3d_hat = perform_forward_kinematics(mean_rotations.cpu(), selected_joints)
    p3d_hat = p3d_hat[..., : p3d.shape[-1]]
    loss_p3d = torch.abs((p3d_hat.cpu() - p3d.cpu()))
    mjp = loss_p3d.norm(dim=-1)
    pm_error = p_mpjpe(p3d_hat, p3d)
    if return_mean:
        return (
            mjp.mean().item(),
            loss_geod.mean().item(),
            pm_error.mean().item(),
            mean_rotations,
        )
    return mjp.mean().item(), loss_geod.mean().item(), pm_error.mean().item()


def get_samples(
    trainer,
    p3d: torch.tensor,
    k_samples: int,
    n_samples: int,
    theta_raw,
    return_ll: bool = False,
):
    p3d_origin = p3d
    thetas_hat = []
    lps = []
    for i_s in range(0, k_samples):
        p3d = p3d_origin.clone()
        theta_hat, lp = trainer.sample_from_estimator(n_samples, p3d)
        thetas_hat.append(theta_hat)
        lps.append(lp)

    with torch.no_grad():
        p3d = p3d_origin.clone()
        o = trainer.forward_estimator({"theta": theta_raw, "condition": p3d})

    ths = torch.stack(thetas_hat)
    if return_ll:
        lps = torch.stack(lps)
        return ths, lps
    return ths


def compute_geod_and_mpjpe_for_based_on_k_samples(
    n_samples: int,
    k_samples: int,
    p3d: torch.tensor,
    theta_raw: torch.tensor,
    compute_fk: bool = True,
    trainer=None,
    selected_joints: list = AmassJoints,
    ths: torch.tensor = None,
    database: str = None,
    skeleton: any = None,
    all_joints: bool = False,
    return_eval_values: bool = False,
    return_eval_and_min_values: bool = False,
    select_best: bool = False,
    return_ll: bool = False,
) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:

    cond_dim = p3d.reshape(p3d.shape[0], 22, -1).shape[-1]
    if ths is None:
        if trainer is None:
            raise ValueError("Please provide a trainer")
        ths, lps = get_samples(
            trainer, p3d, k_samples, n_samples, theta_raw, return_ll=True
        )
    if select_best:
        inds = lps.min(dim=0).indices
        print(f"LL {-lps.min(dim=0).values.mean().item(), -lps.mean().item():.3f}")
        inds2 = torch.stack([inds, torch.arange(ths.shape[1]).to(ths.device)])
        ths = ths[inds2[0], inds2[1]][None]

    if trainer is not None:
        selected_joints = trainer.selected_joints

    if trainer is not None:
        database = trainer.config.data.database
        if trainer.config.data.database == "H36M":
            skeleton = trainer.dataset.skeleton
    else:
        if database is None:
            raise ValueError("Please provide a database name")

    if return_ll:
        return ths, lps

    gs = []
    gds_mean = []
    mjps = []
    p_mjps = []
    for i in range(ths.shape[0]):
        theta_hat = ths[i]
        loss_geod = batch_geodesic_distance_torch(theta_hat, theta_raw)
        gds_mean.append(loss_geod.mean())
        gs.append(loss_geod)

        if database == "H36M":
            if all_joints:
                p3d_hat = skeleton.forward_kinematics(theta_hat, mode="matrix")
                p3d_raw = skeleton.forward_kinematics(theta_raw, mode="matrix")
            else:
                p3d_hat = skeleton.forward_kinematics_from_dynamic_nodes(
                    theta_hat, mode="matrix"
                )
                p3d_raw = skeleton.forward_kinematics_from_dynamic_nodes(
                    theta_raw, mode="matrix"
                )
        elif database == "AMASS":
            if compute_fk:
                p3d_raw = perform_forward_kinematics(theta_raw.cpu(), selected_joints)
            else:
                p3d_raw = p3d.reshape(theta_raw.shape[0], 22, -1).cpu()
            p3d_hat = perform_forward_kinematics(theta_hat.cpu(), selected_joints)
            p3d_hat = p3d_hat.reshape(theta_hat.shape[0], 22, -1)[..., :cond_dim]
            p3d_raw = p3d_raw[..., :cond_dim]

        else:
            raise NotImplementedError(f"Database {database} not implemented")

        loss_p3d = torch.abs((p3d_hat - p3d_raw))
        mjp = loss_p3d.norm(dim=-1)
        mjps.append(mjp)
        pm_error = p_mpjpe(p3d_hat, p3d_raw)
        p_mjps.append(pm_error)

    gs = torch.stack(gs)
    mjps = torch.stack(mjps)
    p_mjps = torch.tensor(p_mjps)
    gds_mean = torch.stack(gds_mean)
    print(f"-- {k_samples} samples --")
    mean_mpjpe_over_samples = mjps.mean().item()
    mean_geo_over_samples = gds_mean.mean().item()
    mos = gs.min(dim=0).values.mean().item()
    mjpe = mjps.min(dim=0).values.mean().item()
    p_mjp = p_mjps.min(dim=0).values.item()
    min_over_samples.append(mos)
    min_mpjpe_over_samples.append(mjpe)

    mjpe_mean, g_mean, p_mjpe_mean = compute_mean_metrics_over_samples(
        trainer,
        ths,
        theta_raw,
        p3d_raw,
        skeleton=skeleton,
        database=database,
        selected_joints=selected_joints,
    )
    geo_over_mean_samples.append(g_mean)
    mpjpe_over_mean_samples.append(mjpe_mean)

    print(
        f"Mean MPJPE over samples={mean_mpjpe_over_samples*1000:.3f} | Mean GEO={mean_geo_over_samples:.3f} | "
        + f"Min MPJPE={mjpe*1000:.3f} | Mean MPJPE={mjpe_mean*1000:.3f} | "
        + f"Min GEO={mos:.3f} | Mean GEO={g_mean:.3f} | "
        + f"P-MPJPE={p_mjp*1000:.3f} | Mean P-MPJPE={p_mjpe_mean*1000:.3f}"
    )

    if return_eval_and_min_values:
        return ths, p3d_raw, gs, mjps, g_mean, mjpe_mean, p_mjpe_mean, mjpe, mos, p_mjp

    if return_eval_values:
        return ths, p3d_raw, gs, mjps, g_mean, mjpe_mean, p_mjpe_mean

    return ths, p3d_raw, gs, mjps
