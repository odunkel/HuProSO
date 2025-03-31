from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from hp.data.amass_cfg import AmassJoints, AmassJoints4Learning
from hp.metrics.similarity_measures import get_correlation_from_rv_on_p_sphere
from hp.utils.rotation_tools import batch_geodesic_distance_matrix_torch
from hp.utils.rotation_tools import RotationConverterTorchRoma as RotationConverterTorch
from hp.metrics.similarity_measures import wasserstein_distance_nd_so3
from hp.data.amass import AMASS
from hp.utils.skeleton_forward_kinematics import perform_forward_kinematics


def compute_correlation_metrics(raw_rot: np.ndarray) -> np.ndarray:

    corr_matrix = np.zeros((raw_rot.shape[1], raw_rot.shape[1]))

    for joint_1 in tqdm(range(raw_rot.shape[1])):
        for joint_2 in range(raw_rot.shape[1]):
            X = raw_rot[:, joint_1, :]
            Y = raw_rot[:, joint_2, :]
            rho = get_correlation_from_rv_on_p_sphere(X, Y)
            corr_matrix[joint_1, joint_2] = rho

    return corr_matrix


def compute_mutual_information_metrics(pdf_pd: pd.DataFrame) -> np.ndarray:

    N = 4608
    eps = 1e-20
    N_rvs = pdf_pd.shape[1] - 1
    mi_matrix = np.zeros((N_rvs, N_rvs))

    for joint_1 in tqdm(range(N_rvs)):
        for joint_2 in range(N_rvs):
            pi = pdf_pd.groupby([f"J_{joint_1}"]).sum()
            pj = pdf_pd.groupby([f"J_{joint_2}"]).sum()
            pij = pdf_pd.groupby([f"J_{joint_1}", f"J_{joint_2}"]).sum()
            k_pij = np.array([*pij["p"].index.values])
            k_pi = pi.index.values
            k_pj = pj.index.values

            p_i_all = np.zeros(N)
            p_i_all[k_pi] = pi["p"].values
            p_j_all = np.zeros(N)
            p_j_all[k_pj] = pj["p"].values

            p_ij_all = np.zeros((N, N))
            p_ij_all[k_pij[:, 0], k_pij[:, 1]] = pij["p"]

            div_ = np.divide(p_ij_all + eps, np.outer(p_i_all, p_j_all) + eps)
            mi = np.sum(np.multiply(p_ij_all, div_))
            mi_matrix[joint_1, joint_2] = mi

    return mi_matrix


line_types = ["-", "--", "-.", ":"]


def compute_precision_recall(
    distance_matrix, recall=False, precision=False, plot=False, fig=None, i_curve=0
):

    lt = line_types[i_curve % 4]
    fig_given = False
    if fig is not None:
        fig_given = True
    if recall:
        min_dists = torch.min(distance_matrix, dim=1).values.cpu().numpy()
        if plot:
            min_dists.sort()
            cumulative_sums = np.linspace(0, 1, len(min_dists))
            if not fig_given:
                fig = plt.figure()
            plt.plot(min_dists, cumulative_sums, linestyle=lt)
    if precision:
        min_dists = torch.min(distance_matrix, dim=0).values.cpu().numpy()
        if plot:
            min_dists.sort()
            cumulative_sums = np.linspace(0, 1, len(min_dists))
            if not fig_given:
                fig = plt.figure()
            plt.plot(min_dists, cumulative_sums, linestyle=lt)


def compute_geo_distances_to_gt_poses(
    dataset,
    trainer,
    N_dataset=100,
    N_pdf=1000,
    plot_curve=False,
    mode: str = "recall",
    pdf_samples=None,
    ds_samples=None,
    rotation_representation="matrix",
    compute_wasser=False,
):
    # dataset samples rot matrices
    if ds_samples is None:
        samples_ds = dataset.sample(N_dataset).reshape(N_dataset, 19, 3, 3)
    else:
        samples_ds = ds_samples[:N_dataset]

    if pdf_samples is None:
        if "VPoser" in str(trainer):
            samples_pdf = (
                trainer.sample_poses(num_poses=N_pdf)["pose_body"]
                .contiguous()
                .view(N_pdf, 21, 3)
                .cpu()
            )
            samples_pdf = RotationConverterTorch.convert_from_to(
                samples_pdf, "axis_angle", "matrix"
            )
            train_inds_for_vposer = [i.value - 1 for i in AmassJoints4Learning]
            samples_pdf = samples_pdf[:, train_inds_for_vposer]
        else:
            if type(trainer) == AMASS:  # In case the trainer is a datset
                samples_pdf = trainer.sample(N_pdf).reshape(N_pdf, 19, -1)
            else:
                with torch.no_grad():
                    samples_pdf = trainer.sample(N_pdf)[0].reshape(N_pdf, 19, -1)
            samples_pdf = RotationConverterTorch.convert_from_to(
                samples_pdf, rotation_representation, "matrix"
            )
            samples_pdf = samples_pdf.cpu().reshape(-1, 19, 3, 3)
    else:
        samples_pdf = pdf_samples[:N_pdf]
        samples_pdf = RotationConverterTorch.convert_from_to(
            samples_pdf, rotation_representation, "matrix"
        )

    if samples_ds.shape[-1] == 4:
        samples_ds = RotationConverterTorch.convert_from_to(
            samples_ds, "quaternion", "matrix"
        )

    if compute_wasser:
        distance = wasserstein_distance_nd_so3(samples_ds[:2000], samples_pdf[:2000])
        print(distance)

    device = "cpu"
    samples_ds = samples_ds.to(device).to(torch.float32)
    samples_pdf = samples_pdf.to(device).to(torch.float32)
    distance_matrix_sum = torch.zeros((samples_ds.shape[0], samples_pdf.shape[0])).to(
        device
    )
    for i in range(samples_pdf.shape[1]):
        distance_matrix = batch_geodesic_distance_matrix_torch(
            samples_ds[:, i], samples_pdf[:, i]
        )
        distance_matrix_sum += distance_matrix

    del distance_matrix
    torch.cuda.empty_cache()

    min_dim = 0 if mode == "precision" else 1
    min_dists = torch.min(distance_matrix_sum, dim=min_dim).values.cpu().numpy()
    if plot_curve:
        compute_precision_recall(
            distance_matrix_sum, recall=True, precision=True, plot=True, fig=None
        )

    return distance_matrix_sum, min_dists


def compute_mpjpe_to_gt_poses(
    dataset,
    trainer,
    N_dataset=100,
    N_pdf=1000,
    plot_curve=False,
    mode: str = "recall",
    pdf_samples=None,
    ds_samples=None,
    rotation_representation="matrix",
    compute_wasser=False,
):
    # dataset samples rot matrices
    if ds_samples is None:
        samples_ds = dataset.sample(N_dataset).reshape(N_dataset, 19, 3, 3)
    else:
        samples_ds = ds_samples[:N_dataset]

    if pdf_samples is None:
        if "VPoser" in str(trainer):
            samples_pdf = (
                trainer.sample_poses(num_poses=N_pdf)["pose_body"]
                .contiguous()
                .view(N_pdf, 21, 3)
                .cpu()
            )
            samples_pdf = RotationConverterTorch.convert_from_to(
                samples_pdf, "axis_angle", "matrix"
            )
            train_inds_for_vposer = [i.value - 1 for i in AmassJoints4Learning]
            samples_pdf = samples_pdf[:, train_inds_for_vposer]
        else:
            if type(trainer) == AMASS:  # In case the trainer is a datset
                samples_pdf = trainer.sample(N_pdf).reshape(N_pdf, 19, -1)
            else:
                with torch.no_grad():
                    samples_pdf = trainer.sample(N_pdf)[0].reshape(N_pdf, 19, -1)
            samples_pdf = RotationConverterTorch.convert_from_to(
                samples_pdf, rotation_representation, "matrix"
            )
            samples_pdf = samples_pdf.cpu().reshape(-1, 19, 3, 3)
    else:
        samples_pdf = pdf_samples[:N_pdf]
        samples_pdf = RotationConverterTorch.convert_from_to(
            samples_pdf, rotation_representation, "matrix"
        )

    if samples_ds.shape[-1] == 4:
        samples_ds = RotationConverterTorch.convert_from_to(
            samples_ds, "quaternion", "matrix"
        )

    device = "cpu"
    samples_ds = samples_ds.to(device).to(torch.float32)
    samples_pdf = samples_pdf.to(device).to(torch.float32)

    samples_ds = perform_forward_kinematics(
        samples_ds, selected_joints=AmassJoints4Learning
    )
    samples_pdf = perform_forward_kinematics(
        samples_pdf, selected_joints=AmassJoints4Learning
    )

    distance_matrix_sum = (
        (samples_ds.unsqueeze(1) - samples_pdf.unsqueeze(0)).norm(dim=-1).sum(dim=-1)
    )
    min_dim = 0 if mode == "precision" else 1
    min_dists = torch.min(distance_matrix_sum, dim=min_dim).values.cpu().numpy()
    if plot_curve:
        compute_precision_recall(
            distance_matrix_sum, recall=True, precision=True, plot=True, fig=None
        )

    return distance_matrix_sum, min_dists
