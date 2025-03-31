import torch
from omegaconf import OmegaConf
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import json
import pickle
import os

from hp.analysis.eval_ik_tools import get_samples as get_n_samples
from hp.analysis.compute_metrics import compute_geo_distances_to_gt_poses
from hp.utils.rotation_tools import RotationConverterTorchRoma as RotationConverter
from hp.visualization.vis_tools import (
    visualize_like_implicit_pdf,
    visualize_so3_probabilities,
)
from hp.data.amass import AMASS
from hp.metrics.similarity_measures import (
    wasserstein_distance_so3,
    wasserstein_distance_nd_so3,
)
from hp.metrics.metrics_inverse_kinematic import (
    compute_mpjpe,
    compute_geodesic_dist,
    mpjpe,
    p_mpjpe,
    compute_3d_keypoints,
    compute_3d_keypoints_h36m,
)


class IKEvaluator:
    def __init__(self) -> None:
        pass

    @staticmethod
    def eval_with_dataset(
        trainer: "PriorTrainer",
        n_samples: int = 2_000,
        dataset: str = "eval",
        num_samples_pdf: int = 1,
        complete_ds: bool = False,
        select_best: bool = False,
    ):

        if trainer.config.conditioning.conditioning_modality == "3D":
            cond_dim = 3
        elif trainer.config.conditioning.conditioning_modality == "2D":
            cond_dim = 2
        else:
            raise NotImplementedError(
                f"Conditioning mode {trainer.config.conditioning.conditioning_modality} not implemented."
            )

        if complete_ds:
            ds = []
            for iter, s in enumerate(trainer.eval_dataloader):
                n_samples = s["condition"].shape[0]
                print(f"==Iteration {iter} with {n_samples} samples ==")
                theta_raw = (
                    s["theta"].to(trainer.config.device).reshape(n_samples, -1, 3, 3)
                )
                p3d = (
                    s["condition"]
                    .to(trainer.config.device)
                    .reshape(n_samples, -1, cond_dim)
                )
                d = IKEvaluator.eval(
                    trainer,
                    theta_raw,
                    p3d,
                    print_eval=True,
                    num_samples_pdf=num_samples_pdf,
                    select_best=select_best,
                )
                ds.append(d)

        else:
            print(f"==Evaluation with {dataset} dataset==")
            if dataset == "train":
                dataset = trainer.train_dataset
            elif dataset == "eval":
                dataset = trainer.eval_dataset
            elif dataset == "valid":
                dataset = trainer.val_dataset
            else:
                raise ValueError(f"Dataset {dataset} not known.")

            idx = list(np.random.permutation(len(dataset))[:n_samples])
            s = dataset[idx]

            if trainer.config.trainer == "Cont6DPriorTrainer":
                theta_raw = s["theta"].to(trainer.config.device).reshape(n_samples, -1)
                p3d = s["condition"].to(trainer.config.device).reshape(n_samples, -1)
            else:
                theta_raw = (
                    s["theta"].to(trainer.config.device).reshape(n_samples, -1, 3, 3)
                )
                p3d = (
                    s["condition"]
                    .to(trainer.config.device)
                    .reshape(n_samples, -1, cond_dim)
                )
            d = IKEvaluator.eval(
                trainer,
                theta_raw,
                p3d,
                print_eval=True,
                num_samples_pdf=num_samples_pdf,
                select_best=select_best,
            )
            ds = d
        return ds

    @staticmethod
    def eval(
        trainer: "PriorTrainer",
        theta_raw: torch.tensor,
        p3d: torch.tensor,
        print_eval: bool = False,
        num_samples_pdf: int = 1,
        select_best: bool = False,
    ):

        n_samples = theta_raw.shape[0]
        print(f"-> Eval with {n_samples} samples")

        if num_samples_pdf > 1:
            ths, lps = get_n_samples(
                trainer, p3d, num_samples_pdf, n_samples, theta_raw, return_ll=True
            )
            if select_best:
                inds = lps.min(dim=0).indices
                print(-lps.min(dim=0).values.mean().item(), -lps.mean().item())
                inds2 = torch.stack([inds, torch.arange(ths.shape[1]).to(ths.device)])
                theta_hat = ths[inds2[0], inds2[1]]
            else:
                ths_np = ths.cpu().numpy()
                mean_rotations = np.zeros((ths_np.shape[1], ths_np.shape[2], 3, 3))
                for i_s in range(ths_np.shape[1]):
                    for i_joint in range(ths_np.shape[2]):
                        ths_np_i = ths_np[:, i_s, i_joint]
                        rot_obj = Rotation.from_matrix(ths_np_i)
                        mean_direction = rot_obj.mean().as_matrix()
                        mean_rotations[i_s, i_joint] = mean_direction
                theta_hat = torch.tensor(mean_rotations, dtype=torch.float32)

            with torch.no_grad():
                lp = -trainer.forward_estimator(
                    {"theta": theta_hat.to(trainer.config.device), "condition": p3d}
                )[1]
        else:
            theta_hat, lp = trainer.sample_from_estimator(n_samples, p3d)

        with torch.no_grad():
            o = trainer.forward_estimator({"theta": theta_raw, "condition": p3d})

        if trainer.config.rotation_representation == "matrix":
            theta_hat = theta_hat.reshape(theta_hat.shape[0], -1, 3, 3)
            theta_raw = theta_raw.reshape(theta_hat.shape[0], -1, 3, 3)
        elif trainer.config.rotation_representation == "6D":
            theta_hat = theta_hat.reshape(theta_hat.shape[0], -1, 6)
            theta_raw = theta_raw.reshape(theta_hat.shape[0], -1, 6)
            theta_hat = RotationConverter.convert_from_to(theta_hat, "6D", "matrix")
            theta_raw = RotationConverter.convert_from_to(theta_raw, "6D", "matrix")

        if trainer.config.data.database == "H36M":
            p3d_hat, p3d_gt = compute_3d_keypoints_h36m(trainer, theta_hat, theta_raw)
        else:
            p3d_hat, p3d_gt = compute_3d_keypoints(
                theta_hat, theta_raw, trainer.selected_joints
            )

        mode_2d = True
        ljs = compute_mpjpe(
            theta_hat,
            p3d_gt,
            trainer.selected_joints,
            p3d_est=p3d_hat,
            theta_raw=theta_raw,
            mode_2d=mode_2d,
        )
        pm_error = p_mpjpe(p3d_hat, p3d_gt)
        if mode_2d:
            ljs, m2d = ljs
        geodists = compute_geodesic_dist(theta_hat, theta_raw)

        if print_eval:
            print(f" Sample Avg LL: {-lp.mean():.03f}")
            print(f" Forward Avg LL: {o.mean().item():.03f}")
            print(" MPJPE:", torch.round(1000 * ljs, decimals=1))
            print(" Geodesic distance:", torch.round(geodists, decimals=4))
            print(f" Mean MPJPE: {1000*ljs.mean():.02f}mm")
            print(f" Mean MPJPE 2D: {1000*m2d.mean():.02f}mm")
            print(
                f" Mean geodesic distance: {geodists.mean():.04f} rad = {180*geodists.mean()/np.pi:.01f} deg"
            )

        d = {
            "mpjpe": ljs.mean().item(),
            "geod_dist": geodists.mean().item(),
            "mpjpe_2d": m2d.mean().item(),
        }
        d["p_mpjpe"] = pm_error.item()
        d["ll"] = o.mean().item()
        return d


class PriorEvaluator:
    def __init__(self) -> None:
        pass

    @staticmethod
    def sample_from_dataset(
        trainer: "PriorTrainer", N_total: int = 1000, mode: str = "eval"
    ) -> np.array:
        if mode == "eval":
            inds = list(np.random.randint(0, len(trainer.eval_dataset), size=N_total))
            samples_data = trainer.eval_dataset[inds]
        elif mode == "valid":
            inds = list(np.random.randint(0, len(trainer.val_dataset), size=N_total))
            samples_data = trainer.val_dataset[inds]
        elif mode == "train":
            inds = list(np.random.randint(0, len(trainer.train_dataset), size=N_total))
            samples_data = trainer.train_dataset[inds]
        else:
            raise ValueError(f"Mode {mode} not known.")

        samples_data = samples_data.reshape(
            N_total,
            trainer.N_joint,
            samples_data.shape[-1] // trainer.N_joint // N_total,
        )

        return samples_data

    @staticmethod
    def compute_wasserstein_distance(
        config: OmegaConf, trainer: "PriorTrainer", mode: str = "valid"
    ) -> float:
        N_total = config.num_emd_samples
        with torch.no_grad():
            samples = trainer.sample(N_total)[0]
            samples = samples.reshape(N_total, trainer.N_joint, -1)
            samples_pdf = samples.detach().cpu().numpy()

        samples_data = PriorEvaluator.sample_from_dataset(
            trainer, N_total=N_total, mode="eval"
        )

        if config.rotation_representation == "matrix":
            samples_pdf = samples_pdf.reshape(N_total, trainer.N_joint, 3, 3)
            samples_data = samples_data.reshape(N_total, trainer.N_joint, 3, 3)
        rotmat_pdf = RotationConverter.convert_from_to(
            samples_pdf,
            from_rot_type=config.rotation_representation,
            to_rot_type="matrix",
        )
        rotmat_data = RotationConverter.convert_from_to(
            samples_data,
            from_rot_type=config.rotation_representation,
            to_rot_type="matrix",
        )
        if config.rotation_representation == "matrix":
            rotmat_pdf = rotmat_pdf.reshape(N_total, trainer.N_joint, 3, 3)
            rotmat_data = rotmat_data.reshape(N_total, trainer.N_joint, 3, 3)
        if samples_pdf.shape[1] == 1:
            # Get Wasserstein distance between data samples and pdf samples as metric
            distance = wasserstein_distance_so3(rotmat_data[:, 0], rotmat_pdf[:, 0])
        else:
            distance = wasserstein_distance_nd_so3(rotmat_data, rotmat_pdf)
        return distance

    @staticmethod
    def eval(
        config: OmegaConf,
        trainer: "PriorTrainer",
        save_plot: bool = False,
        render_human: bool = False,
        plot_samples: bool = False,
        compute_d_wasser: bool = True,
        add_string_to_name: str = "",
        N_eval_wasser_ndim=5,
        complete_ds: bool = False,
    ):
        N_joint = trainer.N_joint

        # Evaluate the density
        eval_dict = trainer.eval(complete_ds=complete_ds)
        eval_dict["eval_avg_ll_0"] = eval_dict["eval_avg_ll"]
        for i in range(1):
            eval_dict[f"avg_ll_{i+1}"] = trainer.eval()["eval_avg_ll"]
        eval_dict["eval_avg_ll"] = np.mean(
            [di for k, di in eval_dict.items() if "avg_ll" in k]
        ).item()
        eval_dict["eval_avg_ll_std"] = np.std(
            [di for k, di in eval_dict.items() if "avg_ll" in k]
        ).item()

        trainer_str = str(trainer).split(".")[-1].split("'")[0].split(" ")[0]
        print(f"Average log-likelihood: {eval_dict['eval_avg_ll']:.02f}")

        # Get samples from data and from pdf
        if not hasattr(config, "num_emd_samples"):
            config.num_emd_samples = 2_000
            print(f"Set config.num_emd_samples to {config.num_emd_samples}")
        N_samples_wasser = config.num_emd_samples

        with torch.no_grad():
            samples = trainer.sample(N_samples_wasser)[0]
            samples = samples.reshape(N_samples_wasser, trainer.N_joint, -1)
            samples_pdf = samples.detach().cpu().numpy()

        samples_data = PriorEvaluator.sample_from_dataset(
            trainer, N_total=N_samples_wasser, mode="eval"
        )

        if config.rotation_representation == "matrix":
            samples_pdf = samples_pdf.reshape(N_samples_wasser, N_joint, 3, 3)
            samples_data = samples_data.reshape(N_samples_wasser, N_joint, 3, 3)
        rotmat_pdf = RotationConverter.convert_from_to(
            samples_pdf,
            from_rot_type=config.rotation_representation,
            to_rot_type="matrix",
        )
        # Get Wasserstein distance between data samples and pdf samples as metric

        if compute_d_wasser:
            print("Computing the wasserstein distance...")

            rotmat_data = RotationConverter.convert_from_to(
                samples_data,
                from_rot_type=config.rotation_representation,
                to_rot_type="matrix",
            )
            selected_joint_ints = [sj.value for sj in trainer.selected_joints]
            for i_joint in range(N_joint):
                distance = wasserstein_distance_so3(
                    rotmat_data[:, i_joint], rotmat_pdf[:, i_joint]
                )
                eval_dict[f"wasserstein_distance_{selected_joint_ints[i_joint]}"] = (
                    distance
                )
            for i in range(N_eval_wasser_ndim):
                if i > 0:
                    with torch.no_grad():
                        samples = trainer.sample(N_samples_wasser)[0]
                        samples = (
                            samples.reshape(N_samples_wasser, trainer.N_joint, -1)
                            if config.rotation_representation != "matrix"
                            else samples.reshape(
                                N_samples_wasser, trainer.N_joint, 3, 3
                            )
                        )
                        samples_pdf = samples.detach().cpu().numpy()
                        rotmat_pdf = RotationConverter.convert_from_to(
                            samples_pdf,
                            from_rot_type=config.rotation_representation,
                            to_rot_type="matrix",
                        )
                    samples_data = PriorEvaluator.sample_from_dataset(
                        trainer, N_total=N_samples_wasser, mode="eval"
                    )
                    samples_data = (
                        samples_data.reshape(N_samples_wasser, trainer.N_joint, -1)
                        if config.rotation_representation != "matrix"
                        else samples_data.reshape(
                            N_samples_wasser, trainer.N_joint, 3, 3
                        )
                    )
                    rotmat_data = RotationConverter.convert_from_to(
                        samples_data,
                        from_rot_type=config.rotation_representation,
                        to_rot_type="matrix",
                    )

                eval_dict[f"wasserstein_distance_ndim_iter_{i}"] = (
                    wasserstein_distance_nd_so3(rotmat_data, rotmat_pdf)
                )

            list_d_wasser = [
                eval_dict[f"wasserstein_distance_ndim_iter_{i}"]
                for i in range(N_eval_wasser_ndim)
            ]
            eval_dict[f"wasserstein_distance_ndim"] = np.mean(list_d_wasser)
            eval_dict[f"wasserstein_distance_ndim_std"] = np.std(list_d_wasser)

            joints_eval_str = f""
            for i_joint in range(N_joint):
                joints_eval_str += f"d_wasser_{selected_joint_ints[i_joint]}: {eval_dict[f'wasserstein_distance_{selected_joint_ints[i_joint]}']:.02f} "
            print(joints_eval_str)
            print(
                f"eval_avg_ll = {eval_dict['eval_avg_ll']:.03f} | d_wasser_ndim = {eval_dict['wasserstein_distance_ndim']:.03f} | wasserstein_distance_ndim_std = {eval_dict['wasserstein_distance_ndim_std']:.05f}"
            )

        if save_plot:
            sel_joint = (
                config.selected_joints[0]
                if len(config.selected_joints) == 1
                else config.selected_joints
            )
            with open(
                f"{config.save.path_dir}/eval_dict_{sel_joint}_{config.rotation_representation}_{trainer_str}{add_string_to_name}.json",
                "w",
            ) as f:
                json.dump(eval_dict, f, indent=4)

        # Visualize evaluated samples
        if plot_samples:
            print("Visualize samples...")
            if save_plot:
                eval_samples_path = f"{config.save.path_dir}/eval_samples"
                isExist = os.path.exists(eval_samples_path)
                if not isExist:
                    os.makedirs(eval_samples_path)
            inds = np.random.choice(rotmat_pdf.shape[0], 500, replace=False)
            for i, j in enumerate(trainer.dataset.selected_joints):
                visualize_like_implicit_pdf(rotmat_pdf[inds, i])
                plt.title(f"Joint {j.name}")
                if save_plot:
                    plt.savefig(
                        f"{eval_samples_path}/eval_samples_{config.rotation_representation}_{trainer_str}_joint_{j:02}{add_string_to_name}.png"
                    )
                    plt.close()
            if samples_pdf.shape[1] == 1:
                N_pdf = 10_000
                samples_eval = Rotation.random(N_pdf).as_quat()
                samples_eval[..., [0, 1, 2, 3]] = samples_eval[..., [3, 0, 1, 2]]
                # print("Converted to pytorch3d convention (w,x,y,z) from (x,y,z,w).")

                samples_eval_rot_mat = RotationConverter.convert_from_to(
                    samples_eval,
                    from_rot_type="quaternion",
                    to_rot_type="matrix",
                    no_batch=True,
                )

                samples_eval_rot_mat = RotationConverter.convert_from_to(
                    samples_eval,
                    from_rot_type="quaternion",
                    to_rot_type="matrix",
                    no_batch=True,
                )
                samples_eval = RotationConverter.convert_from_to(
                    samples_eval,
                    from_rot_type="quaternion",
                    to_rot_type=config.rotation_representation,
                    no_batch=True,
                )
                samples_eval = samples_eval.reshape(N_pdf, -1)

                if config.rotation_representation == "matrix":
                    samples_eval = samples_eval.reshape(N_pdf, 3, 3)
                samples_eval = torch.tensor(samples_eval, dtype=torch.float32).to(
                    config.device
                )
                with torch.no_grad():
                    log_prob = trainer.forward(samples_eval)
                    prob = np.exp((log_prob - log_prob.max()).detach().cpu().numpy())
                plot_scale = 2e3 / np.max(prob)
                visualize_so3_probabilities(
                    samples_eval_rot_mat,
                    prob,
                    scatterpoint_scaling=plot_scale,
                    title=f"",
                )
                if save_plot:
                    viz_implicit_path = f"{config.save.path_dir}/viz_samples"
                    isExist = os.path.exists(viz_implicit_path)
                    if not isExist:
                        os.makedirs(viz_implicit_path)
                    plt.savefig(
                        f"{viz_implicit_path}/eval_pdf_{config.rotation_representation}_{trainer_str}_joint_{j:02}_pdf{add_string_to_name}.png"
                    )
                    plt.savefig(
                        f"{viz_implicit_path}/0_eval_pdf_{config.rotation_representation}_{trainer_str}_joint_{j:02}_pdf{add_string_to_name}.png",
                        dpi=300,
                        bbox_inches="tight",
                    )
                    plt.close()

        # Visualize generated samples
        if render_human:
            from hp.visualization.vis_tools import HumanRender

            print("Render huaman...")
            with torch.no_grad():
                N_plot = 15
                samples = trainer.sample(N_plot)[0]
                test_samples = samples.detach().cpu().numpy()
            pose_quats = RotationConverter.convert_from_to(
                test_samples,
                from_rot_type=config.rotation_representation,
                to_rot_type="quaternion",
            )
            pose_bodys = RotationConverter.convert_from_to(
                pose_quats, from_rot_type="quaternion", to_rot_type="axis_angle"
            )
            hr = HumanRender(device="cpu")
            human_render_path = f"{config.save.path_dir}/render"
            isExist = os.path.exists(human_render_path)
            if not isExist:
                os.makedirs(human_render_path)

            for i, _ in enumerate(test_samples):

                pose_body_render = torch.zeros((22, 3))
                selected_joints_ints = [sj.value for sj in trainer.selected_joints]
                pose_body_render[selected_joints_ints] = torch.tensor(
                    pose_bodys[i], dtype=torch.float32
                )
                hr.render_human(pose_body_render[1:])
                if save_plot:
                    plt.savefig(
                        f"{human_render_path}/render_human_{config.rotation_representation}_{trainer_str}_pose_generated_{i:02}{add_string_to_name}.png"
                    )
                    plt.close()

        return eval_dict

    @staticmethod
    def eval_dataset_coverage(
        trainer: "PriorTrainer",
        ds: AMASS,
        experiment_path: str = f"../docu/experiments/priors/",
        N_large: int = 100_000,
        N_small: int = 1_000,
        plot: bool = False,
        save: bool = False,
        add_string_to_name: str = "",
        return_matrix: bool = False,
    ):
        print("Computing recall...")
        N_dataset = N_small
        N_pdf = N_large
        print(f" -> N_dataset: {N_dataset} | N_pdf: {N_pdf}")
        dist_matrix_recall = compute_geo_distances_to_gt_poses(
            ds, trainer, N_dataset=N_dataset, N_pdf=N_pdf, plot_curve=plot
        )
        if save:
            file_dir = f"{experiment_path}/distance_matrix_data_{N_dataset}_pdf_{N_pdf}{add_string_to_name}.p"
            with open(file_dir, "wb") as handle:
                pickle.dump(
                    dist_matrix_recall, handle, protocol=pickle.HIGHEST_PROTOCOL
                )

        if not return_matrix:
            del dist_matrix_recall
            torch.cuda.empty_cache()

        # Precision: 100k data, 1k pdf
        print("Computing precision...")
        N_dataset = N_large
        N_pdf = N_small
        print(f" -> N_dataset: {N_dataset} | N_pdf: {N_pdf}")
        dist_matrix_prec = compute_geo_distances_to_gt_poses(
            ds, trainer, N_dataset=N_dataset, N_pdf=N_pdf, plot_curve=plot
        )
        if save:
            file_dir = f"{experiment_path}/distance_matrix_data_{N_dataset}_pdf_{N_pdf}{add_string_to_name}.p"
            with open(file_dir, "wb") as handle:
                pickle.dump(dist_matrix_prec, handle, protocol=pickle.HIGHEST_PROTOCOL)

        if not return_matrix:
            del dist_matrix_prec
            torch.cuda.empty_cache()
            return torch.tensor([1]), torch.tensor([1])
        else:
            return dist_matrix_recall, dist_matrix_prec
