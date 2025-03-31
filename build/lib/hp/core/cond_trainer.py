from omegaconf import OmegaConf
import torch
import numpy as np
from typing import Tuple

from hp.core.trainer import PriorTrainer
import hp.estimators.rotnorm.utils.sd as sd
from hp.estimators.simple_mlp import MLPTransform
from hp.estimators.rotnorm.flow.flow_nd import get_nd_flow
from hp.estimators.rotnorm.flow.flow import get_flow
from hp.data.amass import AmassJoints


class CondSO3PriorTrainer(PriorTrainer):
    def __init__(self, config: OmegaConf) -> None:
        print(f"Creating CondSO3PriorTrainer...")
        super().__init__(config)
        self.config = config
        print(
            f"Using {self.pdf_estimator.__class__.__name__} as pdf estimator with {self.number_of_param*1e-3}k parameters."
        )
        print(f"Conditioning mode: {config.conditioning.conditioning_modality}")
        print("Created CondSO3PriorTrainer.")

    @property
    def number_of_param(self) -> int:
        params = list(self.pdf_estimator.parameters()) + list(
            self.feature_transform.parameters()
        )
        return sum(p.numel() for p in params if p.requires_grad)

    def create_feat_conditioner(self):
        in_dim = self.config.so3.dim_cond_feature  # self.config.so3.n_dim
        self.feature_transform = MLPTransform(
            in_dim,
            self.config.so3.feature_channels,
            Nh=self.config.conditioning.hidden_mlp_dims,
            num_hid_layers=self.config.conditioning.num_hid_layers,
        ).to(self.config.device)

    def create_optimizer_and_scheduler(self) -> None:
        lr = self.config.lr
        print("Learning rate: ", lr)
        params = list(self.pdf_estimator.parameters()) + list(
            self.feature_transform.parameters()
        )
        self.optimizer = torch.optim.Adam(params, lr=lr)
        self.create_scheduler()

    def create_estimator(self):
        config_rotnorm = self.config.so3
        config_rotnorm.n_dim = self.N_joint
        config_rotnorm.save_path_dir = self.config.save.path_dir
        if self.N_joint == 1:
            self.pdf_estimator = get_flow(config_rotnorm).to(self.config.device)
        else:
            if self.config.trainer == "SO3PriorTrainer":
                self.pdf_estimator = get_nd_flow(config_rotnorm).to(self.config.device)
            else:
                raise NotImplementedError(
                    f"Trainer {self.config.trainer} not implemented."
                )
        self.create_feat_conditioner()

    def get_6D_bound_characteristics(self) -> Tuple[np.ndarray]:
        bound_dim_inds = np.arange(self.N_joint * 6)
        bound_scales = 2 * np.ones(self.N_joint * 6)
        circ_dim_inds = np.array([])
        return bound_dim_inds, bound_scales, circ_dim_inds

    def save_model(self, end_str="") -> None:
        cpkt_path = f"{self.config.save.path_dir}/model_ckpt_pdf_estimator{end_str}.pt"
        torch.save(self.pdf_estimator.state_dict(), cpkt_path)
        cpkt_path = (
            f"{self.config.save.path_dir}/model_ckpt_feature_extractor{end_str}.pt"
        )
        torch.save(self.feature_transform.state_dict(), cpkt_path)
        if self.config.trainer != "Cont6DPriorTrainer":
            np.savetxt(
                f"{self.config.save.path_dir}/permute_dims.pt",
                self.pdf_estimator.permute_dims,
            )
        print(f"Saved models and permute dims to {self.config.save.path_dir}")

    def load_model(self, end_str="") -> None:
        cpkt_path = f"{self.config.save.path_dir}/model_ckpt_pdf_estimator{end_str}.pt"
        state_dict_loaded = torch.load(cpkt_path, map_location=self.config.device)
        self.pdf_estimator.load_state_dict(state_dict_loaded)
        cpkt_path = (
            f"{self.config.save.path_dir}/model_ckpt_feature_extractor{end_str}.pt"
        )
        state_dict_loaded = torch.load(cpkt_path, map_location=self.config.device)
        self.feature_transform.load_state_dict(state_dict_loaded)
        if self.config.trainer != "Cont6DPriorTrainer":
            self.pdf_estimator.permute_dims = np.loadtxt(
                f"{self.config.save.path_dir}/permute_dims.pt"
            )

    def compute_feature(self, data):
        if data is None:
            return None

        data = data.reshape(data.shape[0], -1)
        if self.config.conditioning.mask:
            data = mask_data(data, self.config.conditioning)
        feature = self.feature_transform(data)

        return feature

    def forward_estimator(self, data: torch.tensor) -> torch.tensor:
        # Returns the log-probability.
        samples = data["theta"]
        if self.config.rotation_representation == "6D":
            samples = samples.reshape(-1, self.N_joint, 6)
        else:
            samples = samples.reshape(-1, self.N_joint, 3, 3)

        condition = data["condition"]
        feature = self.compute_feature(condition)

        bs = samples.shape[0]
        if self.config.trainer == "Cont6DPriorTrainer":
            log_probs = self.pdf_estimator.log_prob(
                samples.reshape(bs, -1), feature.reshape(bs, -1)
            )
        else:
            rotation, log_probs = self.pdf_estimator(samples, feature=feature)

        return log_probs

    def sample_from_estimator(
        self, num_samples: int, condition: torch.tensor
    ) -> Tuple[torch.tensor, torch.tensor]:
        if num_samples != len(condition):
            raise ValueError("Number of samples and condition must be the same.")

        if self.config.trainer == "Cont6DPriorTrainer":
            condition = condition.reshape(num_samples, -1)
            with torch.no_grad():
                feature = self.compute_feature(condition)
                samples, log_prob = self.pdf_estimator.sample(num_samples, feature)
            return samples, log_prob

        if self.config.so3.n_dim == 1:
            sample = sd.generate_queries(num_samples, mode="random").to(
                self.config.device
            )
        else:
            sample = (
                torch.stack(
                    [
                        sd.generate_queries(num_samples, mode="random")
                        for _ in range(self.config.so3.n_dim)
                    ]
                )
                .to(self.config.device)
                .permute(1, 0, 2, 3)
            )
        with torch.no_grad():
            feature = self.compute_feature(condition)
            samples, log_prob = self.pdf_estimator.inverse(sample, feature)
        return samples, log_prob

    def sample(
        self, num_samples: int = 1_000, return_condition: bool = False
    ) -> Tuple[torch.tensor, torch.tensor]:
        idx = np.random.choice(np.arange(num_samples), num_samples, replace=False)
        condition = self.eval_dataset[list(idx)]["condition"].to(self.config.device)
        samples = self.sample_from_estimator(num_samples, condition)
        if return_condition:
            return samples[0], samples[1], condition.reshape(num_samples, -1, 3)
        else:
            return samples

    def eval_estimator(self, num_samples: int = 2_000, complete=False) -> float:
        with torch.no_grad():
            if complete:
                x = self.to_device(self.eval_dataset[:])
                num_samples = len(self.eval_dataset)
                x["theta"] = x["theta"].reshape(num_samples, -1)
            else:
                if type(self.eval_dataset) == torch.utils.data.dataset.Subset:
                    N = len(self.eval_dataset)
                    idx = np.random.choice(np.arange(N), num_samples, replace=False)
                    x = self.to_device(self.eval_dataset[list(idx)])
                    x["theta"] = x["theta"].reshape(num_samples, -1)
                else:
                    x = self.to_device(self.eval_dataset.sample(num_samples))

            log_likelihood = self.forward_estimator(x)
        return torch.mean(log_likelihood).item()


def mask_data(data, cond_config):
    if cond_config.mask_type == "random":
        if cond_config.vary_mask_prob:
            pms = torch.rand(data.shape[0], device=data.device)
        else:
            p_m = cond_config.zero_mask_prob
            pms = p_m * torch.ones(data.shape[0], device=data.device)

        if cond_config.conditioning_modality == "3D":
            a = torch.rand((data.shape[0], 22), device=data.device)
            mask = torch.stack([a[i] > pms[i] for i in range(data.shape[0])])
            mask = mask.repeat_interleave(3, 1)
        elif cond_config.conditioning_modality == "rotation":
            a = torch.rand((data.shape[0], 19), device=data.device)
            mask = torch.stack([a[i] > pms[i] for i in range(data.shape[0])])
            mask = (
                mask.reshape(-1, 19, 1, 1)
                .repeat_interleave(3, 2)
                .repeat_interleave(2, 3)
                .to(data.device)
                .reshape(data.shape[0], -1)
            )
        elif cond_config.conditioning_modality == "2D":
            a = torch.rand((data.shape[0], 22), device=data.device)
            mask = torch.stack([a[i] > pms[i] for i in range(data.shape[0])])
            mask = mask.repeat_interleave(2, 1)
        data = data * mask
        if hasattr(cond_config, "mask_neg_offset") and cond_config.mask_neg_offset:
            data[mask] -= 1e5
    elif cond_config.mask_type == "selected_joints":
        if not hasattr(cond_config, "mask_joints"):
            raise ValueError("You have to specify the joints to mask in the config.")

        sel_joints = cond_config.mask_joints
        mask_joints = [j.value for j in AmassJoints if j.name in sel_joints]
        data = data.reshape(data.shape[0], 22, -1)
        data[:, mask_joints] = 0
        data = data.reshape(data.shape[0], -1)
        if hasattr(cond_config, "mask_neg_offset") and cond_config.mask_neg_offset:
            data[:, mask_joints] = -1e5
    else:
        raise NotImplementedError(f"Mask type {cond_config.mask_type} not implemented.")

    return data
