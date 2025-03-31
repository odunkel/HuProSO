from omegaconf import OmegaConf

import torch
import numpy as np
from tqdm import tqdm
from typing import Tuple
import logging

from typing import Any

from hp.data.amass_cfg import AmassJoints, AmassJoints4Learning
from hp.data.amass import AMASS
from hp.estimators.rotnorm.flow.flow_nd import get_nd_flow
from hp.estimators.rotnorm.flow.flow import get_flow
import hp.estimators.rotnorm.utils.sd as sd
from hp.core.evaluator import PriorEvaluator, IKEvaluator


class PriorTrainer:

    def __init__(self, config: OmegaConf) -> None:

        self.writer = None
        self.config = config
        logging.info(f"Creating {self.config.trainer}...")

        selected_joints = (
            AmassJoints4Learning
            if config.selected_joints == "None"
            else [AmassJoints[joint_str] for joint_str in config.selected_joints]
        )

        conditioning_modality = (
            config.conditioning.conditioning_modality
            if hasattr(config.conditioning, "conditioning_modality")
            else ""
        )
        conditioning_modality = (
            config.conditioning.feat_for_conditiong_mode
            if hasattr(config.conditioning, "feat_for_conditiong_mode")
            else conditioning_modality
        )

        self.only_eval = False
        if hasattr(config, "only_eval") and config.only_eval == True:
            print("Only evaluation mode.")
            self.only_eval = True
        else:
            self.dataset = AMASS(
                datasets=[],
                mode="train",
                rotation_representation=config.rotation_representation,
                selected_joints=selected_joints,
                conditioning_modality=conditioning_modality,
            )

            self.train_dataset = self.dataset

            self.val_dataset = AMASS(
                datasets=[],
                mode="valid",
                rotation_representation=config.rotation_representation,
                selected_joints=selected_joints,
                conditioning_modality=conditioning_modality,
            )

        evaluate = config.evaluate if hasattr(config, "evaluate") else True
        if evaluate:
            self.eval_dataset = AMASS(
                datasets=[],
                mode="test",
                rotation_representation=config.rotation_representation,
                selected_joints=selected_joints,
                conditioning_modality=conditioning_modality,
            )
        else:
            self.eval_dataset = None

        self.selected_joints = selected_joints
        self.N_joint = len(selected_joints)

        self.pdf_estimator = None
        self.create_estimator()

        if not self.only_eval:
            self.create_optimizer_and_scheduler()
        self.create_dataloaders()

    def print_model_overview(self) -> None:
        logging.info(
            f"Using {self.pdf_estimator.__class__.__name__} as pdf estimator with {self.number_of_param*1e-3}k parameters."
        )

    def load_model(self, end_str="") -> None:
        cpkt_path = f"{self.config.save.path_dir}/model_ckpt_pdf_estimator{end_str}.pt"
        state_dict_loaded = torch.load(cpkt_path, map_location=self.config.device)
        self.pdf_estimator.load_state_dict(state_dict_loaded)

    def save_model(self, end_str="") -> None:
        cpkt_path = f"{self.config.save.path_dir}/model_ckpt_pdf_estimator{end_str}.pt"
        torch.save(self.pdf_estimator.state_dict(), cpkt_path)
        print(f"Saved models to {self.config.save.path_dir}")

    def create_optimizer_and_scheduler(self) -> None:
        self.optimizer = torch.optim.Adam(
            self.pdf_estimator.parameters(), lr=self.config.lr
        )
        self.create_scheduler()

    def create_scheduler(self) -> None:
        iterations_by_epoch = int(
            self.config.num_epochs
            * np.ceil(len(self.train_dataset) / self.config.batch_size)
        )
        max_iter = min(iterations_by_epoch, self.config.max_iterations)
        if self.config.lr_scheduler:
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.lrs_step_size * max_iter,
                gamma=self.config.lr_scheduler_step_gamma,
            )
        self.max_iter = max_iter

    def create_dataloaders(self) -> None:

        if not self.only_eval:
            self.dataloader = torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                collate_fn=None,
            )
            self.val_dataloader = torch.utils.data.DataLoader(
                self.val_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                collate_fn=None,
            )
            logging.info(
                f"Validation is performed with {len(self.val_dataset)*1e-3}k samples. Training with {len(self.train_dataset)*1e-3}k samples."
            )

        if self.eval_dataset is not None:
            self.eval_dataloader = torch.utils.data.DataLoader(
                self.eval_dataset,
                batch_size=self.config.num_eval_samples,
                shuffle=True,
                collate_fn=None,
            )

    def create_estimator(self):
        raise NotImplementedError("This method is implemented in the child class.")

    def train(self) -> Tuple[np.ndarray, np.ndarray]:

        self.pdf_estimator.train()
        logging.info(
            f"Training for {self.max_iter} iterations with batch size {self.config.batch_size} and {(self.max_iter * self.config.batch_size / len(self.train_dataset)):.02f} epochs."
            + f" {int(self.max_iter*self.config.batch_size*1e-3)}k times sampled from training dataset."
        )

        loss_hist = np.array([])
        valid_loss_hist = np.array([])
        valid_d_wasser_hist = np.array([])

        valid_iter = self.config.valid.interval
        stop_training = False
        it = 0
        with tqdm(total=self.max_iter) as pbar:
            for i_epoch in range(self.config.num_epochs):
                for it_in_epoch, x in enumerate(self.dataloader):
                    loss = self.training_step(x)
                    # Log loss
                    loss_hist = np.append(loss_hist, loss.to("cpu").data.numpy())
                    if self.writer is not None:
                        self.writer.add_scalar(
                            "Loss/train", loss.to("cpu").data.numpy(), it
                        )
                    if ((it + 1) % valid_iter == 0) and self.val_dataloader is not None:
                        valid_metrics = self.validate()
                        valid_loss = valid_metrics["nll"]
                        valid_loss_hist = np.append(valid_loss_hist, valid_loss)
                        valid_d_wasser_hist = np.append(
                            valid_d_wasser_hist, valid_metrics["d_wasser"]
                        )
                        if self.writer is not None:
                            self.writer.add_scalar("Loss/val", valid_loss, it)
                            if self.config.valid.compute_wasser:
                                self.writer.add_scalar(
                                    "Wasser/val", valid_metrics["d_wasser"], it
                                )
                            if self.config.valid.inv_kin:
                                self.writer.add_scalar(
                                    "MPJPE/val", valid_metrics["mpjpe"], it
                                )
                                self.writer.add_scalar(
                                    "Geod_dist/val", valid_metrics["geod_dist"], it
                                )
                                self.writer.add_scalar(
                                    "MPJPE/train", valid_metrics["mpjpe_train"], it
                                )
                                self.writer.add_scalar(
                                    "Geod_dist/train",
                                    valid_metrics["geod_dist_train"],
                                    it,
                                )

                        print_str = (
                            f"Valid loss: {valid_loss:.02f} | Train loss: {loss:.02f}"
                        )
                        if self.config.valid.compute_wasser:
                            print_str += f" | Wasser: {valid_metrics['d_wasser']:.02f}"
                        if self.config.valid.inv_kin:
                            print_str += f" | MPJPE: {1000*valid_metrics['mpjpe']:.02f}mm | Geod_dist: {valid_metrics['geod_dist']:.03f}"
                            print_str += (
                                f" | MPJPE_2D: {1000*valid_metrics['mpjpe_2d']:.02f}mm"
                            )
                            print_str += f" | MPJPE TRAIN: {1000*valid_metrics['mpjpe_train']:.02f}mm | Geod_dist TRAIN: {valid_metrics['geod_dist_train']:.03f}"
                        logging.info(print_str)

                    if (it + 1) % self.config.save_ckpt_every == 0:
                        self.save_model(end_str=f"_{it:05d}")

                    if it >= self.max_iter:
                        stop_training = True
                        break
                    pbar.update(1)
                    it += 1

                if stop_training:
                    break
        if stop_training:
            logging.info(f"\nFinished training after {it} iterations.")

        valid_metrics_with_iter = np.zeros((3, len(valid_loss_hist)))
        valid_metrics_with_iter[0] = (
            np.arange(1, len(valid_loss_hist) + 1) * self.config.valid.interval
        )
        valid_metrics_with_iter[1] = valid_loss_hist
        valid_metrics_with_iter[2] = valid_d_wasser_hist
        return loss_hist, valid_metrics_with_iter

    def validate(self) -> float:
        self.pdf_estimator.eval()
        losses = []
        with torch.no_grad():
            for it, x in enumerate(self.val_dataloader):
                x = self.to_device(x)
                kld = self.forward_kld(x)
                loss = kld.cpu().data.numpy()
                losses.append(loss)
        loss_mean = np.mean(np.array(losses))
        if self.config.valid.compute_wasser:
            try:
                d_wasser = PriorEvaluator.compute_wasserstein_distance(
                    self.config, self, mode="valid"
                )
            except:
                logging.warning(
                    "Could not compute wasserstein distance in validation. Set it to -1."
                )
                d_wasser = -1
        else:
            d_wasser = -1

        d = {"nll": loss_mean}
        mpjpe = -1
        geod_dist = -1
        mp2d = -1
        mp2d_train = -1
        mpjpe_train = -1
        geod_dist_train = -1
        if (
            self.config.valid.inv_kin
            and self.config.conditioning.conditioning_modality
            in ["2D", "3D", "rotation"]
        ):
            d_eval = IKEvaluator.eval_with_dataset(
                self, n_samples=self.config.num_eval_samples, dataset="valid"
            )
            mpjpe, geod_dist, mp2d = (
                d_eval["mpjpe"],
                d_eval["geod_dist"],
                d_eval["mpjpe_2d"],
            )
            d_eval_train = IKEvaluator.eval_with_dataset(
                self, n_samples=self.config.num_eval_samples, dataset="train"
            )
            mpjpe_train, geod_dist_train, mp2d_train = (
                d_eval_train["mpjpe"],
                d_eval_train["geod_dist"],
                d_eval_train["mpjpe_2d"],
            )
        d.update(
            {
                "d_wasser": d_wasser,
                "mpjpe": mpjpe,
                "geod_dist": geod_dist,
                "mpjpe_2d": mp2d,
                "mpjpe_train": mpjpe_train,
                "geod_dist_train": geod_dist_train,
                "mpjpe_2d_train": mp2d_train,
            }
        )
        self.pdf_estimator.train()
        return d

    def eval(self, complete_ds: bool = False) -> dict:
        if complete_ds:
            lls = []
            for it, x in enumerate(self.eval_dataloader):
                with torch.no_grad():
                    x = self.to_device(x)
                    ll = self.forward_estimator(x)
                lls.append(ll)
            eval_avg_ll = torch.mean(lls).item()
        else:
            eval_avg_ll = self.eval_estimator(self.config.num_eval_samples)
        eval_dict = {"eval_avg_ll": eval_avg_ll}
        return eval_dict

    def sample(self, num_samples: int = 1_000) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sample_from_estimator(num_samples)

    def forward(self, samples: torch.Tensor) -> torch.Tensor:
        return self.forward_estimator(samples)

    def forward_kld(self, samples: torch.Tensor) -> torch.Tensor:
        log_pdf = self.forward(samples)
        l = -torch.nanmean(log_pdf)
        return l

    def to_device(self, element: Any, device=None) -> Any:
        if device is None:
            device = self.config.device
        if type(element) == dict:
            element = {key: value.to(device) for key, value in element.items()}
        else:
            element = element.to(device)
        return element

    def training_step(self, x: torch.Tensor) -> float:
        self.optimizer.zero_grad()
        x = self.to_device(x)

        loss = self.forward_kld(x)

        if ~(torch.isnan(loss) | torch.isinf(loss)):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.pdf_estimator.parameters(), self.config.clip_grad_norm
            )
            self.optimizer.step()
            if self.config.lr_scheduler:
                self.scheduler.step()
        return loss

    def forward_estimator(self, samples: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("This method is implemented in the child class.")

    def sample_from_estimator(
        self, num_samples: int = 1_000
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("This method is implemented in the child class.")

    def eval_estimator(self, num_samples: int = 1_000, mode="eval") -> float:
        ds = self.eval_dataset if mode == "eval" else self.train_dataset
        with torch.no_grad():
            if type(ds) == torch.utils.data.dataset.Subset:
                N = len(ds)
                idx = np.random.choice(np.arange(N), num_samples, replace=False)
                x = ds[list(idx)].reshape(num_samples, -1).to(self.config.device)
            else:
                x = ds.sample(num_samples).to(self.config.device)
            log_likelihood = self.forward_estimator(x)
        return torch.mean(log_likelihood).item()


class SO3PriorTrainer(PriorTrainer):
    def __init__(self, config: OmegaConf) -> None:
        super().__init__(config)
        self.config = config
        logging.info(
            f"Using {self.pdf_estimator.__class__.__name__} as pdf estimator with {self.number_of_param*1e-3}k parameters."
        )

    @property
    def number_of_param(self) -> int:
        return sum(
            p.numel() for p in self.pdf_estimator.parameters() if p.requires_grad
        )

    def create_estimator(self):
        config_rotnorm = self.config.so3
        config_rotnorm.n_dim = self.N_joint
        if self.N_joint == 1:
            self.pdf_estimator = get_flow(config_rotnorm).to(self.config.device)
        else:
            self.pdf_estimator = get_nd_flow(config_rotnorm).to(self.config.device)

    def forward_estimator(self, samples: torch.Tensor) -> torch.Tensor:
        if self.N_joint == 1:
            rotation, log_probs = self.pdf_estimator(samples.reshape(-1, 3, 3))
        else:
            rotation, log_probs = self.pdf_estimator(
                samples.reshape(-1, self.N_joint, 3, 3)
            )
        return log_probs

    def sample_from_estimator(
        self, num_samples: int = 1_000
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
            samples, log_probs = self.pdf_estimator.inverse(sample)
        return samples, log_probs

    def save_model(self, end_str="") -> None:
        cpkt_path = f"{self.config.save.path_dir}/model_ckpt_pdf_estimator{end_str}.pt"
        torch.save(self.pdf_estimator.state_dict(), cpkt_path)
        np.savetxt(
            f"{self.config.save.path_dir}/permute_dims.pt",
            self.pdf_estimator.permute_dims,
        )
        logging.info(f"Saved models and permute dims to {self.config.save.path_dir}.")

    def load_model(self, save_path="", end_str="", device="cuda:0") -> None:
        save_path = self.config.save.path_dir if save_path == "" else save_path
        cpkt_path = f"{save_path}/model_ckpt_pdf_estimator{end_str}.pt"
        state_dict_loaded = torch.load(cpkt_path, map_location=device)
        self.pdf_estimator.load_state_dict(state_dict_loaded)
        self.pdf_estimator.permute_dims = np.loadtxt(f"{save_path}/permute_dims.pt")
