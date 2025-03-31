import torch
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from omegaconf import OmegaConf
import pickle

from hp.core.evaluator import PriorEvaluator
from hp.utils.saving_tools import prepare_saving_process
from hp.core.cond_trainer import CondSO3PriorTrainer
from hp.core.evaluator import IKEvaluator

import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def main(config: OmegaConf = None, save: bool = True):

    if config is None:
        config = OmegaConf.load("./config/config.yaml")
        config.selected_joints = "None"

        config.standard_mode = True
        config.data.dataset = "Train"
        config.valid.dataset = "Valid"
        config.eval.dataset = "Eval"
    
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    config.so3.condition = True
    config.valid.inv_kin = False

    print(f"==== TRAINING OF EXP {config.exp_id} ====")

    if save:
        config = prepare_saving_process(config)

    trainer = CondSO3PriorTrainer(config)
    trainer.writer = SummaryWriter(
        log_dir=f"{config.save.path_dir}/tensorboard/{config.trainer}_{config.rotation_representation}"
    )

    loss, valid_loss_with_iter = trainer.train()

    if save:
        trainer.save_model()

    with open(f"{config.save.path_dir}/loss_dict.p", "wb") as fp:
        save_d = {"loss": loss, "valid_metrics": valid_loss_with_iter}
        pickle.dump(save_d, fp, protocol=pickle.HIGHEST_PROTOCOL)

    print(trainer.eval_estimator(num_samples=5_000))

    print(IKEvaluator.eval_with_dataset(trainer, n_samples=5_000, dataset="eval"))


if __name__ == "__main__":
    main()
