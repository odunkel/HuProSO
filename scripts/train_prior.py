from matplotlib import pyplot as plt
import torch
import numpy as np
from omegaconf import OmegaConf
import json

import importlib

from torch.utils.tensorboard import SummaryWriter

from hp.core.evaluator import PriorEvaluator
from hp.utils.saving_tools import prepare_saving_process

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

    config.so3.condition = False
    config.conditioning.conditioning_modality = ''
    config.valid.inv_kin = False

    print(f"==== TRAINING OF EXP {config.exp_id} ====")

    if save:
        config = prepare_saving_process(config)

    module = importlib.import_module("hp.core.trainer")
    class_ = getattr(module, config.trainer)
    trainer = class_(config)
    trainer.writer = SummaryWriter(
        log_dir=f"{config.save.path_dir}/tensorboard/{config.trainer}_{config.rotation_representation}"
    )

    loss, valid_loss_with_iter = trainer.train()

    if save:
        trainer.save_model()

    plt.figure()
    plt.plot(loss, label=f"train")
    plt.plot(valid_loss_with_iter[0], valid_loss_with_iter[1], "--", label=f"valid")
    plt.xlabel("Iteration")
    plt.ylabel("NLL Loss")
    plt.legend()
    if save:
        plt.savefig(f"{config.save.path_dir}/loss.png")
        plt.close()

    eval_avg_ll = trainer.eval()
    logging.info(eval_avg_ll)
    d = PriorEvaluator.eval(
        config,
        trainer,
        render_human=False,
        plot_samples=False,
        save_plot=False,
        compute_d_wasser=False,
    )
    logging.info(d)

    eval_dict = {
        "eval_avg_ll": eval_avg_ll,
        "loss": list(loss),
        "valid_metrics": {
            "iter": list(valid_loss_with_iter[0]),
            "valid_loss": list(valid_loss_with_iter[1]),
            "valid_d_wasser": list(valid_loss_with_iter[2]),
        },
    }
    if save:
        with open(f"{config.save.path_dir}/eval.json", "w") as fp:
            json.dump(eval_dict, fp)


if __name__ == "__main__":
    main()
