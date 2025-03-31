from omegaconf import OmegaConf
import os


def prepare_saving_process(config: OmegaConf) -> OmegaConf:
    new_path = f"{config.save.path_dir}/id_{config.exp_id:04}/"
    j = 0
    while os.path.exists(new_path):
        new_path = f"{config.save.path_dir}/id_{config.exp_id:04}_{j:01d}/"
        j += 1
        print(f"Renaming to {new_path}")
        if j > 9:
            raise ValueError(f"Experiment {config.exp_id} already exists.")
    config.save.path_dir = new_path
    os.makedirs(config.save.path_dir)

    OmegaConf.save(config, f"{config.save.path_dir}/config.yaml")

    return config
