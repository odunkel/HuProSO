from typing import List, Tuple
import os
import numpy as np
import logging


class AmassLoader:
    """Loader for AMASS data.
    First, unpacking needed: tar -xf CMU.tar.bz2 -C ../../data/AMASS/
    Then, this class can be used to load individual frames.
    """

    def __init__(
        self,
        unpacked_directory: str = "../data/AMASS",
        datasets: List[str] = ["ACCAD", "CMU", "HUMAN4D"],
    ) -> None:
        self.data_dir = unpacked_directory
        self.datasets = datasets
        self.data_dirs = []
        self.int_action_mapping = dict()
        self.get_all_files()

    def get_all_files(self):

        for dataset in self.datasets:
            dataset_dir = f"{self.data_dir}/{dataset}"
            for subject in os.listdir(dataset_dir):
                sub_dir = f"{dataset_dir}/{subject}"
                if ".gz" in sub_dir or "txt" in sub_dir:
                    continue
                for file_name in os.listdir(sub_dir):
                    if (
                        file_name.endswith(".npz")
                        and not file_name.startswith(".")
                        and ("neutral_stagei" not in file_name)
                    ):  # neutral_stagei can come in SMPLX
                        self.data_dirs.append(f"{sub_dir}/{file_name}")

    def get_number_of_frames(self) -> Tuple[int, dict]:
        num_samples = 0
        idx_sample_mapping = dict()
        for i, dd in enumerate(self.data_dirs):
            try:
                d = self[i]
            except:
                logging.warning(f"Error in file {i}: {dd}. Continuing.")
                continue
            if "trans" in d.__dict__["files"]:
                N_frames = d["trans"].shape[0]
            idx_sample_mapping[i] = N_frames
            k = (dd.split("/")[-2], dd.split("/")[-1])
            self.int_action_mapping[k] = np.arange(num_samples, num_samples + N_frames)
            num_samples += N_frames
        return num_samples, idx_sample_mapping

    def __len__(self):
        return len(self.data_dirs)

    def __getitem__(self, idx: int):
        # keys:'trans', 'gender','mocap_framerate','betas','dmpls','poses
        return np.load(self.data_dirs[idx])
