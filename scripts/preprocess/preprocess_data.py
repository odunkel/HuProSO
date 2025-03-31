import argparse
import numpy as np
import logging
from tqdm import tqdm
import pickle
import logging
import os
import roma
import torch

from hp.data.load_amass import AmassLoader


logging.basicConfig(level=logging.INFO)

N_joints = 165 // 3


def main():
    os.chdir(f"{os.path.dirname(__file__)}/..")
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Name of folder of dataset.")
    parser.add_argument("--data_dir", type=str, help="Directory of data")
    parser.add_argument("--save_dir", type=str, help="Directory of pickled data")
    args = parser.parse_args()

    logging.info(f"Processing dataset {args.dataset}.")

    filedir = args.save_dir
    filename_rotations = f"rotations_{args.dataset.lower()}_xn.pt"

    dataset = args.dataset
    ALoader = AmassLoader(
        unpacked_directory=args.data_dir,
        datasets=[
            dataset,
        ],
    )
    N_frames, _ = ALoader.get_number_of_frames()

    poses_ax_angles = np.zeros((N_frames, N_joints, 3))

    i_frame = 0
    for n_subject, subject in enumerate(tqdm(ALoader)):
        N = subject["poses"].shape[0]
        for i_sample in range(N):
            try:
                data = subject["poses"][i_sample]
            except:
                logging.warning(
                    f"Failed to load data for subject {n_subject} sample {i_sample}. Continuing."
                )
                continue
            axis_angle = data[0 : N_joints * 3].reshape(N_joints, 3)
            poses_ax_angles[i_frame] = axis_angle.reshape(N_joints, 3)
            i_frame += 1

    poses_ax_angles = poses_ax_angles[:i_frame]

    logging.info("Number of samples:", i_frame)

    qs = roma.rotvec_to_unitquat(torch.tensor(poses_ax_angles)).numpy()

    save_dir = filedir + filename_rotations

    if not (filename_rotations in os.listdir(filedir)):
        torch.save(qs, f"{save_dir}")
        logging.info(f"Saved rotations to {save_dir}.")


if __name__ == "__main__":
    main()
