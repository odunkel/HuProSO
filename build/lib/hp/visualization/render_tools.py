from human_body_prior.body_model.body_model import BodyModel
from os import path as osp
import torch
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation
from os import path as osp

import numpy as np
import matplotlib.pyplot as plt
import torch

from body_visualizer.tools.vis_tools import render_smpl_params
from body_visualizer.tools.vis_tools import imagearray2file
from body_visualizer.tools.vis_tools import show_image

from hp.data.amass_cfg import AmassJoints

RENDER_WIDTH = 360
RENDER_HEIGHT = 400
dpi = 100


class HumanRender:
    def __init__(self, device: str = "cpu") -> None:
        support_dir = "../human_body_prior/support_data/dowloads/"
        expr_dir = osp.join(
            support_dir, "V02_05"
        )  #'TRAINED_MODEL_DIRECTORY'  in this directory the trained model along with the model code exist
        self.device = device
        bm_fname = osp.join(
            support_dir, "models/smplx/SMPLX_NEUTRAL.npz"
        )  #'PATH_TO_SMPLX_model.npz'  obtain from https://smpl-x.is.tue.mpg.de/downloads
        sample_amass_fname = osp.join(
            support_dir, "amass_sample.npz"
        )  # a sample npz file from AMASS
        self.bm = BodyModel(bm_fname=bm_fname).to(device)

    def render_human(
        self,
        pose_body: np.ndarray,
        N_plot: int = 1,
        dpi=150,
        show_img=True,
        w=None,
        h=None,
    ) -> None:
        if w is None:
            w = RENDER_WIDTH
        if h is None:
            h = RENDER_HEIGHT
        print(f"Rendering with {w}x{h}, DPI={dpi}")
        images = render_smpl_params(
            self.bm,
            {"pose_body": pose_body.reshape(-1, 63)},
            rot_body=np.array([0, 0, 0]),
            imw_imh=(w, h),
        ).reshape(1, N_plot, 1, h, w, 3)
        img = imagearray2file(images)
        if show_img:
            show_image(img[0], dpi=dpi)
        else:
            return img, images

    def get_vertices_from_pose(self, pose_body):
        root_orient = torch.zeros((1, 3)).to(self.device)
        vertices = self.bm(
            root_orient=root_orient, pose_body=pose_body.reshape(1, -1)
        ).v
        return vertices


def render_human(bm, pose_body: np.ndarray, N_plot: int = 1, show_img=True) -> None:
    images = render_smpl_params(
        bm,
        {"pose_body": pose_body.reshape(-1, 63)},
        imw_imh=(RENDER_WIDTH, RENDER_HEIGHT),
    ).reshape(1, N_plot, 1, RENDER_HEIGHT, RENDER_WIDTH, 3)
    img = imagearray2file(images)
    if show_img:
        show_image(img[0], dpi=dpi)

    return img, images


def render_human_from_pose_body(pose_body):
    support_dir = "../human_body_prior/support_data/dowloads/"
    bm_fname = osp.join(support_dir, "models/smplx/SMPLX_NEUTRAL.npz")
    bm = BodyModel(bm_fname=bm_fname)
    pose_body = torch.tensor(pose_body).float()
    render_human(bm, pose_body[1:])
