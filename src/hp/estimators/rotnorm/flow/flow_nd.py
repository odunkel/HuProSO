import torch
from torch import nn
import numpy as np
from typing import Tuple
import logging

from hp.estimators.rotnorm.flow.mobiusflow import get_mobius, MobiusFlow
from hp.estimators.rotnorm.flow.affineflow import get_affine
from hp.estimators.rotnorm.flow.condition import ConditionalTransformND


def get_nd_flow(config):
    return FlowND(config)


_permute_prop = torch.Tensor(
    [[0, 1, 2], [1, 2, 0], [2, 0, 1], [0, 1, 2], [1, 2, 0], [2, 0, 1]]
).type(torch.long)


class FlowND(nn.Module):
    def __init__(
        self,
        config,
    ):
        super(FlowND, self).__init__()
        self.config = config
        self.layers = config.layers
        self.condition_with_feature = config.condition
        self._permute = _permute_prop

        self.n_dim = config.n_dim
        self.cond_feature_dim = (
            self.config.feature_channels if self.condition_with_feature else 0
        )
        self.dims_of_feat = [i * 3 * 2 for i in range(self.n_dim)]

        layers = (
            []
        )  # layers: [flow_0_layer_0, flow_0_layer_1, flow_0_layer_2,...flow_1_layer_0, flow_1_layer_1, flow_1_layer_2, ...]
        self.independent = config.independent
        for k in range(self.n_dim):

            if config.nd_conditioning:
                input_dim = (
                    self.config.feature_channels
                    + config.feature_dim * self.condition_with_feature
                )
                output_dim = config.segments * 4 * config.n_dim
                self.mcl_conditioner = ConditionalTransformND(
                    input_dim, output_dim, Nh=512, num_hidden_layers=3
                )

            else:
                if k == 0 or self.independent:
                    feature_dim = 0 + self.cond_feature_dim * config.condition
                    config.condition = config.condition
                else:
                    config.condition = 1
                    feature_dim = (
                        self.dims_of_feat[k] + self.cond_feature_dim * config.condition
                    )

            if config.last_affine:
                layers.append(
                    get_affine(config, feature_dim, first_layer_condition=True)
                )

            for i in range(self.layers):
                tmp = get_mobius(config, feature_dim)
                if tmp != None:
                    layers.append(tmp)

                if hasattr(config, "only_mobius") and config.only_mobius:
                    tmp = get_mobius(config, feature_dim)
                    if tmp != None:
                        layers.append(tmp)
                else:
                    tmp = get_affine(config, feature_dim)
                    if tmp != None and (i != self.layers - 1 or config.first_affine):
                        layers.append(tmp)

        self.num_layers_per_dim = len(layers) // self.n_dim
        logging.info(
            f"Layers of flow: {self.num_layers_per_dim}. Number of dimensions: {self.n_dim}."
        )
        self.layers = nn.ModuleList(layers)

        self.permute_dims = np.stack(
            [
                np.random.choice(np.arange(self.n_dim), self.n_dim, replace=False)
                for _ in range(self.num_layers_per_dim)
            ]
        )

    @property
    def flow_type(self):
        return "FlowND"

    def get_rotation_and_feature(
        self, rotation_nd, k_dim
    ) -> Tuple[torch.tensor, torch.tensor]:
        rotation = rotation_nd[:, k_dim, :, :]
        if self.independent:
            feature = None
        else:
            feature = rotation_nd[:, :k_dim, :, :2].reshape(rotation_nd.shape[0], -1)

        return rotation, feature

    def inverse(self, rotation, feature=None):
        return self.forward(rotation, feature, inverse=True)

    def forward(self, rotation_nd, feature=None, inverse=False, plot=False):
        return self.forward_v2(rotation_nd, feature, inverse, plot)

    def forward_v2(self, rotation_nd, feature_cond=None, inverse=False, plot=False):
        # rotation: (B,N_dim,3,3)
        # feature_cond: (B,N_dim,3)
        permute = self._permute.to(rotation_nd.device)

        ldjs = 0
        exchange_count = 0

        lps = np.zeros((self.n_dim, self.num_layers_per_dim))
        if inverse:
            exchange_count = (
                len(self.layers) if self.config.frequent_permute else self.config.layers
            )
            for i in range(self.num_layers_per_dim)[::-1]:
                permute_dims = self.permute_dims[i]
                rotation_nd = rotation_nd.clone()[:, permute_dims, :, :]

                rotations_k = []
                for k in range(self.n_dim):
                    if k == 0 or self.independent:
                        feature = feature_cond
                    else:
                        feature_dim = torch.stack(rotations_k, dim=1)[
                            :, :k, :, :2
                        ].reshape(-1, k * 2 * 3)

                        if feature_cond is None or self.condition_with_feature == 0:
                            feature = feature_dim
                        else:
                            feature = torch.cat((feature_dim, feature_cond), dim=-1)

                    rotation = rotation_nd[:, k, :, :]

                    indx = k * self.num_layers_per_dim + i

                    if (
                        (isinstance(self.layers[indx], MobiusFlow))
                        or self.config.frequent_permute
                    ) and k == 0:  # k+1==self.n_dim:
                        exchange_count -= 1
                    rotation, ldj = self.layers[indx].inverse(
                        rotation, permute[exchange_count % 6], feature
                    )
                    ldjs += ldj
                    rotations_k.append(rotation)
                    lps[int(permute_dims[k]), i] = ldj.mean().item()
                rotation_nd = torch.stack(rotations_k, dim=1)
                rotation_nd[:, permute_dims] = rotation_nd.clone()
            nll_dims = np.sum(lps, axis=1)
        else:
            for i in range(self.num_layers_per_dim):
                rotations_k = []

                permute_dims = self.permute_dims[i].astype(int)
                rotation_nd = rotation_nd.clone()[:, permute_dims, :, :]

                for k in range(self.n_dim):
                    rotation, feature_dim = self.get_rotation_and_feature(
                        rotation_nd, k
                    )
                    if feature_cond is None or self.condition_with_feature == 0:
                        feature = feature_dim
                    else:
                        feature = torch.cat((feature_dim, feature_cond), dim=-1)

                    indx = k * self.num_layers_per_dim + i
                    rotation, ldj = self.layers[indx](
                        rotation,
                        permute[exchange_count % 6],
                        feature,
                    )
                    if hasattr(self.config, "clamp_value"):
                        ldj = torch.clamp_(ldj, max=self.config.clamp_value)

                    ldjs += ldj
                    rotations_k.append(rotation)
                    lps[int(permute_dims[k]), i] = ldj.mean().item()
                rotation_nd = torch.stack(rotations_k, dim=1)

                rotation_nd[:, permute_dims] = rotation_nd.clone()

                if (
                    isinstance(self.layers[indx], MobiusFlow)
                ) or self.config.frequent_permute:
                    exchange_count += 1
            nll_dims = np.sum(lps, axis=1)

        if rotation_nd.shape[1] == 1 and self.flow_type != "FlowND":
            rotation_nd = rotation_nd[0]

        return rotation_nd, ldjs
