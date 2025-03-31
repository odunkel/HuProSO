# Partially taken from https://github.com/PKU-EPIC/RotationNormFlow

import torch
from torch import nn
import hp.estimators.rotnorm.flow as flow


def get_mobius(config, feature_dim):
    if config.dist == "noflow":
        return None
    else:
        if hasattr(config, "hidden_dims_moebius_conditioner"):
            Nh = config.hidden_dims_moebius_conditioner
            num_hidden_layers = config.num_hidden_layers_moebius_conditioner
        else:
            Nh, num_hidden_layers = 16, 3
        return MobiusFlow(
            3,
            config.segments,
            condition=config.condition,
            feature_dim=feature_dim,
            Nh=Nh,
            num_hidden_layers=num_hidden_layers,
        )


def _h(z, w, D=3):
    w_norm = torch.norm(w, dim=-1, keepdim=True)  # n x k x 1

    h_z = (1 - w_norm**2) / (
        torch.norm((z.reshape(-1, 1, D) - w), dim=-1, keepdim=True) ** 2
    ) * (z.reshape(-1, 1, D) - w) - w

    return h_z


def _h_nd(z, w, D=3):

    w_norm = torch.norm(w, dim=-1, keepdim=True)  # n x k x 1

    # z is reshaped to shape: n x m x 1 x D to be broadcastable with w
    h_z = (1 - w_norm**2) / (
        torch.norm((z.reshape(-1, z.shape[1], 1, D) - w), dim=-1, keepdim=True) ** 2
    ) * (z.reshape(-1, z.shape[1], 1, D) - w) - w

    return h_z


class MobiusFlow(nn.Module):
    def __init__(self, D, K, condition=0, feature_dim=None, Nh=16, num_hidden_layers=3):
        super(MobiusFlow, self).__init__()

        self.D = D
        self.K = K

        self.condition = condition
        self.feature_dim = feature_dim

        if self.condition:
            input_dim = D + self.feature_dim
        else:
            input_dim = D

        self.conditioner = flow.condition.ConditionalTransform(
            input_dim, 4 * K, Nh=Nh, num_hidden_layers=num_hidden_layers
        )
        self.register_buffer("_I", torch.eye(D), persistent=False)

    def forward(self, rotation, permute=None, feature=None, dof1=False):
        assert permute != None, "The permuting function is needed in this module"
        if self.condition:
            assert feature != None, "The input feature is needed in this module"

        if dof1:
            if permute[0] == 0:
                return rotation, torch.zeros(rotation.shape[0]).to(rotation.device)

        x = rotation[..., permute[0]]
        y = rotation[..., permute[1]]

        if self.condition:
            condition_input = torch.cat((y, feature), dim=-1)
        else:
            condition_input = y
        conditions = self.conditioner(condition_input)
        weights, w = torch.split(conditions, [self.K, 3 * self.K], dim=1)
        w = w.reshape(-1, self.K, self.D)
        proj = self._I[None, ...] - torch.einsum("ni,nj->nij", y, y)
        w = torch.einsum("nij,nkj->nki", proj, w)
        r = -x
        r = r / r.norm(dim=-1, keepdim=True)
        v = torch.cross(y, r, dim=-1)
        v = v / v.norm(dim=-1, keepdim=True)

        weights = torch.nn.functional.softplus(weights)
        sum_weight = weights.sum(dim=-1, keepdim=True)
        weights = weights / sum_weight
        w = 0.7 / (1 + torch.norm(w, dim=-1, keepdim=True)) * w

        tx, ldj = self._forward(x, r, v, weights, w)
        if (permute[1] - permute[0]) == 1 or (permute[1] - permute[0]) == -2:
            tz = torch.cross(tx, y, dim=-1)
        else:
            tz = torch.cross(y, tx, dim=-1)
        tz = tz / tz.norm(dim=-1, keepdim=True)
        trotation = torch.empty(rotation.size()).to(rotation.device)
        trotation[..., permute[0]] = tx
        trotation[..., permute[1]] = y
        trotation[..., permute[2]] = tz

        return trotation, ldj

    def _h(self, z, w):
        return _h(z, w, self.D)

    def _forward(self, x, r, v, weights, w):
        z = x

        h_z = self._h(z, w)
        radians = torch.atan2(
            torch.einsum("nki,ni->nk", h_z, v), torch.einsum("nki,ni->nk", h_z, r)
        )
        tx = radians
        tx = torch.where(tx >= 0, tx, tx + torch.pi * 2)
        tx = torch.sum(weights * tx, dim=1, keepdim=True)

        tx = r * torch.cos(tx) + v * torch.sin(tx)

        z_w = z[:, None, :] - w
        z_w_norm = torch.norm(z_w, dim=-1)
        z_w_unit = z_w / z_w_norm[..., None]

        theta = torch.atan2(
            torch.einsum("ni,ni->n", x, v), torch.einsum("ni,ni->n", x, r)
        ).reshape(-1, 1)
        dz_dtheta = -torch.sin(theta) * r + torch.cos(theta) * v

        # n x k x 2 x 2
        dh_dz = (
            (1 - torch.norm(w, dim=-1) ** 2)[..., None, None]
            * (
                self._I[None, None, ...]
                - 2 * torch.einsum("nki,nkj->nkij", z_w_unit, z_w_unit)
            )
            / (z_w_norm[..., None, None] ** 2)
        )

        dh_dtheta = torch.einsum("nkpq,nq->nkp", dh_dz, dz_dtheta)
        dtx = torch.sum(torch.norm(dh_dtheta, dim=-1) * weights, dim=1)
        return tx, torch.log(dtx)

    def inverse(self, trotation, permute=None, feature=None, dof1=False):
        assert permute != None, "The permuting function is needed in this module"
        if self.condition:
            assert feature != None, "feature input is needed in this module"

        if dof1:
            if permute[0] == 0:
                return rotation, torch.zeros(trotation.shape[0]).to(rotation.device)

        tx = trotation[..., permute[0]]
        ty = trotation[..., permute[1]]

        if self.condition:
            condition_input = torch.cat((ty, feature), dim=-1)
        else:
            condition_input = ty

        conditions = self.conditioner(condition_input)
        weights, w = torch.split(conditions, [self.K, 3 * self.K], dim=1)
        w = w.reshape(-1, self.K, self.D)
        proj = self._I[None, ...] - torch.einsum("ni,nj->nij", ty, ty)
        w = torch.einsum("nij,nkj->nki", proj, w)

        r = -tx
        r = r / r.norm(dim=-1, keepdim=True)
        v = torch.cross(ty, r, dim=-1)
        v = v / v.norm(dim=-1, keepdim=True)
        weights = torch.nn.functional.softplus(weights)
        sum_weight = weights.sum(dim=-1, keepdim=True)
        weights = weights / sum_weight
        w = 0.7 / (1 + torch.norm(w, dim=-1, keepdim=True)) * w

        ttheta = torch.atan2(
            torch.einsum("ni,ni->n", tx, v), torch.einsum("ni,ni->n", tx, r)
        ).reshape(-1, 1)
        ttheta = torch.where(ttheta >= 0, ttheta, ttheta + torch.pi * 2)

        ttheta = torch.where(
            abs(ttheta - torch.pi * 2) < 1e-4,
            torch.zeros(ttheta.size(), dtype=ttheta.dtype, device=ttheta.device),
            ttheta,
        )

        theta = self._bin_find_root(ttheta, r, v, weights, w)
        x = r * torch.cos(theta) + v * torch.sin(theta)
        _, ldj = self._forward(x, r, v, weights, w)

        if (permute[1] - permute[0]) == 1 or (permute[1] - permute[0]) == -2:
            z = torch.cross(x, ty, dim=-1)
        else:
            z = torch.cross(ty, x, dim=-1)
        z = z / z.norm(dim=-1, keepdim=True)
        rotation = torch.empty(trotation.size()).to(trotation.device)
        rotation[..., permute[0]] = x
        rotation[..., permute[1]] = ty
        rotation[..., permute[2]] = z

        return rotation, -ldj

    def _bin_find_root(self, y, r, v, weights, w):
        return BinFind.apply(y, r, v, weights, w)


class MobiusFlow4ND(nn.Module):
    def __init__(self, D, K, condition=False, feature_dim=None):
        super(MobiusFlow4ND, self).__init__()

        self.D = D
        self.K = K

        self.condition = condition
        self.feature_dim = feature_dim

        self.register_buffer("_I", torch.eye(D), persistent=False)

    def compute_mcl_parameters(self, x, y, conditions):

        weights, w = torch.split(conditions, [self.K, 3 * self.K], dim=1)
        w = w.reshape(-1, self.K, self.D)
        proj = self._I[None, ...] - torch.einsum("ni,nj->nij", y, y)
        w = torch.einsum("nij,nkj->nki", proj, w)
        r = -x
        r = r / r.norm(dim=-1, keepdim=True)
        v = torch.cross(y, r, dim=-1)
        v = v / v.norm(dim=-1, keepdim=True)

        weights = torch.nn.functional.softplus(weights)
        sum_weight = weights.sum(dim=-1, keepdim=True)
        weights = weights / sum_weight
        w = 0.7 / (1 + torch.norm(w, dim=-1, keepdim=True)) * w
        return x, y, r, v, weights, w

    def compute_mcl_parameters_nd(self, x, y, conditions):
        """
        conditions: BS X Ndim X 4K
        """
        N, K, D = x.shape[1], self.K, self.D
        weights, w = torch.split(conditions, [K, 3 * K], dim=2)
        w = w.reshape(-1, N, K, D)
        proj = self._I[None, None, ...] - torch.einsum("nmi,nmj->nmij", y, y)
        w = torch.einsum("nmij,nmkj->nmki", proj, w)
        r = -x
        r = r / r.norm(dim=-1, keepdim=True)
        v = torch.cross(y, r, dim=-1)
        v = v / v.norm(dim=-1, keepdim=True)
        weights = torch.nn.functional.softplus(weights)
        sum_weight = weights.sum(dim=-1, keepdim=True)
        weights = weights / sum_weight
        w = 0.7 / (1 + torch.norm(w, dim=-1, keepdim=True)) * w
        return x, y, r, v, weights, w

    def forward_1d(self, rotation, permute=None, conditions=None):
        """
        rotations: BS X 3 X 3
        conditions: BS X 4K
        """
        assert permute != None, "The permuting function is needed in this module"
        if self.condition:
            assert conditions != None, "The input feature is needed in this module"

        x = rotation[..., permute[0]]
        y = rotation[..., permute[1]]

        x, y, r, v, weights, w = self.compute_mcl_parameters(x, y, conditions)

        tx, ldj = self._forward(x, r, v, weights, w)
        if (permute[1] - permute[0]) == 1 or (permute[1] - permute[0]) == -2:
            tz = torch.cross(tx, y, dim=-1)
        else:
            tz = torch.cross(y, tx, dim=-1)
        tz = tz / tz.norm(dim=-1, keepdim=True)
        trotation = torch.empty(rotation.size()).to(rotation.device)
        trotation[..., permute[0]] = tx
        trotation[..., permute[1]] = y
        trotation[..., permute[2]] = tz

        return trotation, ldj

    def forward(self, rotation, permute=None, conditions=None):
        """
        rotations: BS X Ndim X 3 X 3
        conditions: BS X Ndim X 4K
        """
        assert (
            conditions != None
        ), "The conditions are needed in this module to compute the mobius parameters"
        assert permute != None, "The permuting function is needed in this module"
        if self.condition:
            assert conditions != None, "The input feature is needed in this module"

        x = rotation[..., permute[0]]
        y = rotation[..., permute[1]]

        x, y, r, v, weights, w = self.compute_mcl_parameters_nd(x, y, conditions)

        tx, ldj = self._forward_nd(x, r, v, weights, w)
        if (permute[1] - permute[0]) == 1 or (permute[1] - permute[0]) == -2:
            tz = torch.cross(tx, y, dim=-1)
        else:
            tz = torch.cross(y, tx, dim=-1)
        tz = tz / tz.norm(dim=-1, keepdim=True)
        trotation = torch.empty(rotation.size()).to(rotation.device)
        trotation[..., permute[0]] = tx  # f(g(x_1),x_2)
        trotation[..., permute[1]] = y  # x_1
        trotation[..., permute[2]] = tz  # 3rd col after Gram-Schmidt

        return trotation, ldj.sum(dim=-1)

    def _h(self, z, w):
        return _h(z, w, self.D)

    def _h_nd(self, z, w):
        return _h_nd(z, w, self.D)

    def _forward(self, x, r, v, weights, w):
        z = x

        h_z = self._h(z, w)
        radians = torch.atan2(
            torch.einsum("nki,ni->nk", h_z, v), torch.einsum("nki,ni->nk", h_z, r)
        )
        tx = radians
        tx = torch.where(tx >= 0, tx, tx + torch.pi * 2)
        tx = torch.sum(weights * tx, dim=1, keepdim=True)

        tx = r * torch.cos(tx) + v * torch.sin(tx)

        z_w = z[:, None, :] - w
        z_w_norm = torch.norm(z_w, dim=-1)
        z_w_unit = z_w / z_w_norm[..., None]

        theta = torch.atan2(
            torch.einsum("ni,ni->n", x, v), torch.einsum("ni,ni->n", x, r)
        ).reshape(-1, 1)
        dz_dtheta = -torch.sin(theta) * r + torch.cos(theta) * v

        # n x k x 2 x 2
        dh_dz = (
            (1 - torch.norm(w, dim=-1) ** 2)[..., None, None]
            * (
                self._I[None, None, ...]
                - 2 * torch.einsum("nki,nkj->nkij", z_w_unit, z_w_unit)
            )
            / (z_w_norm[..., None, None] ** 2)
        )

        dh_dtheta = torch.einsum("nkpq,nq->nkp", dh_dz, dz_dtheta)
        dtx = torch.sum(torch.norm(dh_dtheta, dim=-1) * weights, dim=1)
        return tx, torch.log(dtx)

    def _forward_nd(self, x, r, v, weights, w):
        N = x.shape[1]
        z = x

        h_z = self._h_nd(z, w)
        radians = torch.atan2(
            torch.einsum("nmki,nmi->nmk", h_z, v), torch.einsum("nmki,nmi->nmk", h_z, r)
        )

        tx = radians
        tx = torch.where(tx >= 0, tx, tx + torch.pi * 2)
        tx = torch.sum(weights * tx, dim=2, keepdim=True)

        tx = r * torch.cos(tx) + v * torch.sin(tx)

        z_w = z[:, :, None, :] - w
        z_w_norm = torch.norm(z_w, dim=-1)
        z_w_unit = z_w / z_w_norm[..., None]

        theta = torch.atan2(
            torch.einsum("nmi,nmi->nm", x, v), torch.einsum("nmi,nmi->nm", x, r)
        ).reshape(-1, N, 1)
        dz_dtheta = -torch.sin(theta) * r + torch.cos(theta) * v

        # n x m x k x 2 x 2
        dh_dz = (
            (1 - torch.norm(w, dim=-1) ** 2)[..., None, None]
            * (
                self._I[None, None, None, ...]
                - 2 * torch.einsum("nmki,nmkj->nmkij", z_w_unit, z_w_unit)
            )
            / (z_w_norm[..., None, None] ** 2)
        )

        dh_dtheta = torch.einsum("nmkpq,nmq->nmkp", dh_dz, dz_dtheta)
        dtx = torch.sum(torch.norm(dh_dtheta, dim=-1) * weights, dim=2)
        return tx, torch.log(dtx)

    def inverse_1d(self, trotation, permute=None, conditions=None):
        assert permute != None, "The permuting function is needed in this module"
        if self.condition:
            assert conditions != None, "conditions input is needed in this module"

        tx = trotation[..., permute[0]]
        ty = trotation[..., permute[1]]

        tx, ty, r, v, weights, w = self.compute_mcl_parameters(tx, ty, conditions)

        ttheta = torch.atan2(
            torch.einsum("ni,ni->n", tx, v), torch.einsum("ni,ni->n", tx, r)
        ).reshape(-1, 1)
        ttheta = torch.where(ttheta >= 0, ttheta, ttheta + torch.pi * 2)

        ttheta = torch.where(
            abs(ttheta - torch.pi * 2) < 1e-4,
            torch.zeros(ttheta.size(), dtype=ttheta.dtype, device=ttheta.device),
            ttheta,
        )

        theta = self._bin_find_root(ttheta, r, v, weights, w)
        x = r * torch.cos(theta) + v * torch.sin(theta)
        _, ldj = self._forward(x, r, v, weights, w)

        if (permute[1] - permute[0]) == 1 or (permute[1] - permute[0]) == -2:
            z = torch.cross(x, ty, dim=-1)
        else:
            z = torch.cross(ty, x, dim=-1)
        z = z / z.norm(dim=-1, keepdim=True)
        rotation = torch.empty(trotation.size()).to(trotation.device)
        rotation[..., permute[0]] = x
        rotation[..., permute[1]] = ty
        rotation[..., permute[2]] = z

        return rotation, -ldj

    def inverse(self, trotation, permute=None, conditions=None):
        N = trotation.shape[1]

        assert permute != None, "The permuting function is needed in this module"
        if self.condition:
            assert conditions != None, "conditions input is needed in this module"

        tx = trotation[..., permute[0]]
        ty = trotation[..., permute[1]]

        tx, ty, r, v, weights, w = self.compute_mcl_parameters_nd(tx, ty, conditions)

        ttheta = torch.atan2(
            torch.einsum("nmi,nmi->nm", tx, v), torch.einsum("nmi,nmi->nm", tx, r)
        ).reshape(-1, N, 1)

        ttheta = torch.where(ttheta >= 0, ttheta, ttheta + torch.pi * 2)

        ttheta = torch.where(
            abs(ttheta - torch.pi * 2) < 1e-4,
            torch.zeros(ttheta.size(), dtype=ttheta.dtype, device=ttheta.device),
            ttheta,
        )

        theta = torch.zeros_like(ttheta)
        for i in range(N):
            theta_i = self._bin_find_root(
                ttheta[:, i], r[:, i], v[:, i], weights[:, i], w[:, i]
            )
            theta[:, i] = theta_i

        x = r * torch.cos(theta) + v * torch.sin(theta)
        _, ldj = self._forward_nd(x, r, v, weights, w)

        if (permute[1] - permute[0]) == 1 or (permute[1] - permute[0]) == -2:
            z = torch.cross(x, ty, dim=-1)
        else:
            z = torch.cross(ty, x, dim=-1)
        z = z / z.norm(dim=-1, keepdim=True)
        rotation = torch.empty(trotation.size()).to(trotation.device)
        rotation[..., permute[0]] = x
        rotation[..., permute[1]] = ty
        rotation[..., permute[2]] = z

        return rotation, -ldj.sum(dim=-1)

    def _bin_find_root(self, y, r, v, weights, w):
        return BinFind.apply(y, r, v, weights, w)


class BinFind(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y, r, v, weights, w):
        ctx.r = r.clone().detach()
        ctx.v = v.clone().detach()
        ctx.weights = weights.clone().detach()
        ctx.w = w.clone().detach()
        a = torch.ones(y.size(), device=y.device, dtype=y.dtype) * torch.pi / 2
        b = torch.ones(y.size(), device=y.device, dtype=y.dtype) * 3 / 2 * torch.pi
        time = 1
        while abs(torch.max(b - a)) >= 1e-4:
            x0 = (a + b) / 2
            fx0 = BinFind._forward_theta(x0, r, v, weights, w) - y

            if time > 100:
                print("fail")
                break

            bigger = fx0 < 0
            lesser = fx0 >= 0
            a = a + (b - a) / 2 * bigger
            b = b - (b - a) / 2 * lesser

            time += 1
        ctx.x = x0.clone().detach()
        ctx.y = y.clone().detach()
        return x0

    @staticmethod
    def _h(z, w, D=3):
        return _h(z, w, D)

    @staticmethod
    def _forward_theta(x, r, v, weights, w):
        """input: theta, return theta' and partial theta'/ partial theta
        used to compute inverse"""
        z = r * torch.cos(x) + v * torch.sin(x)

        h_z = BinFind._h(z, w)
        radians = torch.atan2(
            torch.einsum("nki,ni->nk", h_z, v), torch.einsum("nki,ni->nk", h_z, r)
        )
        tx = radians
        tx = torch.where(tx >= 0, tx, tx + torch.pi * 2)
        tx = torch.sum(weights * tx, dim=1, keepdim=True)

        return tx

    @staticmethod
    def backward(ctx, x_grad):
        x = ctx.x
        y = ctx.y
        r = ctx.r
        v = ctx.v
        weights = ctx.weights
        w = ctx.w
        with torch.set_grad_enabled(True):
            x.requires_grad_(True)
            r.requires_grad_(True)
            v.requires_grad_(True)
            weights.requires_grad_(True)
            w.requires_grad_(True)
            x_grad_2, r_grad, v_grad, weights_grad, w_grad = torch.autograd.grad(
                BinFind._forward_theta(x, r, v, weights, w),
                (x, r, v, weights, w),
                torch.ones_like(x_grad),
            )
            y_grad = (
                torch.where(x_grad_2 != 0, 1 / x_grad_2, torch.zeros_like(x_grad_2))
                * x_grad
            )
            r_grad = (
                torch.where(x_grad_2 != 0, -r_grad / x_grad_2, torch.zeros_like(r_grad))
                * x_grad
            )
            v_grad = (
                torch.where(x_grad_2 != 0, -v_grad / x_grad_2, torch.zeros_like(v_grad))
                * x_grad
            )
            w_grad = torch.where(
                x_grad_2.unsqueeze(-1) != 0,
                -w_grad / x_grad_2.unsqueeze(-1),
                torch.zeros_like(w_grad),
            ) * x_grad.unsqueeze(-1)
            weights_grad = (
                torch.where(
                    x_grad_2 != 0,
                    -weights_grad / x_grad_2,
                    torch.zeros_like(weights_grad),
                )
                * x_grad
            )
        return y_grad, r_grad, v_grad, weights_grad, w_grad
