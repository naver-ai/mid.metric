"""
    mid.metric 
    Copyright (c) 2022-present NAVER Corp.
    Apache-2.0
"""
from typing import *
import random

import torch
from torch import Tensor

from torchmetrics.metric import Metric
from torchmetrics.utilities import rank_zero_info


def log_det(X):
    eigenvalues = X.svd()[1]
    return eigenvalues.log().sum()


def robust_inv(x, eps=0):
    Id = torch.eye(x.shape[0]).to(x.device)
    return (x + eps * Id).inverse()


def exp_smd(a, b, reduction=True):
    a_inv = robust_inv(a)
    if reduction:
        assert b.shape[0] == b.shape[1]
        return (a_inv @ b).trace()
    else:
        return (b @ a_inv @ b.t()).diag()


def _compute_pmi(x: Tensor, y: Tensor, x0: Tensor, limit: int = 30000,
                 reduction: bool = True, full: bool = False) -> Tensor:
    r"""
    A numerical stable version of the MID score.

    Args:
        x (Tensor): features for real samples
        y (Tensor): features for text samples
        x0 (Tensor): features for fake samples
        limit (int): limit the number of samples
        reduction (bool): returns the expectation of PMI if true else sample-wise results
        full (bool): use full samples from real images

    Returns:
        Scalar value of the mutual information divergence between the sets.
    """
    N = x.shape[0]
    excess = N - limit
    if 0 < excess:
        if not full:
            x = x[:-excess]
            y = y[:-excess]
        x0 = x0[:-excess]
    N = x.shape[0]
    M = x0.shape[0]

    assert N >= x.shape[1], "not full rank for matrix inversion!"
    if x.shape[0] < 30000:
        rank_zero_info("if it underperforms, please consider to use "
                       "the epsilon of 5e-4 or something else.")

    z = torch.cat([x, y], dim=-1)
    z0 = torch.cat([x0, y[:x0.shape[0]]], dim=-1)
    x_mean = x.mean(dim=0, keepdim=True)
    y_mean = y.mean(dim=0, keepdim=True)
    z_mean = torch.cat([x_mean, y_mean], dim=-1)
    x0_mean = x0.mean(dim=0, keepdim=True)
    z0_mean = z0.mean(dim=0, keepdim=True)

    X = (x - x_mean).t() @ (x - x_mean) / (N - 1)
    Y = (y - y_mean).t() @ (y - y_mean) / (N - 1)
    Z = (z - z_mean).t() @ (z - z_mean) / (N - 1)
    X0 = (x0 - x_mean).t() @ (x0 - x_mean) / (M - 1)  # use the reference mean
    Z0 = (z0 - z_mean).t() @ (z0 - z_mean) / (M - 1)  # use the reference mean

    alternative_comp = False
    # notice that it may have numerical unstability. we don't use this.
    if alternative_comp:
        def factorized_cov(x, m):
            N = x.shape[0]
            return (x.t() @ x - N * m.t() @ m) / (N - 1)
        X0 = factorized_cov(x0, x_mean)
        Z0 = factorized_cov(z0, z_mean)

    # assert double precision
    for _ in [X, Y, Z, X0, Z0]:
        assert _.dtype == torch.float64

    # Expectation of PMI
    mi = (log_det(X) + log_det(Y) - log_det(Z)) / 2
    rank_zero_info(f"MI of real images: {mi:.4f}")

    # Squared Mahalanobis Distance terms
    if reduction:
        smd = (exp_smd(X, X0) + exp_smd(Y, Y) - exp_smd(Z, Z0)) / 2
    else:
        smd = (exp_smd(X, x0 - x_mean, False) + exp_smd(Y, y - y_mean, False)
               - exp_smd(Z, z0 - z_mean, False)) / 2
        mi = mi.unsqueeze(0)  # for broadcasting

    return mi + smd


class MutualInformationDivergence(Metric):
    r"""
    Calculates the Mutual Information Divergence (MID) which is used to assess 
    the text-image alignment between the conditional texts and the 
    generated images compared with the same texts and the real images as 
    follows:

    .. math::
        \mathbb{E}_{\hat{x}, \hat{y}} \text{PMI}(\hat{x}; \hat{y}) = I(\mathbf{X}; \mathbf{Y}) + 
            \frac{1}{2} \mathbb{E}_{\hat{x}, \hat{y}} \big[ D_M^2(\hat{x}) + D_M^2(\hat{y}) - D_M^2(\hat{z}) \big]
    where

    .. math::
        I(\mathbf{X}; \mathbf{Y}) = \frac{1}{2}\log\Big( \frac{\det(\Sigma_x) \det(\Sigma_y)}{\det(\Sigma_z)} \Big), 
        D_M^2(x) = (x - \mu_x)^\intercal \Sigma_x^{-1} (x - \mu_x).

    The two multivariate normal distributions are estimated from the 
    CLIP (Radford et al., 2021) features calculated on conditional texts and 
    generated images. The joint distribution :math:`\mathcal{N}(\mu, \Sigma)` 
    is from the concatenation of the two features.

    Args:
        feature (int): the number of features
        limit (int): limit the number of samples to calculate
        kwargs:
            Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example:
        >>> import torch
        >>> _ = torch.manual_seed(123)
        >>> from metrics.mid import MutualInformationDivergence
        >>> mid = MutualInformationDivergence(2)
        >>> dist1 = torch.rand(10, 2, dtype=torch.float64) * 2 - 1
        >>> dist2 = torch.rand(10, 2, dtype=torch.float64) * 2 - 1
        >>> dist3 = dist1 * 1.1
        >>> mid.update(dist1, dist2, dist3)
        >>> mid.compute()
        MI of real images: 0.1543
        tensor(0.1516, dtype=torch.float64)

    """
    higher_is_better: bool = True
    full_state_update: bool = False

    def __init__(self, feature: int = 512, limit: int = 30000,
                 **kwargs: Dict[str, Any]) -> None:
        super().__init__(**kwargs)

        self.limit = limit
        self._debug = False
        self._dtype = torch.float64

        for k in ['x', 'y', 'x0']:  # x: real, y: text, x0: fake
            self.add_state(f"{k}_feat", [], dist_reduce_fx=None)

    def update(self, x: Tensor, y: Tensor, x0: Tensor) -> None:
        r"""
        Update the state with extracted features in double precision. It is 
        recommended to use the CLIP ViT-B/32 or ViT-L/14 features which is 
        L2-normalized. This method changes the precision of features into 
        double-precision before saving the features.

        Args:
            x (Tensor): tensor with the extracted real image features
            y (Tensor): tensor with the extracted text features
            x0 (Tensor): tensor with the extracted fake image features
        """
        assert x.shape[0] == y.shape[0] and x.shape[1] == y.shape[1]

        self.orig_dtype = x.dtype
        x, y, x0 = [x.double() for x in [x, y, x0]]
        self.x_feat.append(x)
        self.y_feat.append(y)
        self.x0_feat.append(x0)

    def _modify(self, mode: str = None):
        r"""
        Modify the distribution of generated images for ablation study.

        Arg:
            mode (str): if `mode` is "real", it measure the real's score, if
                `mode` is "shuffle", deliberately break the alignmnet with 
                the condition by randomly-shuffling their counterparts.
        """
        if "real" == mode:
            self.x0_feat = self.x_feat
        elif "shuffle" == mode:
            random.shuffle(self.x0_feat)
        return self

    def compute(self, reduction: bool = True, mode=None) -> Tensor:
        r"""
        Calculate the MID score based on accumulated extracted features.
        """
        feats = [torch.cat(getattr(self, f"{k}_feat"), dim=0)
                 for k in ['x', 'y', 'x0']]

        return _compute_pmi(*feats, self.limit, reduction).to(self.orig_dtype)


if "__main__" == __name__:
    import torch
    import torch.nn.functional as F
    _ = torch.manual_seed(123)
    mid = MutualInformationDivergence(2)
    dist1 = torch.rand(10, 2, dtype=torch.float64) * 2 - 1
    dist2 = torch.rand(10, 2, dtype=torch.float64) * 2 - 1
    dist3 = dist1 * 1.1
    mid.update(dist1, dist2, dist3)
    print(mid.compute())
