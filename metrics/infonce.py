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

from .r_precision import RPrecision


class InfoNCE(RPrecision):
    r"""
    Calculates InfoNCE which is used to assess the alignment between the 
    conditional texts and the generated images.

    Args:
        feature (int): the number of features
        limit (int): limit the number of samples to calculate
        kwargs:
            Additional keyword arguments, see :ref:`Metric kwargs` for more info.
    """
    higher_is_better: bool = True
    full_state_update: bool = False

    def __init__(self, feature: int = 512, limit: int = 30000,
                 **kwargs: Dict[str, Any]) -> None:
        super().__init__(feature, limit, **kwargs)

    def _compute(self, X: Tensor, Y: Tensor, Z: Tensor, reduction):
        def dot(x, y):
            return (x * y).sum(dim=-1)

        excess = X.shape[0] - self.limit
        if 0 < excess:
            X, Y, Z = [x[:-excess] for x in [X, Y, Z]]

        # scores = []
        # scores.append(dot(Z, Y))
        # for i in range(99):  # negative scores
        #     Y_ = Y[torch.randperm(Y.shape[0])]
        #     scores.append(dot(Z, Y_))
        # scores = torch.stack(scores, dim=-1)  # N x 100
        # prob = scores.softmax(dim=-1)[:,0]

        scores = Z @ Y.t()
        prob = scores.softmax(dim=-1).diag()

        if reduction:
            return prob.float().mean()
        else:
            return prob.float()
