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


class ClipScore(Metric):
    r"""
    Calculates CLIP-S which is used to assess the alignment between the 
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
        super().__init__(**kwargs)

        self.limit = limit
        self._debug = True
        self._dtype = torch.float64

        for k in ['x', 'y', 'x0']:  # x: real, y: text, x0: fake
            self.add_state(f"{k}_feat", [], dist_reduce_fx=None)

    def update(self, x: Tensor, y: Tensor, x0: Tensor) -> None:
        r"""
        Update the state with extracted features in double precision. This 
        method changes the precision of features into double-precision before 
        saving the features. 

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

    def _modify(self, mode: Any = None):
        r"""
        Modify the distribution of generated images for ablation study.

        Arg:
            mode (str): if `mode` is "real", it measure the real's score, if
                `mode` is "shuffle", deliberately break the alignmnet with 
                the condition by randomly-shuffling their counterparts.
        """
        if "real" == mode:
            self.x0_feat = self.x_feat  # real text instead of fake text
        elif "shuffle" == mode:
            random.shuffle(self.x0_feat)
        return self

    def compute(self, reduction: bool = True, mode=None) -> Tensor:
        r"""
        Calculate the CLIP-S score based on accumulated extracted features.
        """
        feats = [torch.cat(getattr(self, f"{k}_feat"), dim=0)
                 for k in ['x', 'y', 'x0']]

        return self._compute(*feats, reduction).to(self.orig_dtype)

    def _compute(self, X: Tensor, Y: Tensor, Z: Tensor, reduction):
        def dot(x, y):
            return (x * y).sum(dim=-1)

        excess = X.shape[0] - self.limit
        if 0 < excess:
            X, Y, Z = [x[:-excess] for x in [X, Y, Z]]

        scores = dot(Z, Y)
        weight = 2.5

        if reduction:
            return weight * scores.float().mean()
        else:
            return weight * scores.float()
