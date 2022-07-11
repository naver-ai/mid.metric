"""
    mid.metric 
    Copyright (c) 2022-present NAVER Corp.
    Apache-2.0
"""
from typing import *
import random
import json
import os

import torch
from torch import Tensor

from torchmetrics.metric import Metric
from torchmetrics.utilities import rank_zero_info

from .clips import ClipScore


class CaptionClipScore(ClipScore):
    r"""
    Calculates Captioning+CLIP-S which is used to assess the alignment between 
    the conditional texts and the generated texts from the generated images.

    Args:
        feature (int): the number of features
        limit (int): limit the number of samples to calculate
        gen_json (str): path to the generated captions by the OFA-Large
        kwargs:
            Additional keyword arguments, see :ref:`Metric kwargs` for more info.
    """
    higher_is_better = True

    def __init__(self, feature: int = 512, limit: int = 30000,
                 gen_json: str = "data/ofa_caption/",
                 **kwargs: Dict[str, Any]) -> None:
        super().__init__(feature, limit, **kwargs)
        self.generated_captions = {}
        for g in ['lafite_gtcap', 'vqdiffusion_gtcap']:
            path = os.path.join(gen_json, f"{g}_predict.json")
            if os.path.exists(path):
                if g not in self.generated_captions:
                    self.generated_captions[g] = {}
                for d in json.load(open(path)):
                    self.generated_captions[g][int(d["image_id"])] = \
                        d["caption"]
            else:
                print(f"warning: {path} not found!")

    def get_captions(self, caption_ids: List[int], gan: List[str] = None) \
            -> List[str]:
        r"""
        Return the generated caption by the lists of caption ids and generative 
        model names as keys.

        Args:
            caption_ids (List[int]): List of caption ids
            gan (List[str]): List of generative model names

        Returns:
            List[str]: List of generated captions
        """
        outputs = []
        for x, y in zip(caption_ids, gan):
            if y in self.generated_captions:
                outputs.append(self.generated_captions[y][int(x)])
            else:
                outputs.append(f"Unavailable! ({y})")
                print(f"warning: unavailable for {y} {x}")
        return outputs
