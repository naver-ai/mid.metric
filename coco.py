"""
    mid.metric 
    Copyright (c) 2022-present NAVER Corp.
    Apache-2.0
"""
from typing import *

from PIL import Image
import copy
import json
import numpy as np
import os
import pickle
import random

from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import CocoCaptions
import torch


class GeneratedCocoDataset(Dataset):
    """Generated COCO Captions.

    Attributes:
        text_dir (string): Path to caption dataset
        media_size (int): Shorter one between height and width
        media_dir (string): Path to image dataset
    """

    def __init__(self,
                 media_dir: str = "data/coco2014/",
                 text_dir: str = "data/coco2014/annotations/",
                 info_path: str = None,
                 gen_path: str = None,
                 amt_path: str = None,
                 split: str = 'val',
                 media_size: int = 256):
        """
        Args:
            media_dir (str, optional): Path containing image data
            text_dir (str, optional): Path containing caption data
            split (str, optional): Dataset split
            media_size (int, optional): Shorter one between height and width
            max_text_len (int, optional): Max length of text
        """
        super().__init__()
        self.media_dir = media_dir
        self.text_dir = text_dir
        self.split = split
        self.media_size = media_size

        self.info_path = info_path
        self.gen_path = gen_path
        self.amt_path = amt_path
        if amt_path is not None:
            self.load_amt_judgments(amt_path)
        self.load_sample_info(self.split)
        self.load_coco_captions(self.split)
        self.transforms = transforms.Compose([
            transforms.Resize(media_size),
            transforms.CenterCrop(media_size),
            transforms.ToTensor(),
        ])

    def load_amt_judgments(self, amt_path: str):
        self.amt_data = pickle.load(open(amt_path, "rb"))
        self.amt_image_ids = []
        for key in self.amt_data:
            image_id = int(key.split('_')[0])
            self.amt_image_ids.append(image_id)

    def load_sample_info(self, split: str):
        with open(self.info_path, 'rb') as f:
            d = pickle.load(f)
        data = [x for x in d if x['foil']]  # only foiled samples
        self.data = []
        self.image_ids = set()  # 32,150 for test
        self.amt_dup_count = 0
        for s in data:
            if s['image_id'] not in self.image_ids:
                self.image_ids.add(s['image_id'])
                cnt = 0
                for k, v in self.amt_data.items():
                    if k.startswith(f"{s['image_id']}_{s['id']}"):
                        s_ = copy.deepcopy(s)
                        s_['postfix'] = '_'.join(k.split('_')[2:])
                        self.data.insert(0, s_)
                        cnt += 1
                self.amt_dup_count += max(0, cnt - 1)
                if 0 == cnt:
                    self.data.append(s)
        # if self.amt_path is not None:
        #     print(f"#unique imageid for amt judgments: {self.amt_dup_count}")

    def load_coco_captions(self, split: str):
        path = os.path.join(self.text_dir, f"captions_{split}2014.json")
        d = json.load(open(path, 'r'))['annotations']
        self.captions = {}
        for x in d:
            key = x['image_id']
            if key in self.image_ids:
                self.captions[key] = self.captions.get(key, [])
                self.captions[key].append(x)
        # sanity check
        for k in self.captions:
            if 5 < len(self.captions[k]):
                # print(len(self.captions[k]))
                pass
            elif 5 > len(self.captions[k]):
                print(self.captions[k])

    def get_image(self, image_id: int):
        def _get_image(path: str, filename: str) -> torch.Tensor:
            img = Image.open(os.path.join(path, filename)).convert('RGB')
            return self.transforms(img)
        path = f"COCO_{self.split}2014_{image_id:012d}.jpg"
        return _get_image(
            os.path.join(self.media_dir, f"{self.split}2014/"), path)

    def __len__(self):
        """The nubmer of samples.

        Returns:
            long: The number of samples
        """
        return len(self.data)

    def __getitem__(self, idx):
        """Implemetation of the `__getitem__` magic method.

        Args:
            idx (int): The index of samples

        Returns:
            Tuple[torch.Tensor, str, str, Dict]: A sample enveloped in dict
        """
        image_id = self.data[idx]['image_id']
        caption_id = self.data[idx]['id']
        feat = self.get_image(image_id)
        caption = None
        for c in self.captions[image_id]:
            if self.data[idx]['id'] == c['id']:
                caption = c['caption']
        assert caption is not None

        if image_id in self.amt_image_ids:
            postfix = self.data[idx]['postfix']
            gan = postfix.split('.')[0].split('_')[0] + "_gtcap"
            key = f"{image_id}_{caption_id}_{postfix}"
            alignment = np.median(self.amt_data[key]["alignment"])
            quality = np.median(self.amt_data[key]["quality"])
            label = [quality, alignment]
            gen_type = key.split('_')[2]
            fake = self.transforms(Image.open(os.path.join(
                self.gen_path, f"{image_id}_{caption_id}_{gen_type}.png")
            ).convert("RGB"))
        else:
            # Real samples are only used for the references.
            # Fake images and label is not used by masking out.
            label = [-1, -1]  # not collected
            gan = random.choice(["lafite_gtcap", "vqdiffusion_gtcap"])
            fake = torch.zeros(3, self.media_size, self.media_size)
        return feat, caption, image_id, caption_id, fake, label, gan
