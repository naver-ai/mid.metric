"""
    mid.metric 
    Copyright (c) 2022-present NAVER Corp.
    Apache-2.0
"""
from PIL import Image
from statsmodels.stats.inter_rater import fleiss_kappa, aggregate_raters
from tqdm import tqdm
from typing import *
import clip
import csv
import krippendorff as kd
import os
import pickle
import scipy.stats as ss

from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader
from torchmetrics.metric import Metric
import torch
import torch.nn.functional as F
import torchvision.transforms as T

from coco import *
from metrics import *


def escape(x):
    return x.replace('-', '_').replace('/', '_')


def get_clip(eval_model: Module, device: Union[torch.device, int]) \
        -> Tuple[Module, Module]:
    """Get the CLIP model

    Args:
        eval_model (Module): The CLIP model to evaluate
        device (Union[torch.device, int]): Device index to select

    Returns:
        Tuple[Module, Module]: The CLIP model and a preprocessor
    """
    clip_model, _ = clip.load(eval_model)
    clip_model = clip_model.cuda(device)
    clip_prep = T.Compose([T.Resize(224),
                           T.Normalize((0.48145466, 0.4578275, 0.40821073),
                                       (0.26862954, 0.26130258, 0.27577711))])
    return clip_model, clip_prep


def init_metric(root: str, metric: Type[Metric], eval_model: Module,
                limit: int, device: torch.device) -> Metric:
    """Initialize a given metric class.

    Args:
        root (str): Path to data directory
        metric (Type[Metric]): Metric class
        eval_model (Module): Evaluating CLIP model
        limit (int, optional): Number of reference samples
        device (torch.device): Device index to select

    Returns:
        Metric: Initialized metric instance
    """
    if metric is SemanticObjectAccuracy:
        m = metric(limit=limit)
    elif metric is CaptionClipScore:
        m = metric(limit=limit, gen_json=os.path.join(root, "ofa_caption"))
    else:
        m = metric(768 if eval_model == 'ViT-L/14' else 512,
                   limit=limit)
    m.cuda(device)
    m._debug = False
    return m


@torch.no_grad()
def populate_metrics(dataloader: DataLoader, metrics: List[Metric],
                     clip_model: Module) -> Tensor:
    """Populate the list of metrics using a given data loader.

    Args:
        dataloader (DataLoader): Data loader
        metrics (List[Metric]): List of metrics
        clip_model (Module): Evaluating CLIP model

    Returns:
        Tensor: Labels
    """
    device = next(clip_model.parameters()).device
    labels = []
    for i, (real, gt, iid, cid, fake, label, gen_type) in enumerate(
            tqdm(dataloader)):

        real = real.cuda(device)
        fake = fake.cuda(device)
        labels.append(torch.stack(label, dim=1))

        txt = clip.tokenize(gt, truncate=True).cuda(device)
        txt_features = clip_model.encode_text(txt).float()

        real_im_features = clip_model.encode_image(
            clip_prep(real)).float()
        fake_im_features = clip_model.encode_image(
            clip_prep(fake)).float()

        # float16 of CLIP may suffer in l2-normalization
        txt_features = F.normalize(txt_features, dim=-1)
        real_im_features = F.normalize(real_im_features, dim=-1)
        fake_im_features = F.normalize(fake_im_features, dim=-1)

        X_ref = real_im_features
        Y_ref = txt_features
        X = fake_im_features

        # metrics handle features in float64
        for idx, m in enumerate(metrics):
            if isinstance(m, SemanticObjectAccuracy):
                m.update(real, gt, is_real=True)
                m.update(fake, gt, is_real=False)
            elif isinstance(m, CaptionClipScore):
                captions = m.get_captions(iid.tolist(), gen_type)
                cap = clip.tokenize(captions, truncate=True).cuda(device)
                cap_features = clip_model.encode_text(cap).float()
                cap_features = F.normalize(cap_features, dim=-1)
                m.update(X_ref, Y_ref, cap_features)
            else:
                m.update(X_ref, Y_ref, X)

        if (i + 1) * real.shape[0] > metrics[0].limit:
            print(f"break loop due to the limit of {metrics[0].limit}.")
            break

    return torch.cat(labels, dim=0).to(device)  # N x (quality, alignment)


if "__main__" == __name__:
    # config
    # _ = torch.manual_seed(123)
    eval_model = os.getenv('EVAL_MODEL')
    if eval_model is None:
        eval_model = "ViT-B/32"
    root = "./data/"
    info_path = root + "sample_info.pkl"
    amt_path = root + "amt/amt_result_tot.pkl"
    worker_path = root + "worker_info_tot.pkl"
    fake_path = root + "fakeim/"
    limit = 30000  # number of reference samples

    METRICS = [MutualInformationDivergence,  # Ours
               ClipScore,                    # CLIP-S
               RPrecision,                   # CLIP-R-Precision
               SemanticObjectAccuracy,       # Piece-wise SOA
               InfoNCE,                      # Negative InfoNCE loss
               CaptionClipScore,             # OFA-Large+CLIP-S
               ]

    cache_path = os.path.join(
        root, ".cache",
        f"likert_amt_{escape(eval_model)}_metric{len(METRICS)}.pth")
    os.makedirs(os.path.join(root, ".cache"), exist_ok=True)

    force = False
    if not os.path.exists(cache_path) or force:
        # get clip model
        device = torch.device("cuda:0")
        print(eval_model)
        clip_model, clip_prep = get_clip(eval_model, device)

        metrics = [
            init_metric(root, x, eval_model, limit, device) for x in METRICS]

        # load dataset
        ds = GeneratedCocoDataset(info_path=info_path, gen_path=fake_path,
                                  amt_path=amt_path)
        dl = DataLoader(ds, batch_size=60,
                        drop_last=False, shuffle=False,
                        num_workers=8)

        # compute clip features
        label = populate_metrics(dl, metrics, clip_model)
        results = [m.compute(reduction=False) for m in metrics]

        torch.save([label, results], cache_path)
        print(f"[INFO] score cache is saved to `{cache_path}`.")
    else:
        label, results = torch.load(cache_path)
        print(f"[INFO] score cache is loaded from `{cache_path}`.")

    label = label[:limit, 1]  # select the text-image alignment judgments
    label = label.cpu()
    mask = label > 0  # the mask of valid 2k judgments
    label = label[mask]  # select valid 2k judgments
    results = [x[mask] for x in results]  # select 2k evaluating samples

    for x in results:
        assert x.shape[0] == 2000

    print(f"[INFO] {mask.sum()} samples have amt judgments.\n")

    for variant in ['c', 'b']:
        print(f"Kendall tau {variant} correlation")
        tau = [ss.kendalltau(
            ss.rankdata(x[:limit].cpu().tolist()),
            ss.rankdata(label.cpu().tolist()), variant=variant)
            for x in results]
        print("MID, CLIP-S, CLIP-R-Precision, SOA, InfoNCE, Caption")
        print(", ".join([f"{x.correlation:.3f}" for x in tau]))

        print("\n\tp-values:")
        print("\t" + ", ".join([f"{x.pvalue:.5f}" for x in tau]))

        soa = results[3][:limit]
        mask = soa >= 0
        rate = mask.sum() / label.shape[0]
        tau = ss.kendalltau(ss.rankdata(soa[mask].cpu().tolist()),
                            ss.rankdata(label[mask].cpu().tolist()),
                            variant=variant)
        print(f"\n\tKendall tau {variant} for valid SOA samples: ", end="")
        print(f"{tau.correlation:.3f}")
        print(f"\tThe # of valid samples: {mask.sum()} ({rate})\n")
