"""
    mid.metric 
    Copyright (c) 2022-present NAVER Corp.
    Apache-2.0
"""
from typing import *
import random
import os
import pickle

import torch
from torch import Tensor

from torchmetrics.metric import Metric
from torchmetrics.utilities import rank_zero_info

from .darknet import *


class SemanticObjectAccuracy(Metric):
    r"""
    Calculates the Semantic Object Accuracy which is used to assess the 
    alignment between the conditional texts and the generated images. This
    metric is a little different from SOA-I and SOA-C since this is a piece-wise
    evaluating metric of SOA-I.

    Args:
        root (str): Path to darknet for the YOLO-V3
        img_size (int): Image size
        confidence (float): confidence for the YOLO-V3
        nms_thresh (float): NMS threshold for the YOLO-V3
        limit (int): Limit the number of samples to calculate
        kwargs:
            Additional keyword arguments, see :ref:`Metric kwargs` for more info.
    """
    higher_is_better: bool = True
    full_state_update: bool = False

    def __init__(self, root: str = "darknet",
                 img_size: int = 256, confidence: float = .5,
                 nms_thresh: float = .4, limit: int = 30000,
                 **kwargs: Dict[str, Any]) -> None:
        super().__init__(**kwargs)
        self._debug = True
        self.root = root
        self.img_size = img_size
        self.confidence = confidence
        self.nms_thresh = nms_thresh
        self.limit = limit
        self.setup()
        self.classes = load_classes(os.path.join(self.root, "data/coco.names"))

        self.add_state(f"reals", [], dist_reduce_fx=None)
        self.add_state(f"predictions", [], dist_reduce_fx=None)
        self.add_state(f"labels", [], dist_reduce_fx=None)

    def setup(self):
        # Set up the neural network
        print("Loading network ...")
        try:
            self.model = Darknet(os.path.join(self.root, "cfg/yolov3.cfg"))
            self.model.load_weights(os.path.join(self.root, "yolov3.weights"))
        except:
            print("Did you install darknet for YOLO-V3?")
            print("$ git clone https://github.com/pjreddie/darknet.git")
            print("$ cd darknet")
            print("$ make")
            print("$ wget https://pjreddie.com/media/files/yolov3.weights")
        print("Network successfully loaded")

        self.model.net_info["height"] = 256
        inp_dim = int(self.model.net_info["height"])
        assert inp_dim % 32 == 0
        assert inp_dim > 32

        # Set the model in evaluation mode
        self.model.eval()

    def get_labels(self, caption):
        # The rules from Table 4 (Hinz et al., 2020)
        labels = []
        tokens = caption.lower().split(' ')

        WORDS = {
            "person": ["person", "people", "human", "man", "men", "woman",
                       "women", "child", "children"],
            "diningtable": ["dining table", "table", "desk"],
            "cat": ["cat", "kitten"],
            "dog": ["dog", "pup"],
            "boat": ["boat", "ship"],
            "car": ["car", "auto"],
            "sports ball": ["ball"],
            "bicycle": ["bicycle", "bike"],
            "monitor": ["monitor", "tv", "screen"],
            "hot dog": ["hot dog", "chili dog", "cheese dog", "corn dog"],
            "fire hydrant": ["fire hydrant", "hydrant"],
            "sofa": ["sofa", "couch"],
            "aeroplane": ["plane", "jet", "aircraft"],
            "cell phone": ["cell phone", "mobile phone"],
            "refrigerator": ["refrigerator", "fridge"],
            "motocycle": ["motocycle", "dirt bike", "motobike", "scooter"],
            "backpack": ["backpack", "rucksack"],
            "handbag": ["handbag", "purse"],
            "mouse": ["computer mouse"],
            "scissor": ["scissors"],
            "orange": ["oranges"]
        }

        def remove_words(caption, excludes):
            for w in excludes:
                caption = caption.replace(w, "")
            return caption

        for c in self.classes:
            if c not in WORDS.keys():
                WORDS[c] = [c]

        # multiple words
        for label, words in WORDS.items():
            if "dog" == label:
                caption_ = remove_words(caption, [
                    "hot dog", "cheese dog", "chili dog", "corn dog"])
            elif "elephant" == label:
                caption_ = remove_words(caption, [
                    "toy elephant", "stuffed elephant"])
            elif "car" == label:
                caption_ = remove_words(caption, [
                    "train car", "car window", "side car", "passenger car",
                    "subway car", "car tire", "rail car", "tram car",
                    "street car", "trolly car"])
            elif "kite" == label:
                caption_ = remove_words(caption, ["kite board", "kiteboard"])
            elif "cake" == label:
                caption_ = remove_words(caption, ["cupcake"])
            elif "bicycle" == label:
                caption_ = remove_words(caption, [
                    "train car", "car window", "side car", "passenger car",
                    "subway car", "car tire", "rail car", "tram car",
                    "street car", "trolly car"])
            elif "tie" == label:
                caption_ = remove_words(caption, ["to tie"])
            elif "apple" == label:
                caption_ = remove_words(caption, ["pineapple"])
            elif "oven" == label:
                caption_ = remove_words(caption, ["microwave oven"])
            else:
                caption_ = caption

            for w in words:
                if 1 == len(w.split(' ')):
                    if w in caption_.lower().split(' '):
                        labels.append(label)
                else:
                    if w in caption_.lower():
                        labels.append(label)
        return labels

    def update(self, images: Tensor, captions: List[str],
               is_real: bool = False) -> None:
        r"""
        Update the state with images and captions.

        Args:
            images (Tensor): tensor with the extracted fake images
            captions (List[str]): List of captions
            is_real (bool): Is the real image?
        """
        with torch.no_grad():
            predictions = self.model(images)
            predictions = non_max_suppression(
                predictions, self.confidence, self.nms_thresh)

            for preds in predictions:
                img_preds_id = set()
                img_preds_name = set()  # handling multiple object
                img_bboxs = []
                if preds is not None and len(preds) > 0:
                    for pred in preds:
                        pred_id = int(pred[-1])
                        pred_name = self.classes[pred_id]

                        bbox_x = pred[0] / self.img_size
                        bbox_y = pred[1] / self.img_size
                        bbox_width = (pred[2] - pred[0]) / self.img_size
                        bbox_height = (pred[3] - pred[1]) / self.img_size

                        img_preds_id.add(pred_id)
                        img_preds_name.add(pred_name)
                        img_bboxs.append([bbox_x.cpu().numpy(),
                                          bbox_y.cpu().numpy(),
                                          bbox_width.cpu().numpy(),
                                          bbox_height.cpu().numpy()])
                if not is_real:
                    self.predictions.append(list(img_preds_name))
                else:
                    self.reals.append(img_preds_name)

        if not is_real:
            for caption in captions:
                self.labels.append(self.get_labels(caption))

            assert len(self.predictions) == len(self.labels)

    def _modify(self, mode: Any = None):
        r"""
        Modify the distribution of generated images for ablation study.

        Arg:
            mode (str): if `mode` is "real", it measure the real's score, if
                `mode` is "shuffle", deliberately break the alignmnet with 
                the condition by randomly-shuffling their counterparts.
        """
        if "real" == mode:
            self.predictions = self.reals
        elif "shuffle" == mode:
            random.shuffle(self.predictions)
        return self

    def compute(self, reduction: bool = True) -> Tensor:
        r"""
        Calculate the point-wise SOA score.
        """
        accuracy = []
        division_by_zero = 0
        for preds, labels in zip(self.predictions, self.labels):
            if 0 == len(labels):
                division_by_zero += 1
                accuracy.append(-1)
            else:
                accuracy.append(
                    sum([1. for x in set(preds) if x in labels]) / len(labels))
        accuracy = torch.Tensor(accuracy)
        if 0 < division_by_zero:
            print(f"warning: {division_by_zero} samples have no detection.")

        if reduction:
            return accuracy[:self.limit].mean()
        else:
            return accuracy[:self.limit]


if "__main__" == __name__:
    from PIL import Image
    import torchvision.transforms as T
    soa = SemanticObjectAccuracy()
    transforms = T.Compose([
        T.ToTensor(),
        T.Resize(256),
        T.CenterCrop(256)])
    img = transforms(
        Image.open("darknet/data/dog.jpg"))
    imgs = torch.stack([img, img], dim=0)
    soa.update(
        soa, ["the dog is there with motocycle and plane",
              "dog is beside a bicycle."])
    print(soa.compute(reduction=False))
