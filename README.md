An official code of "Mutual Information Divergence: A Unified Metric for Multimodal Generative Models."

## Installation

```bash
pip install -r requirements.txt
``` 

## For SOA, install the darknet (YOLO-V3)

```bash
git clone https://github.com/pjreddie/darknet.git
cd darknet
make
wget https://pjreddie.com/media/files/yolov3.weights
```

## Data preparation

We need `./data` directory containing the following directories and files to run. 

```
data/.cache/likert_amt_{ViT_B_32|ViT_L_14}_metric6.pth  # cached results
data/amt/amt_result_tot.pkl
data/coco2014/annotations  # not included in data.tar
data/coco2014/val2014  # not included in data.tar
data/fakeim/{image_id}_{caption_id}_{vqdiffusion|lafite}.png
data/ofa_caption/{vqdiffusion|lafite}_gtcap_predict.json
data/sample_info.pkl
```

You can access the data via [Google Drive](https://drive.google.com/file/d/1xsoRzyt-1DIBGfhoPvD4XD7LQg2-mkk9/view?usp=sharing) (243.4MB) and the MD5 (128-bit) checksum is `3323f51ca62788989bc331cc7f79c4e6`. You can verify via

```bash
echo "3323f51ca62788989bc331cc7f79c4e6 data.tar" | md5sum -c
````

For the downloading in the command-line, you can use [gdrive](https://github.com/prasmussen/gdrive) (:warning: notice for x64 binary: [#580](https://github.com/prasmussen/gdrive/issues/580#issuecomment-864729091)) as follows:

```bash
gdrive download 1xsoRzyt-1DIBGfhoPvD4XD7LQg2-mkk9
tar xvf data.tar
```

For the `data/coco2014`, we need to download [val2014](http://images.cocodataset.org/zips/val2014.zip) and [annotations](http://images.cocodataset.org/annotations/annotations_trainval2014.zip) in the [COCO dataset](https://cocodataset.org/#download).

## Run the evaluation protocol using the AMT judgments

For the CLIP with the ViT-B/32 backbone, run
```bash
EVAL_MODEL=ViT-B/32 python main.py
``` 

For the CLIP with the ViT-B/32 backbone, run
```bash
EVAL_MODEL=ViT-L/14 python main.py
```

## Citation

We humbly encourage you to consider citing our work if you use this work:

```
@inproceedings{kim2022mid,
    title = {{M}utual {I}nformation {D}ivergence: A Unified Metric for Multimodal Generative Models},
    author = {Kim, Jin-Hwa and Kim, Yunji and Lee, Jiyoung and Yoo, Kang Min and Lee, Sang-Woo},
    booktitle = {Advances in Neural Information Processing Systems 35},
    url = {https://openreview.net/forum?id=wKd2XtSRsjl}
    year = {2022}
}
```

## Software License

This repository is licensed under Apache License 2.0.

```
Copyright (c) 2022-present NAVER Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
