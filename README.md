# Fast End-to-end Video Quality Assessment

We support 

An Open Source Library for Fast **E**nd-to-**E**nd **V**ideo **Q**uality **A**ssessment Models

Supported Full Method:

- FAST-VQA (Train/Infer)
- DiViDe (Train/Infer, in development)

*InDevelopment Version: 1.0.1*

## Modularized Parts

Supported Data Pre-processing:

- Fragments (as proposed in our work [FAST-VQA](FAST_VQA_Paper.pdf))
- Resizing
- Cropping

And combinations of the data pre-processing ways.

Supported Backbone:

- Video Swin Transformer (with GRPB, as proposed in [FAST-VQA](FAST_VQA_Paper.pdf))
- Video Swin Transformer (vanilla)
- ConvNext-I3D (vanilla)

Supported Head:

- Non-linear Head (with post-pooling, as proposed in [FAST-VQA](FAST_VQA_Paper.pdf)))



The default train and test is for FAST-VQA. We will support configs for different methods when more end-to-end VQA methods come out.



## Installation

### Requirements

The original library is build with

- python=3.8.8
- torch=1.10.2
- torchvision=0.11.3

while using decord module to read original videos (so that you don't need to make any transform on your original .mp4 input).

To get all the requirements, please run

```shell
pip install -r requirements.txt
```

### Direct install

You can run

```shell
pip install .
```

or 

```shell
python setup.py installl
```

to install the full FAST-VQA with its requirements.

## Usage

### Visualize *fragments* and Quality Maps

If you would like to visualize the proposed *fragments*, you can generate the demo visualizations by yourself, via the following script:


```shell
python visualize.py -d $DATASET$ 
```

You can also visualize the patch-wise local quality maps rendered on fragments, via 

```shell
python visualize.py -d $DATASET$ -nm
```



### Inference on Scripts

You can install this directory by running

```shell
pip install .
```

Then you can embed these lines into your python scripts:

```python
from fastvqa import deep_end_to_end_vqa

dum_video = torch.randn((3,240,720,1280)) # A sample 720p, 240-frame video
vqa = deep_end_to_end_vqa(True, model_type=model_type)
score = vqa(dum_video)
print(score)
```

This script will automatically download the model weights pretrained from LSVQ.

### Benchmarking FAST-VQA

You can directly benchmark the model with mainstream benchmark VQA datasets.

```shell
python inference.py -d $DATASET$
```

Available datasets are LIVE_VQC, KoNViD, CVD2014, YouTubeUGC), LSVQ.



## Train FAST-VQA


### Train from Recognition Features

You might need to download the original [Swin-T Weights](https://github.com/SwinTransformer/storage/releases/download/v1.0.4/swin_tiny_patch244_window877_kinetics400_1k.pth) to initialize the model.


#### Intra Dataset Training

This training will split the dataset into 10 random train/test splits (with random seed 42) and report the best result on the random split of the test dataset. 

```shell
python train.py -d $DATASET$ --from_ar
```

Supported datasets are KoNViD-1k, LIVE_VQC, CVD2014, YouTube-UGC.

#### Cross Dataset Training

This training will do no split and directly report the best result on the provided validation dataset.

```shell
python inference.py -d $TRAINSET$-$VALSET$ --from_ar -lep 0 -ep 30
```

Supported TRAINSET is LSVQ, and VALSETS can be LSVQ(LSVQ-test+LSVQ-1080p), KoNViD, LIVE_VQC.


### Finetune with provided weights

#### Intra Dataset Training

This training will split the dataset into 10 random train/test splits (with random seed 42) and report the best result on the random split of the test dataset. 

```shell
python inference.py -d $DATASET$
```

Supported datasets are KoNViD-1k, LIVE_VQC, CVD2014, YouTube-UGC.

## Switching to FASTER-VQA

You can add the argument `-m FASTER` in any scripts (```finetune.py, inference.py, visualize.py```) above to switch to FAST-VQA-M instead of FAST-VQA.

## Citation

Please cite the following paper when using this repo.

```
@article{wu2022fastquality,
  title={FAST-VQA: Efficient End-to-end Video Quality Assessment with Fragment Sampling},
  author={Wu, Haoning and Chen, Chaofeng and Hou, Jingwen and Liang, Liao and Wang, Annan and Sun, Wenxiu and Yan, Qiong and Weisi, Lin},
  journal={arXiv preprint},
  year={2022}
}
```