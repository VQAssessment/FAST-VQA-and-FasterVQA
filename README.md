# :1st_place_medal:FAST-VQA

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/fast-vqa-efficient-end-to-end-video-quality/video-quality-assessment-on-konvid-1k)](https://paperswithcode.com/sota/video-quality-assessment-on-konvid-1k?p=fast-vqa-efficient-end-to-end-video-quality)
	
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/fast-vqa-efficient-end-to-end-video-quality/video-quality-assessment-on-live-fb-lsvq)](https://paperswithcode.com/sota/video-quality-assessment-on-live-fb-lsvq?p=fast-vqa-efficient-end-to-end-video-quality)
	
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/fast-vqa-efficient-end-to-end-video-quality/video-quality-assessment-on-live-vqc)](https://paperswithcode.com/sota/video-quality-assessment-on-live-vqc?p=fast-vqa-efficient-end-to-end-video-quality)
	
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/fast-vqa-efficient-end-to-end-video-quality/video-quality-assessment-on-youtube-ugc)](https://paperswithcode.com/sota/video-quality-assessment-on-youtube-ugc?p=fast-vqa-efficient-end-to-end-video-quality)

An Open Source Deep End-to-End Video Quality Assessment Toolbox,

开源的端到端视频质量评价工具箱，

& Reproducible Code for ECCV2022 Paper [FAST-VQA: Efficient End-to-end Video Quality Assessment with Fragment Sampling](https://arxiv.org/abs/2207.02595v1).

暨 可复现 ECCV2022 论文 [FAST-VQA: Efficient End-to-end Video Quality Assessment with Fragment Sampling](https://arxiv.org/abs/2207.02595v1) 的代码。



In this release, we have refactored the training and testing code. The refactored code can achieve the same performance as the original version and allow modification of (1) the backbone structures; (2) the sampling hyper-parameters; (3) loss functions.

在这一版本中，我们对训练和测试的代码进行了重构。重构后的代码可以达到与原始版本相同的性能，并允许修改网络结构/采样的超参数/损失函数。





## :triangular_flag_on_post: Modularized Parts Designed for Development
为开发设计的模块化架构

### Data Preprocessing
数据预处理

Please view [Data Processing](./fastvqa/datasets/fusion_datasets.py) to see the source codes for data processing.

#### Spatial Sampling
空间采样

We have supported spatial sampling approachs as follows:


- fragments
- resize
- arp_resize (resize while keeping the original Aspect Ratio)
- crop

We also support the combination of those sampling approaches (multi-branch networks) for more flexibility.

#### Temporal Sampling (New)
时域采样（新）

We also support different temporal sampling approaches:

- SampleFrames (sample continuous frames, imported from MMAction2)
- FragmentSampleFrames (:sparkles: New, sample fragment-like discontinuous frames)


### Network Structures
网络结构

#### Network Backbones
骨干网络

- Video Swin Transformer (with GRPB, as proposed in [FAST-VQA](https://arxiv.org/abs/2207.02595v1))
- Video Swin Transformer (vanilla)
- ConvNext-I3D (vanilla)

### Network Heads
网络头

- IP-NLR Head (as proposed in [FAST-VQA](https://arxiv.org/abs/2207.02595v1))

IP-NLR head can generate local quality maps for videos.

## Installation
安装

### Dependencies
依赖

The original library is build with

- python=3.8.8
- torch=1.10.2
- torchvision=0.11.3

while using decord module to read original videos (so that you don't need to make any transform on your original .mp4 input).

To get all the requirements, please run

```shell
pip install -r requirements.txt
```

### Direct Install
直接安装

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
使用方法

### Quick Benchmark
快速测试


#### Step 1: Get Pretrained Weights

We supported pretrained weights for several versions:




###

### Train FAST-VQA


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

The following paper is to be cited in the bibliography if relevant papers are proposed.
```
@article{wu2022fastquality,
  title={FAST-VQA: Efficient End-to-end Video Quality Assessment with Fragment Sampling},
  author={Wu, Haoning and Chen, Chaofeng and Hou, Jingwen and Liang, Liao and Wang, Annan and Sun, Wenxiu and Yan, Qiong and Weisi, Lin},
  journal={Proceedings of European Conference of Computer Vision (ECCV)},
  year={2022}
}
```



