# FAST-VQA

The official open source training and inference code for future paper 'FAST-VQA: Fragment Sub-sample Transformer for End-to-end Blind Video Quality Assessment'.

## Abstract

This paper has shown that discontinuous local fragment strips can be an effective spatial sampling strategy for building efficient end-to-end deep video quality assessment (VQA) regardless of its original resolution.  We cut the video into grids and sample fixed-size patches from all grids, so that the re-sampled input keeps sufficient local textures. We also propose two core designs on them. First, we align the patches onto the temporal dimension to keep sufficient perception on inter-frame temporal distortions. Second, we design the Fragment Correlation Transformer based on video swin transformer to learn the global spatial relations between these fragments with transformer network. The proposed FAST-VQA has become the first end-to-end VQA method that reaches state-of-the-art performance. It has also provided 100x faster inference speed for VQA on 1080p videos, while simultaneously increasing 10% more accuracy. We further prove that the FAST-VQA has learnt generalized video quality representations as it reaches state-of-the-art or competitive performance on cross dataset evaluations in diverse VQA datasets without fine-tuning process. The FAST-VQA has also shown good scaling ability by achieving significant improvement on evaluating high-resolution videos and performing well on cross-resolution video quality comparisons.


## Introduction


- We propose the grid-wise fragment sub-sampling (GFS) strategy with temporal fragment alignment (TFA) for building the end-to-end deep learning for video quality assessment. This sub-sampling strategy can reduce the redundant information in videos while keeping sufficient local textures and clearness that are important on predicting video quality precisely.
- We propose the fragment attention network (FANet) to learn both local textures in these fragments and the contextual attention between these fragments. The FANet is based on video swin transformer backbone and learns the perceptual quality of these patches with the global semantic context.
- The whole proposed FAST-VQA pipeline has reached state-of--art performance while boosting the speed of existing VQA models by 50x on 1080p high resolution videos. It also shows excellent generalization ability on different testing scenarios and scaling ability on increasing resolutions.
See in [demos](./demos/) for more examples.

## Build FAST-VQA

### Requirements

The original method is build with

- python=3.8.8
- torch=1.8.1
- torchvision=0.9.1

while using decord module to read original videos (so that you don't need to make any transform on your original .mp4 input).

To get all the requirements, please run

```shell
pip install -r requirements.txt
```

### Benchmarking 

You can directly benchmark the model with mainstream benchmark VQA datasets.

```shell
python inference.py --dataset $DATASET$
```

Available datasets are LIVE_VQC, KoNViD, CVD2014, LSVQ (or 'all' if you want to infer all of them).

## Fine-tune on Small Labeled VQA Datasets


```shell
python finetune.py --dataset $DATASET$
```

Available datasets are LIVE_VQC, KoNViD, CVD2014.

### Inference on Scripts

You can install this directory by running

```shell
python setup.py install
```

Then you can embed these lines into your python scripts:

```python
from fastvqa import deep_end_to_end_vqa

video = torch.randn((96,3,224,224))
vq_evaluator = deep_end_to_end_vqa(pretrained=True)
score = vq_evaluator(video)
print(score)
```