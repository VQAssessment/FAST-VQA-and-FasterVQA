# STAFF-VQ

The official open source inference code for paper 'Spatial-Temporal Aligned Fractal Fragments for Video Quality Assessment'.

This repo is now only for the baseline model (without Fractal Fragments).

## Demos

![Demo](./demos/demo_F003.mp4.png)

![Demo](./demos/demo_A004.mp4.png)


With the fragments, the proposed model can distinguish quality defects both on **Bad Content** and **Technical Distortion**.

See in [demos](./demos/) for more examples.

## Use STAFF-VQ

### Get Started

```shell
pip install -r requirements.txt
```

### Inference on VQA Datasets


```shell
python inference_dataset.py --dataset $DATASET$
```

Available datasets are LIVE_VQC, KoNViD, CVD2014, LSVQ.

### Inference on Videos (Not Yet Supported)

TBA.
