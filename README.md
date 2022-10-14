# FaceDancer: Pose- and Occlusion-Aware High Fidelity Face Swapping
![demo_vid_0](133_to_4.gif)
## Abstract
>In this work, we present a new single-stage method for
>subject agnostic face swapping and identity transfer, named
>FaceDancer. We have two major contributions: Adaptive
>Feature Fusion Attention (AFFA) and Interpreted Feature
>Similarity Regularization (IFSR). The AFFA module is embedded
> in the decoder and adaptively learns to fuse attribute
> features and features conditioned on identity information
> without requiring any additional facial segmentation process.
>In IFSR, we leverage the intermediate features
> in an identity encoder to preserve important attributes
> such as head pose, facial expression, lighting, and occlusion
> in the target face, while still transferring the identity
> of the source face with high fidelity. We conduct extensive
> quantitative and qualitative experiments on various
> datasets and show that the proposed FaceDancer outperforms
> other state-of-the-art networks in terms of identity
> transfer, while having significantly better pose preservation
> than most of the previous methods.

![overview](facedancer_ov.png)

![result_matrix](sdfsd.png)

## Requirements
This project was implemented in TensorFlow 2.X. For evaluation we used models implemented in both TensorFlow and PyTorch (e.g CosFace from [InsightFace](https://github.com/deepinsight/insightface/blob/master/recognition/arcface_torch)).

You can find the exported enviroment in enviroment.yml. Run following command to install the tensorflow enviroment:
```shell
conda env create -f environment.yml
```

## How to Run
```shell
python run stuff
```

### TODO:
- [ ] Add complete code for calculating IFSR.
- [ ] Add code for all evaluation steps.
- [ ] Provide download links to pretrained models.
