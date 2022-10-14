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
First you need to download the pretrained ArcFace [here](https://drive.google.com/drive/folders/1z2-346FHvh3U85CEbtrjVbCbY2FyYwxt?usp=sharing) (both arc_res50.h5 and arc_percept_res50.h5) and RetinaFace [here](https://drive.google.com/drive/folders/1MhEwzpgJaD4zEQbeL9O-tj0Pu3hjw3En?usp=sharing). Secondly you need to train FaceDancer or download a pretrained model weights and its structure from [here](https://drive.google.com/drive/folders/159UscBao617Oe7k_Lq9AQ1S-XoHMVFGU?usp=sharing)(coming soon... ). Put the ArcFace models inside the /arcface_model/ directory, put the RetinaFace model inside the /retinaface/ directory and put the weights (e.g. gen_xx.h5) and the model strucutre json (e.g. gen.json) inside /checkpoints/log_name*/gen/.
Note that currently, the face swapping script demands you choose a checkpoint folder (e.g checkpoints) and the *log_name (e.g facedancer) and specify this in the arguments.
```shell
python run stuff
```

### TODO:
- [ ] Add complete code for calculating IFSR.
- [ ] Add code for all evaluation steps.
- [ ] Provide download links to pretrained models.
