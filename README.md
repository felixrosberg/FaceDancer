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

![overview](assets/facedancer_ov.png)

![result_matrix](assets/result_matrix.png)

## Requirements
This project was implemented in TensorFlow 2.X. For evaluation we used models implemented in both TensorFlow and PyTorch (e.g CosFace from [InsightFace](https://github.com/deepinsight/insightface/blob/master/recognition/arcface_torch)).

You can find the exported enviroment in enviroment.yml. Run following command to install the tensorflow enviroment:
```shell
conda env create -f environment.yml
```

## How to Run
First you need to download the pretrained ArcFace [here](https://huggingface.co/felixrosberg/ArcFace) (both ArcFace-Res50.h5 and ArcFacePerceptual-Res50.h5 for training) and RetinaFace [here](https://huggingface.co/felixrosberg/RetinaFace). Secondly you need to train FaceDancer or download a pretrained model weights and its structure from [here](https://huggingface.co/felixrosberg/FaceDancer)(coming soon... ).
- Put the ArcFace models inside the /arcface_model/ directory.
- Put the RetinaFace model inside the /retinaface/ directory.
Note: You can put the ArcFace + RetinaFace models where ever you like, but you have to specify the path in the arguments then, as current default arguments points to the aforementioned directories.
- Put the FaceDancer model somewhere, e.g. pretrained/FaceDancer-C.h5.
To face swap all faces with one source run:
```shell
python video_swap/multi_face_single_source.py --facedancer_path path/to/facedancer.h5 --vid_path path/to/video.mp4 --swap_source path/to/source_face.png
```

### TODO:
- [ ] Add complete code for calculating IFSR.
- [ ] Add code for all evaluation steps.
- [ ] Provide download links to pretrained models.
