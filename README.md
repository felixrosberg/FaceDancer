# FaceDancer: Pose- and Occlusion-Aware High Fidelity Face Swapping
![demo_vid_0](assets/133_to_4.gif)
\[[Arxiv](https://arxiv.org/abs/2210.10473)\] \[WACV 2023](Coming soon...)\]  \[[Video Results](https://drive.google.com/drive/folders/1hHjK0W-Oo1HD6OZb97IdSifPs4_c6NNo?usp=sharing)\]
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

For a quick play around, you can check out a version of FaceDancer hosted on [Hugging Face](https://huggingface.co/spaces/felixrosberg/face-swap). The Space allow you to face swap images, but also try some other functionality I am currently researching, which I plan to publish soon. For example, reconstruction attacks and adversarial defense against the reconstruction attacks.

![result_matrix](assets/result_matrix.png)

## Requirements
This project was implemented in TensorFlow 2.X. For evaluation we used models implemented in both TensorFlow and PyTorch (e.g CosFace from [InsightFace](https://github.com/deepinsight/insightface/blob/master/recognition/arcface_torch)).

You can find the exported enviroment in enviroment.yml. Run following command to install the tensorflow enviroment:
```shell
conda env create -f environment.yml
```

## How to Face Swap Video
and ArcFacePerceptual-Res50.h5 for training
First you need to download the pretrained ArcFace [here](https://huggingface.co/felixrosberg/ArcFace) (only ArcFace-Res50.h5 is needed for swapping) and RetinaFace [here](https://huggingface.co/felixrosberg/RetinaFace). Secondly you need to train FaceDancer or download a pretrained model weights and its structure from [here](https://huggingface.co/felixrosberg/FaceDancer).
- Put the ArcFace models inside the /arcface_model/ directory.
- Put the RetinaFace model inside the /retinaface/ directory.

Note: You can put the ArcFace + RetinaFace models where ever you like, but you have to specify the path in the arguments then, as current default arguments points to the aforementioned directories.
- Put the FaceDancer model somewhere, e.g. pretrained/FaceDancer-C.h5.

To face swap all faces with one source run:
```shell
python video_swap/multi_face_single_source.py --facedancer_path path/to/facedancer.h5 --vid_path path/to/video.mp4 --swap_source path/to/source_face.png
```

This will output a manipulated video named swapped_video.mp4

## Using the Models in Custom script
```python
from networks.layers import AdaIN, AdaptiveAttention
from tensorflow_addons.layers import InstanceNormalization
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import cv2

# To hide "WARNING:root:The given value for groups will be overwritten."
import logging
logging.getLogger().setLevel(logging.ERROR)

# To hide very long tensorflow log like:
# Model: "model"
# __________________________________________________________________________________________________
#  Layer (type)                   Output Shape         Param #     Connected to                     
# ==================================================================================================
#  input_1 (InputLayer)           [(None, 256, 256, 3  0           []                               
#
# Can be added directly to networks/layers.py
import tensorflow as tf
tf.keras.utils.disable_interactive_logging()

# Add compile=False to hide
# "WARNING:tensorflow:No training configuration found in the save file, so the model was *not* compiled. Compile it manually."

model = load_model("path/to/model.h5", compile=False, custom_objects={"AdaIN": AdaIN, "AdaptiveAttention": AdaptiveAttention, "InstanceNormalization": InstanceNormalization})
arcface = load_model("path/to/arcface.h5", compile=False)

# target and source images need to be properly cropeed and aligned
target = np.asarray(Image.open("path/to/target_face.png").resize((256, 256)))
source = np.asarray(Image.open("path/to/source_face.png").resize((112, 112)))

source_z = arcface(np.expand_dims(source / 255.0, axis=0))

face_swap = model([np.expand_dims((target - 127.5) / 127.5, axis=0), source_z]).numpy()
face_swap = (face_swap[0] + 1) / 2
face_swap = np.clip(face_swap * 255, 0, 255).astype('uint8')

cv2.imwrite("./swapped_face.png", cv2.cvtColor(face_swap, cv2.COLOR_BGR2RGB))
```

The important part is that you need ArcFace as well and make sure the target image is normalized between -1 and 1, and the source between 0 and 1.

## How to Preprocess Data

### Aligning Faces
Before you can train FaceDancer you must make sure the data is properly aligned and processed. Learning capabilites is crippled without this step, if not impossible. The expected folder structure is DATASET/subfolders/im_0, ..., im_x. If using an image dataset not divided into subfolders you can put the DATASET folder inside a parent folder like this: PARENT_FOLDER/DATASET/im_0, ..., im_x. Then specify the PARENT_FOLDER as the --data_dir and the DATASET will be treated as a subfolder. This step requires the pretrained RetinaFace for face detection and facial landmark extraction.

To align the faces run:
```shell
python dataset/crop_align.py --data_dir path/to/DATASET --target_dir path/to/processed_DATASET
```

Remaining arguments consist of:
- --im_size, default=256, final image size of the processed image
- --min_size, defualt=128, threshold to ignore image with a width or height lower than min_size
- --shrink_factor, defualt=1.0, this argument controls how much of the background to keep. Default is 1.0 which produces images appropriate as direct input into ArcFace. If the shrink factor is e.g 0.75, you must center crop the image, keeping 0.75% of the image, before inputting into ArcFace.
- --device_id, default=0, which device to use

### Sharding the Data
This step will convert the image data to tfrecords. If using large datasets such as VGGFace2 this will take some time. However, the training code is designed around this step and it speeds up training significantly. The expected folder structure is DATASET/subfolders/im_0, ..., im_x. If using an image dataset not divided into subfolders you can put the DATASET folder inside a parent folder like this: PARENT_FOLDER/DATASET/im_0, ..., im_x. Then specify the PARENT_FOLDER as the --data_dir and the DATASET will be treated as a subfolder.

To shard the data run:
```shell
python dataset/dataset_sharding.py --data_dir path/to/DATASET --target_dir path/to/tfrecords/dir --data_name dataset_name
```

Remaining arguments consist of:
- --data_type, default="train", identifier for the output file names
- --shuffle, defualt=True, where to shuffle the order of sharding the images
- --num_shards, defualt=1000, how many shards to divide the data into

## How to Train
After you have processed and sharded all your desired datasets, you can train a version of FaceDancer. You still need to the pretrained ArcFace [here](https://huggingface.co/felixrosberg/ArcFace) (both ArcFace-Res50.h5 and ArcFacePerceptual-Res50 is needed). Secondly you need to the expression embedding model used for a rough estimation [here](https://huggingface.co/felixrosberg/ExpressionEmbedder). Put the .h5 files into arcface_model/arcface/ and arcface_model/expface/ respectively and you should need to specify the path in arguments. The trining scipt has the IFSR margins built-in into the default field of its argument. The training and validation data path uses a specific format: C:/path/to/tfrecords/train/DATASET-NAME_DATA-TYPE_\*-of-\*.records, where DATASET-NAME and DATA-TYPE is the arguments specified in the sharding. For example, DATASET-NAME=vggface2 and DATA-TYPE=train: C:/path/to/tfrecords/train/vggface2_train_\*-of-\*.records.

To train run:
```shell
python train/train.py --data_dir C:/path/to/tfrecords/train/dataset_train_*-of-*.records --eval_dir C:/path/to/tfrecords/val/dataset_val_*-of-*.records
```

You can monitor the training with tensorboard. The train.py script will automatically log losses and images into logs/runs/facdancer/ unless you specify a different log directory and/or log name (facedancer is the default log name). Checkpoints will automatically be saved into checkpoints/ directory unless you specify a different directory. The checkpointing saves the model structures to .json and the weights to .h5 files. If you want the complete model in a single .h5 file you can rerun train.py with --load XX and --export True. This will save the complete model as a .h5 file in exports/facedancer/. XX is the checkpoint weight identifier, which can be found if you go to your checkpoints directory and for example, look up gen/gen_XX.h5.

## PyTorch Implementation
Currently I am working on a PyTorch version of FaceDancer. The training and network code is kind of done. Currently the behaviour compare to TensorFlow is drastically different. Some interesting notes is that the mapping network does not allow for the FaceDancer to learn its task. In current state it provides decent results with the mapping network ommited. I will post the PyTorch version as soon as these issues is diagnosed and resolved.

## Citation
If you use this repository in your work, please cite us:
```
@inproceedings{Rosberg2023FaceDancer,
  title     = {FaceDancer: Pose- and Occlusion-Aware High Fidelity Face Swapping},
  author    = {F. Rosberg, E. Aksoy. C. Englund, F. Alonso-Fernandez},
  booktitle = {Proc. IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year      = {2023}
}
```

### TODO:
- [ ] Add complete code for calculating IFSR.
- [ ] Add code for all evaluation steps.
- [x] Provide download links to pretrained models.
- [ ] Image swap script.
- [ ] Debugging?
