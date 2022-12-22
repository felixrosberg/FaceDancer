import argparse
import logging
import sys

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from scipy.ndimage import gaussian_filter
from tensorflow.keras.models import load_model
from tensorflow_addons.layers import InstanceNormalization
from tqdm import tqdm

sys.path.insert (0, '.')
from networks.generator import get_generator
from networks.layers import AdaIN, AdaptiveAttention
from retinaface.models import *
from utils.utils import (estimate_norm, get_lm, inverse_estimate_norm,
                         norm_crop, transform_landmark_points)


logging.getLogger().setLevel(logging.ERROR)

arcface_src = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32)

arcface_src = np.expand_dims(arcface_src, axis=0)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False


def swap(opt):
    if len(tf.config.list_physical_devices('GPU')) != 0:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        tf.config.set_visible_devices(gpus[opt.device_id], 'GPU')

    RetinaFace = load_model(opt.retina_path, compile=False,
                            custom_objects={"FPN": FPN,
                                            "SSH": SSH,
                                            "BboxHead": BboxHead,
                                            "LandmarkHead": LandmarkHead,
                                            "ClassHead": ClassHead})
    ArcFace = load_model(opt.arcface_path, compile=False)

    G = load_model(opt.facedancer_path, compile=False,
                   custom_objects={"AdaIN": AdaIN,
                                   "AdaptiveAttention": AdaptiveAttention,
                                   "InstanceNormalization": InstanceNormalization})
    G.summary()

    # Prepare to load video
    cap = cv2.VideoCapture(opt.vid_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    _, image = cap.read()
    im = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    im_h, im_w, _ = im.shape
    im_shape = (im_w, im_h)

    print("Video resolution:", im_shape)
    detection_scale = (im_w // 640) if (im_w > 640) else 1

    vid_out = None

    if opt.compare:
        vid_out = cv2.VideoWriter(opt.output, cv2.VideoWriter_fourcc(*'mp4v'),
                                  int(cap.get(cv2.CAP_PROP_FPS)),
                                  (im_shape[0] * 2,
                                   im_shape[1]))
    else:
        vid_out = cv2.VideoWriter(opt.output, cv2.VideoWriter_fourcc(*'mp4v'),
                                  int(cap.get(cv2.CAP_PROP_FPS)),
                                  (im_shape[0],
                                   im_shape[1]))

    source = np.asarray(Image.open(opt.swap_source).convert('RGB'))
    source_h, source_w, _ = source.shape

    if opt.align_source:
        source_a = RetinaFace(np.expand_dims(source, axis=0)).numpy()[0]
        source_lm = get_lm(source_a, source_w, source_h)
        source_aligned = norm_crop(source, source_lm, image_size=112)
    else:
        source_aligned = cv2.resize(source, [112, 112])

    source_z = ArcFace.predict(np.expand_dims(source_aligned / 255.0, axis=0))

    blend_mask_base = np.zeros(shape=(256, 256, 1))
    blend_mask_base[100:240, 32:224] = 1
    blend_mask_base = gaussian_filter(blend_mask_base, sigma=7)

    for fno in tqdm(range(0, int(total_frames * opt.length), opt.sample_rate)):
        # read frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, fno)
        _, image = cap.read()

        im = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        im_h, im_w, _ = im.shape
        im_shape = (im_w, im_h)

        faces = RetinaFace(np.expand_dims(cv2.resize(im,
                                                     (im_w // detection_scale,
                                                      im_h // detection_scale)), axis=0)).numpy()
        total_img = im / 255.0

        for annotation in faces:
            lm_align = np.array([[annotation[4] * im_w, annotation[5] * im_h],
                                 [annotation[6] * im_w, annotation[7] * im_h],
                                 [annotation[8] * im_w, annotation[9] * im_h],
                                 [annotation[10] * im_w, annotation[11] * im_h],
                                 [annotation[12] * im_w, annotation[13] * im_h]],
                                dtype=np.float32)

            # align the detected face
            M, pose_index = estimate_norm(lm_align, 256, "arcface", shrink_factor=1.0)
            im_aligned = cv2.warpAffine(im, M, (256, 256), borderValue=0.0)

            # face swap
            changed_face_cage = G.predict([np.expand_dims((im_aligned - 127.5) / 127.5, axis=0), source_z])
            changed_face = (changed_face_cage[0] + 1) / 2

            # get inverse transformation landmarks
            transformed_lmk = transform_landmark_points(M, lm_align)

            # warp image back
            iM, _ = inverse_estimate_norm(lm_align, transformed_lmk, 256, "arcface", shrink_factor=1.0)
            iim_aligned = cv2.warpAffine(changed_face, iM, im_shape, borderValue=0.0)

            # blend swapped face with target image
            blend_mask = cv2.warpAffine(blend_mask_base, iM, im_shape, borderValue=0.0)
            blend_mask = np.expand_dims(blend_mask, axis=-1)
            total_img = (iim_aligned * blend_mask + total_img * (1 - blend_mask))

        if opt.compare:
            total_img = np.concatenate((im / 255.0, total_img), axis=1)
            total_img[-112:, :112, :] = source_aligned / 255.0

        vid_out.write(cv2.cvtColor((np.clip(total_img * 255, 0, 255)).astype('uint8'), cv2.COLOR_BGR2RGB))

    vid_out.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Video/Image necessary models
    parser.add_argument('--retina_path', type=str,
                        default="./retinaface/RetinaFace-Res50.h5",
                        help='Path to retinaface model.')
    parser.add_argument('--arcface_path', type=str,
                        default="./arcface_model/ArcFace-Res50.h5",
                        help='Path to arcface model. Used to extract identity from source.')
    parser.add_argument('--facedancer_path', type=str,
                        default="./model_zoo/FaceDancer_config_c_HQ.h5",
                        help='Path to pretrained FaceDancer model.')

    # video / image data to use
    parser.add_argument('--vid_path', type=str,
                        help='Path to video to face swap.')
    parser.add_argument('--swap_source', type=str,
                        help='Path to source face for video swap.')
    parser.add_argument('--output', type=str,
                        default="./results/swapped_video.mp4",
                        help='Path to output manipulated video.')

    # video arguments
    parser.add_argument('--compare', type=str2bool,
                        default=False, const=False, nargs='?',
                        help='If true, concatenates the frame with the manipulated frame.')

    parser.add_argument('--sample_rate', type=int,
                        default=1,
                        help='Sample rate, 1 would include all frames, 2 would only process every 2.')
    parser.add_argument('--length', type=float,
                        default=1,
                        help='0 to 1. How much of the video to process.')
    parser.add_argument('--align_source', type=str2bool, default=True, const=True, nargs='?',
                        help='If true, detects the face and aligns it before extracting identity.')
    # data and devices
    parser.add_argument('--device_id', type=int, default=0,
                        const=0, nargs='?', help='Which device to use.')

    opt = parser.parse_args()

    swap(opt)
