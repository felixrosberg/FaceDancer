# -*- coding: utf-8 -*-
# @Author: netrunner-exe
# @Date:   2022-11-23 09:52:13
# @Last Modified by:   netrunner-exe
# @Last Modified time: 2022-12-21 17:19:56
import glob
import os
import shutil
import sys

import cv2
import numpy as np
import proglog
from moviepy.editor import AudioFileClip, VideoFileClip
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

from utils.utils import (estimate_norm, get_lm, inverse_estimate_norm,
                         norm_crop, transform_landmark_points)


class suppress_con_output(object):
    def __init__(self):
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        for fd in self.null_fds + self.save_fds:
            os.close(fd)


def run_inference(opt, source, target, RetinaFace,
                  ArcFace, G, result_img_path):
    try:
        source = cv2.imread(source)
        source = cv2.cvtColor(source, cv2.COLOR_RGB2BGR)  
        source = np.array(source)

        if not isinstance(target, str):
          target = target
        else:
          target = cv2.imread(target)

        target = np.array(target)

        source_h, source_w, _ = source.shape
        source_a = RetinaFace(np.expand_dims(source, axis=0)).numpy()[0]
        source_lm = get_lm(source_a, source_w, source_h)
        source_aligned = norm_crop(source, source_lm, image_size=112, shrink_factor=1.0)

        source_z = ArcFace.predict(np.expand_dims(source_aligned / 255.0, axis=0))

        blend_mask_base = np.zeros(shape=(256, 256, 1))
        blend_mask_base[77:240, 32:224] = 1
        blend_mask_base = gaussian_filter(blend_mask_base, sigma=7)

        im = cv2.cvtColor(target, cv2.COLOR_RGB2BGR)
        im_h, im_w, _ = im.shape
        im_shape = (im_w, im_h)

        detection_scale = (im_w // 640) if (im_w > 640) else 1
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

        if opt.compare == True:
            total_img = np.concatenate((im / 255.0, total_img), axis=1)

        total_img = np.clip(total_img * 255, 0, 255).astype('uint8')

        cv2.imwrite(result_img_path, cv2.cvtColor(total_img, cv2.COLOR_BGR2RGB))

    except Exception as e:
        print('\n', e)
        sys.exit(0)


def video_swap(opt, face, input_video, RetinaFace, ArcFace, G, out_video_filename):
    video_forcheck = VideoFileClip(input_video)

    if video_forcheck.audio is None:
        no_audio = True
    else:
        no_audio = False

    del video_forcheck

    if not no_audio:
        video_audio_clip = AudioFileClip(input_video)

    video = cv2.VideoCapture(input_video)
    ret = True
    frame_index = 0
    temp_results_dir = './tmp_frames'

    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)

    if os.path.exists(temp_results_dir):
        shutil.rmtree(temp_results_dir)
    os.makedirs(temp_results_dir, exist_ok=True)

    for frame_index in tqdm(range(frame_count)):
        ret, frame = video.read()
        if ret:
            run_inference(opt, face, frame, RetinaFace, ArcFace, G,
                          os.path.join('./tmp_frames', 'frame_{:0>7d}.png'.format(frame_index)))
    video.release()

    path = os.path.join('./tmp_frames', '*.png')
    image_filenames = sorted(glob.glob(path))
    clips = ImageSequenceClip(image_filenames, fps=fps)

    if not no_audio:
        clips = clips.set_audio(video_audio_clip)

    try:
        clips.write_videofile(out_video_filename, codec='libx264', audio_codec='aac', ffmpeg_params=[
            '-pix_fmt:v', 'yuv420p', '-colorspace:v', 'bt709', '-color_primaries:v', 'bt709',
            '-color_trc:v', 'bt709', '-color_range:v', 'tv', '-movflags', '+faststart'], logger=proglog.TqdmProgressBarLogger(print_messages=False))
    except:
        sys.exit(0)

    print('\nDone! {}'.format(out_video_filename))
