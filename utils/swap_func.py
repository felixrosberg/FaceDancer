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
import subprocess

from utils.utils import (estimate_norm, get_lm, inverse_estimate_norm,
                         norm_crop, transform_landmark_points)


def run_inference(opt, source, target, RetinaFace,
                  ArcFace, FaceDancer, result_img_path, source_z=None):
    try:

        if not isinstance(target, str):
            target = target
        else:
            target = cv2.imread(target)

        target = np.array(target)

        if source_z is None:
            source = cv2.imread(source)
            source = cv2.cvtColor(source, cv2.COLOR_RGB2BGR)
            source = np.array(source)

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
            lm_align = get_lm(annotation, im_w, im_h)

            # align the detected face
            M, pose_index = estimate_norm(lm_align, 256, "arcface", shrink_factor=1.0)
            im_aligned = cv2.warpAffine(im, M, (256, 256), borderValue=0.0)

            # face swap
            face_swap = FaceDancer.predict([np.expand_dims((im_aligned - 127.5) / 127.5, axis=0), source_z])
            face_swap = (face_swap[0] + 1) / 2

            # get inverse transformation landmarks
            transformed_lmk = transform_landmark_points(M, lm_align)

            # warp image back
            iM, _ = inverse_estimate_norm(lm_align, transformed_lmk, 256, "arcface", shrink_factor=1.0)
            iim_aligned = cv2.warpAffine(face_swap, iM, im_shape, borderValue=0.0)

            # blend swapped face with target image
            blend_mask = cv2.warpAffine(blend_mask_base, iM, im_shape, borderValue=0.0)
            blend_mask = np.expand_dims(blend_mask, axis=-1)

            total_img = (iim_aligned * blend_mask + total_img * (1 - blend_mask))

        total_img = np.clip(total_img * 255, 0, 255).astype('uint8')

        cv2.imwrite(result_img_path, cv2.cvtColor(total_img, cv2.COLOR_BGR2RGB))

        return total_img, source_z

    except Exception as e:
        print('\n', e)
        sys.exit(0)


def video_swap(opt, face, input_video, RetinaFace, ArcFace, FaceDancer, out_video_filename):
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

    source_z = None

    for frame_index in tqdm(range(frame_count)):
        ret, frame = video.read()
        if ret:
            _, source_z = run_inference(opt, face, frame, RetinaFace, ArcFace, FaceDancer,
                                        os.path.join('./tmp_frames', 'frame_{:0>7d}.png'.format(frame_index)),
                                        source_z=source_z)
    video.release()

    path = os.path.join('./tmp_frames', '*.png')
    image_filenames = sorted(glob.glob(path))
    clips = ImageSequenceClip(image_filenames, fps=fps)
    name = os.path.splitext(out_video_filename)[0]

    if not no_audio:
        clips = clips.set_audio(video_audio_clip)

    if out_video_filename.lower().endswith('.gif'):
        print("\nCreating GIF with FFmpeg...")
        try:
           subprocess.run('ffmpeg -y -v -8 -f image2 -framerate {} \
               -i "./tmp_frames/frame_%07d.png" -filter_complex "[0:v]split [a][b];[a] \
                   palettegen=stats_mode=single [p];[b][p]paletteuse=dither=bayer:bayer_scale=4" \
                       -y "{}.gif"'.format(fps, name), shell=True, check=True)
           print("\nGIF created: {}".format(out_video_filename))

        except subprocess.CalledProcessError:
            print("\nERROR! Failed to export GIF with FFmpeg")
            print('\n', sys.exc_info())
            sys.exit(0)

    elif out_video_filename.lower().endswith('.webp'):
        try:
            print("\nCreating WEBP with FFmpeg...")
            subprocess.run('ffmpeg -y -v -8 -f image2 -framerate {} \
                -i "./tmp_frames/frame_%07d.png" -vcodec libwebp -lossless 0 -q:v 80 -loop 0 -an -vsync 0 \
                    "{}.webp"'.format(fps, name), shell=True, check=True)
            print("\nWEBP created: {}".format(out_video_filename))

        except subprocess.CalledProcessError:
            print("\nERROR! Failed to export WEBP with FFmpeg")
            print('\n', sys.exc_info())
            sys.exit(0)
    else:
        try:
            clips.write_videofile(out_video_filename, codec='libx264', audio_codec='aac', ffmpeg_params=[
                '-pix_fmt:v', 'yuv420p', '-colorspace:v', 'bt709', '-color_primaries:v', 'bt709',
                '-color_trc:v', 'bt709', '-color_range:v', 'tv', '-movflags', '+faststart'],
                                  logger=proglog.TqdmProgressBarLogger(print_messages=False))
        except Exception as e:
            print("\nERROR! Failed to export video")
            print('\n', e)
            sys.exit(0)

        print('\nDone! {}'.format(out_video_filename))
