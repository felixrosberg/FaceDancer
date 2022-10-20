import argparse

import numpy as np
import tensorflow as tf
import os

from PIL import Image
from tqdm import tqdm
from utils.utils import load_model_internal
from tensorflow.keras.models import load_model
from networks.layers import AdaptiveAttention, AdaIN


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str,
                        default='affa_hq_v2',
                        help='name of the run, change this to track several experiments')
    parser.add_argument('--log_name', type=str,
                        default='affa_hq_v2',
                        help='name of the run, change this to track several experiments')
    parser.add_argument('--chkp_dir', type=str, default='../checkpoints/',
                        help='checkpoint directory (will use same name as log_name!)')
    parser.add_argument('--output_dir', type=str, default='D:/AFLW2000-3D/fs_evaluation/',
                        help='directory to output images.')
    parser.add_argument('--load', type=int, default=50,
                        help='which checkpoint to load.')
    parser.add_argument('--device_id', type=int, default=1,
                        help='which device to use')
    parser.add_argument('--arcface_path', type=str,
                        default="../arcface_model/arcface/arc_res50.h5",
                        help='path to arcface model. Used to extract identity from source.')
    parser.add_argument('--test_data', type=str,
                        default="D:/AFLW2000-3D/fs_evaluation/SimSwap/",
                        help='directory to the test data.')

    opt = parser.parse_args()

    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.set_visible_devices(gpus[opt.device_id], 'GPU')

    G = load_model('../exports/affa_hq_v2/generator_t_28.h5', custom_objects={'AdaIN': AdaIN,
                                                                        'AdaptiveAttention': AdaptiveAttention})

    if not os.path.isdir(opt.output_dir):
        os.mkdir(opt.output_dir)
    if not os.path.isdir(opt.output_dir + opt.log_name):
        os.mkdir(opt.output_dir + opt.log_name)
    if not os.path.isdir(opt.output_dir + opt.log_name + "/change"):
        os.mkdir(opt.output_dir + opt.log_name + "/change")
    if not os.path.isdir(opt.output_dir + opt.log_name + "/target"):
        os.mkdir(opt.output_dir + opt.log_name + "/target")
    if not os.path.isdir(opt.output_dir + opt.log_name + "/source"):
        os.mkdir(opt.output_dir + opt.log_name + "/source")
    if not os.path.isdir(opt.output_dir + opt.log_name + "/meta"):
        os.mkdir(opt.output_dir + opt.log_name + "/meta")

    # Identity and expression encoders
    arcface = load_model(opt.arcface_path)

    sample_counter = 0
    for source_p, target_p, id_p in tqdm(zip(os.listdir(opt.test_data + 'source'),
                                             os.listdir(opt.test_data + 'target'),
                                             os.listdir(opt.test_data + 'meta')),
                                         total=len(os.listdir(opt.test_data + 'source'))):
        target_path = opt.test_data + 'target/' + target_p
        source_path = opt.test_data + 'source/' + source_p

        current_target = (np.asarray(Image.open(target_path)) - 127.5) / 127.5
        current_source = (np.asarray(Image.open(source_path)) - 127.5) / 127.5

        # Extract identity, perform face swap
        arc_id_source = arcface(tf.image.resize((np.expand_dims(current_source, axis=0) + 1) / 2, [112, 112]))
        change = G([np.expand_dims(current_target, axis=0), arc_id_source])[0]

        # Scale back faces
        change = ((change.numpy() + 1) / 2) * 255.0
        target = ((current_target + 1) / 2) * 255.0
        source = ((current_source + 1) / 2) * 255.0

        # Clip and cast faces
        target = np.clip(target, 0, 255).astype('uint8')
        source = np.clip(source, 0, 255).astype('uint8')
        change = np.clip(change, 0, 255).astype('uint8')

        # Save images
        Image.fromarray(target).save(opt.output_dir + opt.log_name + '/target/' + target_p)
        Image.fromarray(source).save(opt.output_dir + opt.log_name + '/source/' + target_p)
        Image.fromarray(change).save(opt.output_dir + opt.log_name + '/change/' + target_p)

        # Prepare meta data: target id, source id used
        meta_data = np.load(opt.test_data + 'meta/' + id_p)
        np.save(opt.output_dir + opt.log_name + '/meta/' + target_p.split(".")[0] + '.npy', meta_data)

        sample_counter += 1

