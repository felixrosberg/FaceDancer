import numpy as np
import tensorflow as tf
import argparse
import os
import pandas as pd

from scipy.spatial.distance import cosine
from tensorflow.keras.models import load_model, Model

from tqdm import tqdm
from PIL import Image


def main(opt):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.set_visible_devices(gpus[opt.device_id], 'GPU')

    blocks = ['conv5_block3_out',
              'conv5_block2_out',
              'conv5_block1_out',
              'conv4_block6_out',
              'conv4_block5_out',
              'conv4_block4_out',
              'conv4_block3_out',
              'conv4_block2_out',
              'conv4_block1_out',
              'conv3_block4_out',
              'conv3_block3_out',
              'conv3_block2_out',
              'conv3_block1_out',
              'conv2_block3_out',
              'conv2_block2_out',
              'conv2_block1_out',
              'conv1_relu'
              ]

    arcface = load_model(opt.arcface_path)
    extractor = arcface.layers[1]
    extractor.summary()
    extractor.trainable = False

    arcface_p = Model(extractor.input, [extractor.get_layer(layer_name).output for layer_name in blocks])

    data_path = opt.data_path

    name = opt.arcface_path.split("/")[0] + "/"
    print(name)

    if not os.path.isdir(name + "block_distances/" + opt.data_name):
        os.mkdir(name + "block_distances/")
        os.mkdir(name + "block_distances/" + opt.data_name)

    # data dicts
    cos_sim_t_s = {}
    cos_sim_t_c = {}
    cos_sim_c_s = {}
    for block in blocks:
        cos_sim_t_s[block] = []
        cos_sim_t_c[block] = []
        cos_sim_c_s[block] = []

    print("[*] Calculating perceptual similarity between targets, sources and changes...")
    for file_name in tqdm(os.listdir(data_path + 'target/'), total=len(os.listdir(data_path + 'target/'))):
        target = np.asarray(Image.open(data_path + 'target/' + file_name).resize((112, 112))) / 255.0
        source = np.asarray(Image.open(data_path + 'source/' + file_name).resize((112, 112))) / 255.0
        change = np.asarray(Image.open(data_path + 'change/' + file_name).resize((112, 112))) / 255.0

        target_features = arcface_p.predict(np.expand_dims(target, axis=0))
        source_features = arcface_p.predict(np.expand_dims(source, axis=0))
        change_features = arcface_p.predict(np.expand_dims(change, axis=0))

        for t_f, s_f, c_f, block_name in zip(target_features, source_features, change_features, blocks):
            t_f = tf.reshape(t_f, (1, -1))
            s_f = tf.reshape(s_f, (1, -1))
            c_f = tf.reshape(c_f, (1, -1))

            t_s = cosine(t_f, s_f)
            t_c = cosine(t_f, c_f)
            c_s = cosine(c_f, s_f)

            cos_sim_t_s[block_name].append(t_s)
            cos_sim_t_c[block_name].append(t_c)
            cos_sim_c_s[block_name].append(c_s)

    print("[*] Saving distances as data frame pickles...")
    for block in blocks:
        df_t_s = pd.DataFrame(cos_sim_t_s[block], columns=['cos_sim'])
        df_t_s.to_pickle(name + "block_distances/" + opt.data_name + "/arcface_" + block + "_t_s.pkl")

        df_t_c = pd.DataFrame(cos_sim_t_c[block], columns=['cos_sim'])
        df_t_c.to_pickle(name + "block_distances/" + opt.data_name + "/arcface_" + block + "_t_c.pkl")

        df_c_s = pd.DataFrame(cos_sim_c_s[block], columns=['cos_sim'])
        df_c_s.to_pickle(name + "block_distances/" + opt.data_name + "/arcface_" + block + "_c_s.pkl")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arcface_path', type=str, default="arcface/arc_res50.h5",
                        help='path to arcface model')
    parser.add_argument('--data_path', type=str, default="C:/path/to/facial_recognition_data/",
                        help='path to data to run comparisons,'
                             'structure should be: data_path/change, target, source/0.png, 1.png, ... n.png')
    parser.add_argument('--device_id', type=int, default=0,
                        help='which device to use')
    parser.add_argument('--data_name', type=str, default='FF',
                        help='name of the data set.')

    opt = parser.parse_args()

    main(opt)







