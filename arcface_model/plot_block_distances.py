import pandas as pd
import numpy as np
import argparse
import os
from matplotlib import pyplot as plt


def main(opt):
    pd.set_option('display.max_rows', 1000)
    pd.set_option('display.max_columns', 151)

    dataset = opt.data_name
    name = opt.arcface_path.split("/")[0] + "/"

    meta_dict = {'af': 'ArcFace', 'fn': 'FaceNet', 'l2': 'L2',
                 'sl2': 'squared L2', 'cos': 'cosine similarity', 'l1': 'L1'}
    columns = ['cos_sim']

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

    t_c_margins = []

    if not os.path.isdir(name + "block_distances_plots/"):
        os.mkdir(name + "block_distances_plots")
    if not os.path.isdir(name + "block_distances_plots/" + opt.data_name):
        os.mkdir(name + "block_distances_plots/" + opt.data_name)

    for block in blocks:

        #positive_samples = pd.read_pickle('block_distances/imagenet_pos_neg_c2t_c2s/arcface_' + block + '_distance_df_pos.pkl')
        t_s = pd.read_pickle(name + 'block_distances/' + dataset + '/arcface_' + block + '_t_s.pkl')
        t_c = pd.read_pickle(name + 'block_distances/' + dataset + '/arcface_' + block + '_t_c.pkl')
        c_s = pd.read_pickle(name + 'block_distances/' + dataset + '/arcface_' + block + '_c_s.pkl')

        print(block)
        print(t_s.mean())
        print(t_c.mean())
        print(c_s.mean())
        print('==========================================================')
        print()

        for col in columns:
            bins = np.linspace(0, 1.5, 100)

            fig = plt.figure(dpi=500)
            t_s_s, _, _ = plt.hist(t_s[col].values, bins, alpha=0.5, label='y')
            t_c_s, _, _ = plt.hist(t_c[col].values, bins, alpha=0.5, label='y')
            c_s_s, _, _ = plt.hist(c_s[col].values, bins, alpha=0.5, label='y')
            plt.legend(['t2s', 't2c', 'c2s'], loc='upper right')

            plt.axvline(t_c[col].values.mean(), color='k', linestyle='dashed', linewidth=1)

            min_ylim, max_ylim = plt.ylim()
            plt.text(t_c[col].values.mean() * 1.1, max_ylim * 0.9, 'mean {:.2f}'.format(t_c[col].values.mean()))

            plt.title('ArcFace ResNet50 - ' + block + ' feature cosine similarity distance.')
            plt.xlabel('distance')
            plt.ylabel('facial comparisons')
            plt.savefig(name + 'block_distances_plots/' + dataset + '/' + block + '.png')
            plt.close()

            t_c_margins.append(t_c[col].values.mean())

    np.save(name + 'block_distances_plots/' + dataset + '/perceptual_similarity_margins.npy', np.asarray(t_c_margins))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arcface_path', type=str, default="arcface/arc_res50.h5",
                        help='path to arcface model')
    parser.add_argument('--data_path', type=str, default="D:/forensic_face_swap_data/SimSwap/",
                        help='path to data to run comparisons,'
                             'structure should be: data_path/change, target, source/0.png, 1.png, ... n.png')
    parser.add_argument('--data_name', type=str, default='FF',
                        help='name of the data set.')

    opt = parser.parse_args()

    main(opt)
