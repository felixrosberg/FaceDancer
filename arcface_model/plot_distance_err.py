import pandas as pd
import numpy as np

from tqdm import tqdm
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 151)

meta_dict = {'af': 'ArcFace', 'fn': 'FaceNet', 'l2': 'L2', 'sl2': 'squared L2', 'cos': 'cosine similarity', 'l1': 'L1'}

#positive_samples = distance_df[distance_df.comparison_label == 1]
#negative_samples = distance_df[distance_df.comparison_label == 0]
columns = ['af_cos']

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

blocks.reverse()

t_c_margins = []


def calculate_th(t, h0, h1, h_min, h_max):
    compare = []
    for i in range(len(t)):
        bin_val = t[i]
        if h_min < bin_val < h_max:
            compare.append(np.max([h0[i], h1[i]]))
        else:
            compare.append(100000)

    th_0 = np.argmin(compare)
    th_1 = len(compare) - np.argmin(compare[::-1]) - 1

    th_mid = int((th_0 + th_1) / 2)

    return th_0, th_1, th_mid


def calculate_th_2(t, h0, h1, h_min, h_max):
    compare = []
    compare_val_0 = []
    compare_val_1 = []
    for i in range(len(h0)):
        h_0_l = np.sum(h0[:i])
        h_1_r = np.sum(h1[i:])

        err_0 = np.abs(h_0_l - h_1_r)

        compare.append(err_0)
        compare_val_0.append(h_0_l / np.sum(h0))
        compare_val_1.append(h_1_r / np.sum(h1))

    #print(compare_val_0[np.argmin(compare)], compare_val_1[np.argmin(compare)])

    eer_val = (compare_val_0[np.argmin(compare)] + compare_val_1[np.argmin(compare)]) / 2
    return np.argmin(compare), eer_val


th_list_neg_c_s = []
th_list_t_c_c_s = []
th_list_t_c_neg = []

val_list_neg_c_s = []
val_list_t_c_c_s = []
val_list_t_c_neg = []

th_list_neg_c_s_0 = []
th_list_neg_c_s_1 = []

th_list_t_c_neg_0 = []
th_list_t_c_neg_1 = []

th_list_t_c_c_s_0 = []
th_list_t_c_c_s_1 = []

smooth = False


for block in tqdm(blocks):

    #positive_samples = pd.read_pickle('block_distances/imagenet_pos_neg_c2t_c2s/arcface_' + block + '_distance_df_pos.pkl')
    neg = pd.read_pickle('block_distances/pos_neg_c2t_c2s/arcface_' + block + '_distance_df_neg.pkl')
    t_c = pd.read_pickle('block_distances/pos_neg_c2t_c2s/arcface_' + block + '_distance_df_c2t.pkl')
    c_s = pd.read_pickle('block_distances/pos_neg_c2t_c2s/arcface_' + block + '_distance_df_c2s.pkl')
    col = 'af_cos'

    bins = np.linspace(0, 1.5, 250)

    fig = plt.figure(dpi=500)

    meta_info = col.split('_')

    if smooth:
        neg_s, _, _ = plt.hist(gaussian_filter(neg[col].values, sigma=0.1), bins, alpha=0.5, label='y')
        t_c_s, _, _ = plt.hist(gaussian_filter(t_c[col].values, sigma=0.1), bins, alpha=0.5, label='y')
        c_s_s, _, _ = plt.hist(gaussian_filter(c_s[col].values, sigma=0.1), bins, alpha=0.5, label='y')
    else:
        neg_s, _, _ = plt.hist(neg[col].values, bins, alpha=0.5, label='y')
        t_c_s, _, _ = plt.hist(t_c[col].values, bins, alpha=0.5, label='y')
        c_s_s, _, _ = plt.hist(c_s[col].values, bins, alpha=0.5, label='y')

    plt.close()

    min_ylim, max_ylim = plt.ylim()

    neg_mean = neg[col].values.mean()
    t_c_mean = t_c[col].values.mean()
    c_s_mean = c_s[col].values.mean()

    # target-change compared to change_source
    min_th = np.min([c_s_mean, t_c_mean])
    max_th = np.max([c_s_mean, t_c_mean])

    th0, th1, th_mid = calculate_th(bins, c_s_s, t_c_s, min_th, max_th)
    eer_th, eer_val = calculate_th_2(bins, c_s_s, t_c_s, min_th, max_th)

    th_list_t_c_c_s.append(bins[eer_th])
    val_list_t_c_c_s.append(eer_val)
    th_list_t_c_c_s_0.append(bins[th0])
    th_list_t_c_c_s_1.append(bins[th1])

    # target-change compared to negative samples
    min_th = np.min([neg_mean, t_c_mean])
    max_th = np.max([neg_mean, t_c_mean])

    th0, th1, th_mid = calculate_th(bins, neg_s, t_c_s, min_th, max_th)
    eer_th, eer_val = calculate_th_2(bins, neg_s, t_c_s, min_th, max_th)

    th_list_t_c_neg.append(bins[eer_th])
    val_list_t_c_neg.append(eer_val)
    th_list_t_c_neg_0.append(bins[th0])
    th_list_t_c_neg_1.append(bins[th1])

    # negative compared to change-source
    min_th = np.min([neg_mean, c_s_mean])
    max_th = np.max([neg_mean, c_s_mean])

    th0, th1, th_mid = calculate_th(bins, neg_s, c_s_s, min_th, max_th)
    eer_th, eer_val = calculate_th_2(bins, neg_s, c_s_s, min_th, max_th)

    th_list_neg_c_s.append(bins[eer_th])
    val_list_neg_c_s.append(eer_val)
    th_list_neg_c_s_0.append(bins[th0])
    th_list_neg_c_s_1.append(bins[th1])

fig = plt.figure(dpi=500)
plt.plot(range(len(val_list_t_c_c_s)), val_list_t_c_c_s, 's--', color='#F542C2')

plt.plot(range(len(val_list_t_c_neg)), val_list_t_c_neg, 'o--', color='#2ACAEA')

plt.plot(range(len(val_list_neg_c_s)), val_list_neg_c_s, '^--', color='#EB8F34')
plt.legend(['c2t-c2s-eer', 'c2t-neg-eer', 'neg-c2s-eer'], loc='upper right')
plt.title('EER between feature similarity distributions')
plt.xlabel('ArcFace block')
plt.ylabel('EER')
plt.savefig('arcface_block_eer_plots/cos_t2c_neg_c2s_eer.png')
plt.close()

fig = plt.figure(dpi=500)
plt.plot(range(len(th_list_t_c_c_s)), th_list_t_c_c_s, 's--', color='#F542C2')
plt.plot(range(len(th_list_t_c_neg)), th_list_t_c_neg, 'o--', color='#2ACAEA')
plt.plot(range(len(th_list_neg_c_s)), th_list_neg_c_s, '^--', color='#EB8F34')
#plt.axvline(21.5, color='k', linestyle='dashdot', linewidth=1)
plt.legend(['t2c-c2s-eer', 't2c-neg-eer', 'neg-c2s-eer'], loc='upper right')
plt.title('EER threshold between feature similarity distributions')
plt.xlabel('ArcFace block')
plt.ylabel('threshold')
plt.savefig('arcface_block_eer_plots/cos_t2s_t2c_c2s_threshold.png')
plt.close()


"""
#p_s, _, _ = plt.hist(positive_samples[col].values, bins, alpha=0.5, label='x')
n_s, _, _ = plt.hist(negative_samples[col].values, bins, alpha=0.5, label='y')
c2t_s, _, _ = plt.hist(c2t_samples[col].values, bins, alpha=0.5, label='y')
c2s_s, _, _ = plt.hist(c2s_samples[col].values, bins, alpha=0.5, label='y')
plt.legend(['Negative samples', 'c2t samples', 'c2s samples'], loc='upper right')

#positive_area = np.sum(p_s)
negative_area = np.sum(n_s)
#intersection = histogram_intersection(p_s, n_s)
#union = positive_area + negative_area - intersection
#iou = np.round((intersection / union) * 100, decimals=2)

plt.title(meta_dict[meta_info[0]] + ' embedding ' + meta_dict[meta_info[1]] + ' distance.')
plt.xlabel('Distance')
plt.ylabel('Facial comparisons')
#plt.text(np.min(p_s), np.max(n_s) * 0.9, r'IoU = ' + str(iou) + '%')
plt.savefig('arcface_block_swapping_distance_plots_no_pos/' + col + '/' + block + '.png')
plt.close()
"""



