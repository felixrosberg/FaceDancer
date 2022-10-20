import numpy as np
import tensorflow as tf
import os
import datetime
import json

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.set_visible_devices(gpus[0], 'GPU')

from PIL import Image
from tqdm import tqdm
from tensorflow.keras.models import load_model
from scipy.ndimage import gaussian_filter
from options.base_options import BaseOptions
from networks.generator import get_generator
from utils.utils import save_model_internal, load_model_internal, save_training_meta, load_training_meta, log_info


def identity_path(id_idx, identities, test_data_id):
    id = identities[id_idx]
    samples = os.listdir(test_data_id + id + '/')

    sample_idx = np.random.randint(len(samples))
    sample = samples[sample_idx]

    return test_data_id + id + '/' + sample


if __name__ == '__main__':
    opt = BaseOptions().parse()

    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.set_visible_devices(gpus[opt.device_id], 'GPU')

    G = get_generator(up_types=opt.up_types, mapping_depth=opt.mapping_depth, mapping_size=opt.mapping_size)
    G.load_weights(opt.chkp_dir + opt.log_name + "/gen/" + "gen" + '_' + str(opt.load) + '.h5')
    G.summary()

    if not os.path.isdir(opt.result_dir + opt.log_name):
        os.mkdir(opt.result_dir + opt.log_name)
    if not os.path.isdir(opt.result_dir + opt.log_name + "/attention"):
        os.mkdir(opt.result_dir + opt.log_name + "/attention")
        for i, ut in enumerate(opt.up_types):
            current_resolution = str(2**(i + 4)) if i < 5 else 'final'
            if ut == 'affa':
                os.mkdir(opt.result_dir + opt.log_name + "/attention/maps_" + current_resolution)
    if not os.path.isdir("../config/" + opt.log_name):
        os.mkdir("../config/" + opt.log_name)

    # save options
    date = datetime.datetime.today().strftime('%Y_%m_%d_%H_%M')
    with open('../config/' + opt.log_name + '/test_options_' + date + '.txt', 'w') as f:
        json.dump(opt.__dict__, f, indent=2)

    ArcFace = load_model(opt.arcface_path)

    G = get_generator(up_types=opt.up_types, mapping_depth=opt.mapping_depth, mapping_size=opt.mapping_size)
    G.load_weights(opt.chkp_dir + opt.log_name + "/gen/" + "gen" + '_' + str(opt.load) + '.h5')

    feature_maps = [G.get_layer(layer_name).output for layer_name in ["16x16",
                                                                      "32x32",
                                                                      "64x64",
                                                                      "128x128",
                                                                      "256x256",
                                                                      "final"]]

    AttentionModel = tf.keras.models.Model(G.input, feature_maps)

    test_data_id = 'D:/vggface2/test_aligned/'
    identities = os.listdir(test_data_id)

    id_idx_source = np.random.randint(len(identities))
    id_idx_target = np.random.randint(len(identities))

    source_path = identity_path(id_idx_source, identities, test_data_id)
    target_path = identity_path(id_idx_target, identities, test_data_id)
    source = Image.open(source_path)
    target = Image.open(target_path)
    source.save(opt.result_dir + opt.log_name + "/attention" + "/source.png")
    target.save(opt.result_dir + opt.log_name + "/attention" + "/target.png")
    source_array = np.asarray(source)
    target_array = np.asarray(target)
    source = (source_array - 127.5) / 127.5
    target = (target_array - 127.5) / 127.5

    arc_id_source = ArcFace(tf.image.resize((np.expand_dims(source, axis=0) + 1) / 2, [112, 112]))
    attention_maps = AttentionModel.predict([np.expand_dims(target, axis=0), arc_id_source])

    idx = 0
    for i, ut in enumerate(opt.up_types):
        current_resolution = str(2**(i + 4)) if i < 5 else 'final'
        if ut == 'affa':
            att = attention_maps[idx][0]
            idx += 1
            for j in range(att.shape[-1]):
                attention_map = att[:, :, j] * 255.0
                file_name = opt.result_dir + opt.log_name + "/attention/maps_"\
                            + current_resolution + "/attention_map_%d.png" % j
                Image.fromarray(attention_map.astype('uint8')).resize((256, 256)).convert('L').save(file_name)

    change = np.clip(((attention_maps[-1][0] + 1) / 2), 0, 1) * 255.0
    Image.fromarray(change.astype('uint8')).save(opt.result_dir + opt.log_name + "/attention" + "/change.png")

    blend_mask_base = np.zeros(shape=(128, 128))
    blend_mask_base[32:120, 16:112] = 1
    blend_mask_base = gaussian_filter(blend_mask_base, sigma=7) * 255.0

    Image.fromarray(blend_mask_base.astype('uint8')).resize((256, 256)).convert('L').save(opt.result_dir +
                                                                                          opt.log_name +
                                                                                          "/attention" +
                                                                                          "/blend_sq.png")

    seg = np.clip(attention_maps[-2][0][:, :, 0], 0, 1) * 255.0
    seg = seg * (blend_mask_base / 255.0)
    Image.fromarray(seg.astype('uint8')).resize((256, 256)).convert('L').save(opt.result_dir + opt.log_name +
                                                                              "/attention" + "/seg_0.png")

    seg = np.expand_dims(np.asarray(Image.fromarray(seg.astype('uint8')).resize((256, 256))) / 255.0, axis=-1)

    change_blended = change * seg + (np.clip(((target + 1) / 2), 0, 1) * 255.0) * (1 - seg)
    Image.fromarray(change.astype('uint8')).save(opt.result_dir + opt.log_name + "/attention" + "/change_b.png")

    seg_blur = np.squeeze(gaussian_filter(np.round(seg + 0.2), sigma=3) * 255.0, axis=-1)
    Image.fromarray(seg_blur.astype('uint8')).convert('L').save(opt.result_dir + opt.log_name +
                                                                "/attention" + "/seg_blur.png")
