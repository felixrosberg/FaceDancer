import tensorflow as tf
from tensorflow.keras.models import load_model, model_from_json

import os
import warnings

warnings.filterwarnings("ignore")

from tqdm import tqdm

from dataset.tf_records_parser import get_tf_dataset
from dataset.dataloader import DataLoader
from options.base_options import BaseOptions
from networks.generator import get_generator

if __name__ == '__main__':
    opt = BaseOptions().parse()

    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.set_visible_devices(gpus[opt.device_id], 'GPU')

    G = get_generator(up_types=opt.up_types, mapping_depth=opt.mapping_depth, mapping_size=opt.mapping_size)
    #G.load_weights(opt.chkp_dir + opt.log_name + "/gen/" + "gen" + '_' + str(opt.load) + '.h5')
    # G = load_model_internal("C:/Users/Berge001/PycharmProjects/faceshifter_plus_plus/modules/results/"
    # "adaptive_fusion_models/checkpoints/"
    # "20220104-155634-adaptive_fusion_no_final_residual/gen/", "gen", opt.load)

    if not os.path.isdir(opt.result_dir + opt.log_name):
        os.mkdir(opt.result_dir + opt.log_name)
        os.mkdir(opt.result_dir + opt.log_name + "/inference")

    # Identity and expression encoders
    ArcFace = load_model(opt.arcface_path)


    @tf.function
    def test_step(target, source):
        source_z = ArcFace(tf.image.resize((source + 1) / 2, [112, 112]))
        change = G([target, source_z], training=False)

        return change

    eval_dataset = DataLoader(opt.inference_load_dir, batch_size=32)

    print("[*] Starting profiling of model...")
    step_num = 0
    tf.profiler.experimental.start(opt.profile_dir + opt.log_name)
    for data in tqdm(range(1000)):
        try:
            data = eval_dataset.get_eval_batch()
            target_batch, source_batch = data
            change = test_step(target_batch, source_batch)
        except Exception as e:
            print(e)
    tf.profiler.experimental.stop()
    print("[*] Done.")
