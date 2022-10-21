import tensorflow as tf
from tensorflow.keras.models import load_model, model_from_json

import numpy as np
import os
import warnings
import json
import math
import argparse

warnings.filterwarnings("ignore")

from tqdm import tqdm
from datetime import datetime

from dataset.tf_records_parser import get_tf_dataset
from networks.generator import get_generator
from networks.discriminator import get_discriminator
from utils.loss import perceptual_loss_flagged, perceptual_similarity_loss, fs_reconstruction_loss_l1
from utils.utils import save_model_internal, load_model_internal, save_training_meta, load_training_meta, log_info


def run(opt):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.set_visible_devices(gpus[opt.device_id], 'GPU')
    lr = opt.lr

    # evaluation models
    expface_eval = load_model(opt.eval_model_expface, compile=False)

    # training models
    ArcFace = load_model(opt.arcface_path)

    ifsr_blocks = opt.ifsr_blocks
    ifsr_margin = np.asarray(opt.ifsr_margin) * opt.ifsr_scale
    ifsr_weight = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    p_blocks = opt.p_blocks

    G = get_generator(up_types=opt.up_types,
                      mapping_depth=opt.mapping_depth,
                      mapping_size=opt.mapping_size)
    D = get_discriminator()

    # loss weights
    r_lambda = opt.r_lambda
    p_lambda = opt.p_lambda
    i_lambda = opt.i_lambda
    c_lambda = opt.c_lambda
    ifsr_lambda = opt.ifsr_lambda

    # init
    ifsr_loss_function = perceptual_similarity_loss(ifsr_blocks, ifsr_weight, ifsr_margin, opt.arcface_path)
    percept_loss = perceptual_loss_flagged((opt.image_size, opt.image_size, 3), p_blocks, [1, 1, 1, 1, 1])

    if not os.path.isdir(opt.chkp_dir + opt.log_name):
        os.mkdir(opt.chkp_dir + opt.log_name)
        os.mkdir(opt.chkp_dir + opt.log_name + "/dis")
        os.mkdir(opt.chkp_dir + opt.log_name + "/gen")
        os.mkdir(opt.chkp_dir + opt.log_name + "/state")
    if not os.path.isdir("../config/" + opt.log_name):
        os.mkdir("../config/" + opt.log_name)

    print("[*] begin training...")
    iteration = 0
    epoch_s = 0

    # load checkpoint
    if opt.load is not None:
        print("[*] loading checkpoint " + str(opt.load) + "...")
        G = load_model_internal(opt.chkp_dir + opt.log_name + "/gen/", "gen", opt.load)
        D = load_model_internal(opt.chkp_dir + opt.log_name + "/dis/", "dis", opt.load)

        checkpoint_state = load_training_meta(opt.chkp_dir + opt.log_name + "/state/", opt.load)

        epoch_s = checkpoint_state["epoch"]
        iteration = checkpoint_state["iteration"] + 1

        print("[*] continuing at iteration " + str(iteration) + "...")

        # export the model to .h5 and exit
        if opt.export:
            print('exporting G to h5...')
            if not os.path.isdir('../exports/' + opt.log_name):
                os.mkdir('../exports/' + opt.log_name)
            G.save('../exports/' + opt.log_name + '/facedancer_' + str(opt.load) + '.h5')

            exit()

    # save options
    date = datetime.today().strftime('%Y_%m_%d_%H_%M')
    with open('../config/' + opt.log_name + '/options_' + date + '.txt', 'w') as f:
        json.dump(opt.__dict__, f, indent=2)

    # prepare learning rate schedule and optimizers
    p = iteration / 100000
    lr_p = lr * (opt.lr_decay ** p)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        lr_p, 100000, opt.lr_decay, staircase=False, name=None
    )

    g_optim = tf.keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=0, beta_2=0.99)
    d_optim = tf.keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=0, beta_2=0.99)

    @tf.function
    def test_step(target, source):

        source_z = ArcFace(tf.image.resize((source + 1) / 2, [112, 112]))

        change = G([target, source_z])

        target_exp = expface_eval(tf.image.resize((target + 1) / 2, [224, 224]))[0]
        change_exp = expface_eval(tf.image.resize((change + 1) / 2, [224, 224]))[0]

        exp_distance_n_l2 = tf.square(tf.cast(target_exp, tf.float32) - tf.cast(change_exp, tf.float32))

        results = {'validation/expression': tf.reduce_mean(exp_distance_n_l2)}

        return results

    @tf.function
    def train_step(target, source, flags):

        # deterministic seeding for deterministic augmentation
        target_seed = tf.random.uniform(shape=[2], minval=0, maxval=100000, dtype=tf.int32)
        source_seed = tf.random.uniform(shape=[2], minval=0, maxval=100000, dtype=tf.int32)

        # image distortions
        target = tf.image.stateless_random_brightness(target, 0.3, seed=target_seed)
        target = tf.image.stateless_random_contrast(target, 0.9, 1.1, seed=target_seed)
        target = tf.image.stateless_random_saturation(target, 0.9, 1.1, seed=target_seed)

        source = tf.image.stateless_random_brightness(source, 0.3, seed=source_seed)
        source = tf.image.stateless_random_contrast(source, 0.9, 1.1, seed=source_seed)
        source = tf.image.stateless_random_saturation(source, 0.9, 1.1, seed=source_seed)

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            with gen_tape.stop_recording() and disc_tape.stop_recording():
                source_z = ArcFace(tf.image.resize((source + 1) / 2, [112, 112]))
                target_z = ArcFace(tf.image.resize((target + 1) / 2, [112, 112]))

            # generate
            change = G([target, source_z], training=True)

            # discriminate
            real_output = D(source, training=True)
            fake_output = D(change, training=True)

            # adversarial loss
            g_loss = tf.reduce_mean(tf.nn.softplus(-fake_output))

            d_loss_f = tf.reduce_mean(tf.nn.softplus(fake_output))
            d_loss_r = tf.reduce_mean(tf.nn.softplus(-real_output))

            d_loss = d_loss_f + d_loss_r

            # gradient penalty
            real_grads = tf.gradients(tf.reduce_sum(real_output), source)[0]
            gp = tf.reduce_sum(tf.square(real_grads), axis=[1, 2, 3])
            gp_loss = tf.cast(tf.reduce_mean(gp * (10 * 0.5)), tf.float32)

            d_loss += gp_loss

            # reconstruction loss
            r_loss = fs_reconstruction_loss_l1(target, change, flags)
            r_loss = tf.clip_by_value(r_loss, clip_value_min=0, clip_value_max=5)

            p_loss = percept_loss(target, change, flags)

            # identity loss
            change_z = ArcFace(tf.image.resize((change + 1) / 2, [112, 112]))
            i_loss = tf.reduce_mean(1 + tf.losses.cosine_similarity(tf.cast(source_z, tf.float32),
                                                                    tf.cast(change_z, tf.float32)))

            # cycle loss
            cycled = G([change, target_z], training=True)
            c_loss = tf.reduce_mean(tf.abs(cycled - target))

            # interpreted feature similarity regularization
            ifsr_loss = ifsr_loss_function(tf.image.resize((target + 1) / 2, [112, 112]),
                                           tf.image.resize((change + 1) / 2, [112, 112]))

            # total loss
            total_loss = g_loss + \
                         r_lambda * r_loss + \
                         i_lambda * i_loss + \
                         c_lambda * c_loss + \
                         p_lambda * p_loss + \
                         ifsr_lambda * ifsr_loss

            # Optimization step
            gradients_of_generator = gen_tape.gradient(total_loss, G.trainable_variables)
            gradients_of_discriminator = disc_tape.gradient(d_loss, D.trainable_variables)

            g_optim.apply_gradients(zip(gradients_of_generator, G.trainable_variables))
            d_optim.apply_gradients(zip(gradients_of_discriminator, D.trainable_variables))

        # Gather log information
        results = {'generator/g_loss': g_loss,
                   'generator/r_loss': r_lambda * r_loss,
                   'generator/i_loss': i_lambda * i_loss,
                   'generator/c_loss': c_lambda * c_loss,
                   'generator/p_loss': p_lambda * p_loss,
                   'generator/isfr_loss': ifsr_lambda * ifsr_loss,
                   'generator/total_loss': total_loss,
                   'discriminator/d_loss': d_loss,
                   'discriminator/d_loss_f': d_loss_f,
                   'discriminator/d_loss_r': d_loss_r,
                   'discriminator/gradient_penalty': gp_loss}

        return results

    def log_image(sw, target, source, iteration, category='validation/'):

        # extract id information
        source_z = ArcFace(tf.image.resize((source + 1) / 2, [112, 112]))
        target_z = ArcFace(tf.image.resize((target + 1) / 2, [112, 112]))

        # generate face swap and reconstruction
        change = (G([target, source_z]) + 1) / 2
        change_s = (G([target, target_z]) + 1) / 2

        target = (target + 1) / 2
        source = (source + 1) / 2

        # stitch images
        r = []
        change = change.numpy()
        change_s = change_s.numpy()
        for i in range(0, 10, 5):
            r.append(np.concatenate(change[i:i + 5], axis=1))
            r.append(np.concatenate(change_s[i:i + 5], axis=1))
            r.append(np.concatenate(target[i:i + 5], axis=1))
            r.append(np.concatenate(source[i:i + 5], axis=1))

        c1 = np.concatenate(r, axis=0)
        c1 = np.clip(c1, 0.0, 1.0)

        # log images to tensorboard
        with sw.as_default():
            tf.summary.image(category + 'samples', np.expand_dims(c1, axis=0), step=iteration, max_outputs=10)

    # if exporting the model, skip creating summary writer
    if not opt.export:
        summary_writer = tf.summary.create_file_writer(opt.log_dir + opt.log_name)

    iteration_num = 500000
    eval_dataset = iter(get_tf_dataset(opt.eval_dir, opt.image_size, 10, repeat=True))

    # begin/resume training
    for epoch in range(epoch_s, opt.num_epochs):

        train_dataset = iter(get_tf_dataset(opt.data_dir, opt.image_size, opt.batch_size))

        for i in tqdm(range(iteration_num), total=iteration_num):
            try:
                data = train_dataset.__next__()
                target_batch, source_batch = data
                target_batch = target_batch.numpy()
                source_batch = source_batch.numpy()

                # randomly choose data points in batch to have target = source
                source_same = np.random.choice([0, 1], opt.batch_size, p=[1 - opt.same_ratio, opt.same_ratio])

                # at least 1 data point must be target = source
                if np.sum(source_same) == 0:
                    source_same[np.random.randint(opt.batch_size)] = 1

                # set target = source
                for i in range(opt.batch_size):
                    if source_same[i] == 1:
                        source_batch[i] = target_batch[i]

                # optimize
                losses = train_step(target_batch, source_batch, source_same)

                # logging
                if iteration % 100 == 0:
                    log_info(summary_writer, losses, iteration)

                if (iteration % 100 == 0 and iteration < 10000) or (iteration % 1000 == 0):
                    log_image(summary_writer, target_batch, source_batch, iteration, 'training/')

                # checkpoint
                if iteration % 1000 == 0:
                    chkp_num = str(int(np.floor(iteration / 10000)))

                    save_model_internal(G, opt.chkp_dir + opt.log_name + "/gen/", "gen", chkp_num)
                    save_model_internal(D, opt.chkp_dir + opt.log_name + "/dis/", "dis", chkp_num)

                if iteration % 100 == 0:
                    checkpoint_state = {'iteration': iteration, 'epoch': epoch}
                    chkp_num = str(int(np.floor(iteration / 10000)))
                    save_training_meta(checkpoint_state, opt.chkp_dir + opt.log_name + "/state/", chkp_num)

                # soft evaluation
                if iteration % 100 == 0:
                    (e_target, e_source) = eval_dataset.__next__()
                    e_losses = test_step(e_target, e_source)

                    log_info(summary_writer, e_losses, iteration)

                    if (iteration % 100 == 0 and iteration < 10000) or (iteration % 1000 == 0):
                        (v_t_v, v_s_v) = eval_dataset.__next__()
                        log_image(summary_writer, v_t_v, v_s_v, iteration)

                iteration += 1
            except Exception as e:
                print(e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str,
                        default="C:/path/to/tfrecords/train/vgg_ls3dx4_train_*-of-*.records",
                        help='path to train data set shards')
    parser.add_argument('--eval_dir', type=str,
                        default="C:/path/to/tfrecords/validation/vgg_ls3dx4_validation_*-of-*.records",
                        help='path to validation data set shards')
    parser.add_argument('--arcface_path', type=str,
                        default="../arcface_model/arcface/ArcFace-Res50.h5",
                        help='path to arcface model. Used to extract identity from source.')
    parser.add_argument('--eval_model_expface', type=str,
                        default="../arcface_model/expface/ExpressionEmbedder-B0.h5",
                        help='path to arcface model. Used to evaluate experssion performance.')

    parser.add_argument('--load', type=int,
                        default=None,
                        help='int of number to load checkpoint weights.')
    parser.add_argument('--export', type=bool,
                        default=False,
                        help='exports the generator to a complete h5 file and exits the training script.')

    # general
    parser.add_argument('--batch_size', type=int, default=10,
                        help='batch size')
    parser.add_argument('--image_size', type=int, default=256,
                        help='image size')
    parser.add_argument('--shift', type=float, default=0.5,
                        help='image normalization: shift')
    parser.add_argument('--scale', type=float, default=0.5,
                        help='image normalization: scale')
    parser.add_argument('--num_epochs', type=int, default=500,
                        help='number of epochs')

    # hyper parameters
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.97,
                        help='learning rate')

    parser.add_argument('--r_lambda', type=float, default=5,
                        help='reconstruction loss (l1) weighting')
    parser.add_argument('--p_lambda', type=float, default=0.2,
                        help='perceptual loss weighting')
    parser.add_argument('--i_lambda', type=float, default=10,
                        help='identity loss weighting')
    parser.add_argument('--c_lambda', type=float, default=1,
                        help='cycle loss weighting')
    parser.add_argument('--ifsr_lambda', type=float, default=1,
                        help='perceptual similarity loss weighting')

    parser.add_argument('--ifsr_scale', type=float, default=1.2,
                        help='perceptual similarity margin scaling.'
                             '(lower value forces harder similarity between target and change.)')
    parser.add_argument('--ifsr_margin', type=list,
                        default=[0.121357,
                                 0.128827,
                                 0.117972,
                                 0.109391,
                                 0.097296,
                                 0.089046,
                                 0.044928,
                                 0.048719,
                                 0.047487,
                                 0.047970,
                                 0.035144],
                        help='IFSR margins')
    parser.add_argument('--ifsr_blocks', type=list,
                        default=['conv4_block6_out',
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
                                 ],
                        help='block outputs from ArcFace to use for IFSR.')
    parser.add_argument('--p_blocks', type=list,
                        default=['block1_pool',
                                 'block2_pool',
                                 'block3_pool',
                                 'block4_pool',
                                 'block5_pool',
                                 ],
                        help='block outputs from VGG16 to use for perceptual loss.')

    parser.add_argument('--z_id_size', type=int, default=512,
                        help="size (dimensionality) of the identity vector")
    parser.add_argument('--mapping_depth', type=int, default=4,
                        help="depth of the mapping network")
    parser.add_argument('--mapping_size', type=int, default=512,
                        help="size of the fully connected layers in the mapping network")
    parser.add_argument('--up_types', type=list,
                        default=['no_skip', 'no_skip', 'affa', 'affa', 'affa', 'concat'],
                        help='what kind of decoding blocks to use')

    # data and devices
    parser.add_argument('--shuffle', type=bool, default=True,
                        help='whether to shuffle the data or not')
    parser.add_argument('--same_ratio', type=float, default=0.2,
                        help='chance of an image pair being the same image')
    parser.add_argument('--device_id', type=int, default=0,
                        help='which device to use')

    # logging and checkpointing
    parser.add_argument('--log_dir', type=str, default='../logs/runs/',
                        help='logging directory')
    parser.add_argument('--log_name', type=str, default='facedancer',
                        help='name of the run, change this to track several experiments')

    parser.add_argument('--chkp_dir', type=str, default='../checkpoints/',
                        help='checkpoint directory (will use same name as log_name!)')
    parser.add_argument('--result_dir', type=str, default='../results/',
                        help='test results directory (will use same name as log_name!)')

    opt = parser.parse_args()

    run(opt)
