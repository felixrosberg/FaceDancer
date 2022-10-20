import argparse


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        # paths (data, models, etc...)
        self.parser.add_argument('--data_dir', type=str,
                                 default="I:/Datasets/VGGFace2/tfrecords/train/vgg_ls3dx4_train_*-of-*.records",
                                 help='path to train data set shards')
        self.parser.add_argument('--eval_dir', type=str,
                                 default="I:/Datasets/VGGFace2/tfrecords/validation/vgg_ls3dx4_validation_*-of-*.records",
                                 help='path to validation data set shards')
        self.parser.add_argument('--unprocessed_data_dir', type=str, default="D:/vggface2/train_aligned_faceswap/",
                                 help='path to training images (used to calculate data set length)')
        self.parser.add_argument('--arcface_path', type=str,
                                 default="../arcface_model/arcface/arc_res50.h5",
                                 help='path to arcface model. Used to extract identity from source.')
        self.parser.add_argument('--eval_model_arcface', type=str,
                                 default="../arcface_model/arcface/arc_res50_ccrop.h5",
                                 help='path to arcface model. Used to evaluate identity performance.')
        self.parser.add_argument('--eval_model_expface', type=str,
                                 default="../arcface_model/expface/expression_encoder.h5",
                                 help='path to arcface model. Used to evaluate identity performance.')

        # Video/Image necessary models
        self.parser.add_argument('--retina_path', type=str,
                                 default="../retinaface/retinaface_res50.h5",
                                 help='path to retinaface model.')
        self.parser.add_argument('--vid_path', type=str,
                                 default="C:/Users/Berge001/PycharmProjects/"
                                         "anonymizer/v2/manipulation/istanbull.mp4",
                                 help='path to video to face swap')
        self.parser.add_argument('--output', type=str,
                                 default="istanbull_s.mp4",
                                 help='path to output manipulated video')
        self.parser.add_argument('--compare', type=bool,
                                 default=True,
                                 help='If true, concatenates the frame with the manipulated frame')
        self.parser.add_argument('--sample_rate', type=int,
                                 default=1,
                                 help='Sample rate, 1 would include all frames, 2 would only process every 2.')
        self.parser.add_argument('--length', type=float,
                                 default=1,
                                 help='0 to 1. How much of the video to process.')
        self.parser.add_argument('--flip_ratio', type=float,
                                 default=0.8,
                                 help='How much of the identity to flip.')
        self.parser.add_argument('--swap_source', type=str,
                                 default="D:/forensic_face_swap_data/"
                                         "20211218-185442-add_baseline/target/7777.png",
                                 help='path to source face for video sswap.')

        self.parser.add_argument('--load', type=int,
                                 default=None,
                                 help='int of number to load checkpoint weights.')
        self.parser.add_argument('--export', type=bool,
                                 default=False,
                                 help='exports the generator to a complete h5 file.')

        # general
        self.parser.add_argument('--batch_size', type=int, default=10,
                                 help='batch size')
        self.parser.add_argument('--image_size', type=int, default=256,
                                 help='image size')
        self.parser.add_argument('--shift', type=float, default=0.5,
                                 help='image normalization: shift')
        self.parser.add_argument('--scale', type=float, default=0.5,
                                 help='image normalization: scale')
        self.parser.add_argument('--num_epochs', type=int, default=500,
                                 help='number of epochs')

        # hyper parameters
        self.parser.add_argument('--lr', type=float, default=0.0001,
                                 help='learning rate')
        self.parser.add_argument('--lr_decay', type=float, default=0.97,
                                 help='learning rate')
        self.parser.add_argument('--r_lambda', type=float, default=5,
                                 help='reconstruction loss (l1) weighting')
        self.parser.add_argument('--p_lambda', type=float, default=2,
                                 help='perceptual loss weighting')
        self.parser.add_argument('--i_lambda', type=float, default=10,
                                 help='identity loss weighting')
        self.parser.add_argument('--c_lambda', type=float, default=1,
                                 help='cycle loss weighting')
        self.parser.add_argument('--info_lambda', type=float, default=1,
                                 help='perceptual similarity loss weighting')
        self.parser.add_argument('--ifsr_lambda', type=float, default=1,
                                 help='perceptual similarity loss weighting')
        self.parser.add_argument('--ifsr_scale', type=float, default=1.2,
                                 help='perceptual similarity margin scaling.'
                                      '(lower value forces harder similarity between target and change.)')
        self.parser.add_argument('--ifsr_margin', type=list,
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
                                 help='path to IFSR margins')
        self.parser.add_argument('--ifsr_blocks', type=list,
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
        self.parser.add_argument('--p_blocks', type=list,
                                 default=['block1_pool',
                                          'block2_pool',
                                          'block3_pool',
                                          'block4_pool',
                                          'block5_pool',
                                          ],
                                 help='block outputs from VGG19 to use for perceptual loss.')

        self.parser.add_argument('--g_type', type=str, default="affa_5anf_rerun",
                                 help="what kind of generator?")
        self.parser.add_argument('--z_id_size', type=int, default=512,
                                 help="size (dimensionality) of the identity vector")
        self.parser.add_argument('--mapping_depth', type=int, default=4,
                                 help="depth of the mapping network")
        self.parser.add_argument('--mapping_size', type=int, default=512,
                                 help="size of the fully connected layers in the mapping network")
        self.parser.add_argument('--code_size', type=int, default=32,
                                 help="size of the information control code")
        self.parser.add_argument('--up_types', type=list,
                                 default=['affa', 'affa', 'affa', 'affa', 'affa', 'concat'],
                                 help='what kind of decoding blocks to use')

        # data and devices
        self.parser.add_argument('--shuffle', type=bool, default=True,
                                 help='whether to shuffle the data or not')
        self.parser.add_argument('--same_ratio', type=float, default=0.2,
                                 help='chance of an image pair being the same image')
        self.parser.add_argument('--num_workers', type=int, default=4,
                                 help='number of data loading workers')
        self.parser.add_argument('--device_id', type=int, default=1,
                                 help='which device to use')
        self.parser.add_argument('--amp', type=bool, default=True,
                                 help='if true, uses automatic mixed precision')
        self.parser.add_argument('--inference_load_dir', type=str,
                                 default="C:/DATASETS/vggface2/test_aligned/n000078/",
                                 help='path to inference measurement data.')

        # logging and checkpointing
        self.parser.add_argument('--log_dir', type=str, default='../logs/runs/',
                                 help='logging directory')
        self.parser.add_argument('--profile_dir', type=str, default='../logs/profiling/',
                                 help='profiling directory')
        self.parser.add_argument('--log_name', type=str, default='affa_info_v7',
                                 help='name of the run, change this to track several experiments')

        self.parser.add_argument('--chkp_dir', type=str, default='../checkpoints/',
                                 help='checkpoint directory (will use same name as log_name!)')
        self.parser.add_argument('--result_dir', type=str, default='../results/',
                                 help='test results directory (will use same name as log_name!)')
        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        return self.opt