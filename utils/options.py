# -*- coding: utf-8 -*-
# @Author: netrunner-exe
# @Date:   2022-05-30 11:09:04
# @Last Modified by:   netrunner-exe
# @Last Modified time: 2022-12-22 14:45:11
import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False


class FaceDancerOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        # Base optinons
        self.parser.add_argument('--device_id', type=int, default=0, const=0, nargs='?',
                                 help='Which device to use.')
        self.parser.add_argument('--retina_path', type=str, default='./retinaface/RetinaFace-Res50.h5',
                                 help='Path to RetinaFace model.')
        self.parser.add_argument('--arcface_path', type=str, default='./arcface_model/ArcFace-Res50.h5',
                                 help='Path to ArcFace model. Used to extract identity from source.')
        self.parser.add_argument('--facedancer_path', type=str, default='./model_zoo/FaceDancer_config_c_HQ.h5',
                                 help='Path to pretrained FaceDancer model')

        self.parser.add_argument('--swap_source', type=str,
                                 help='Path to source face for video swap.')

        self.parser.add_argument('--vid_path', type=str,
                                 help='Path to video to face swap.')

        self.parser.add_argument('--img_path', type=str,
                                 help='Path to image to face swap.')

        self.parser.add_argument('--vid_output', type=str, default="results/swapped_video.mp4",
                                 help='Path to output manipulated video.')
        self.parser.add_argument('--img_output', type=str, default="results/swapped_image.jpg",
                                 help='Path to output manipulated image.')
        self.parser.add_argument('--align_source', type=str2bool, default=True, const=True, nargs='?',
                                 help='If true, detects the face and aligns it before extracting identity.')


    def parse(self, save=True):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        return self.opt
