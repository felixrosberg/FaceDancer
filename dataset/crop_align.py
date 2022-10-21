import numpy as np
import argparse
import tqdm
import os

from utils.utils import norm_crop
from PIL import Image

from keras.models import load_model
from retinaface.models import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="C:/path/to/DATASET/",
                        help='path to dataset directory of face images')
    parser.add_argument('--target_dir', type=str, default="C:/path/to/OUTPUTS/",
                        help='path to save the aligned data.')
    parser.add_argument('--im_size', type=int, default=256,
                        help='Image size of the processed images.')
    parser.add_argument('--min_size', type=int, default=128,
                        help='minimum allow resolution')
    parser.add_argument('--shrink_factor', type=float, default=1.0,
                        help='This argument controls how much of the background to keep.'
                             'Default is 1.0 which produces images appropriate as direct input into ArcFace.'
                             'If the shrink factor is e.g 0.75, you must center crop the image, keeping 0.75% of the'
                             'image, before inputting into ArcFace.')
    parser.add_argument('--device_id', type=int, default=0,
                        help='which device to use')

    opt = parser.parse_args()

    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.set_visible_devices(gpus[opt.device_id], 'GPU')

    im_folder = opt.data_dir

    im_target = opt.target_dir

    face_detector = load_model("models/retinaface_res50.h5", custom_objects={"FPN": FPN,
                                                                             "SSH": SSH,
                                                                             "BboxHead": BboxHead,
                                                                             "LandmarkHead": LandmarkHead,
                                                                             "ClassHead": ClassHead})

    print("[*] Aligning and extracting images...")
    for fld in tqdm.tqdm(os.listdir(im_folder)):
        for img in os.listdir(im_folder + fld):
            try:
                im = Image.open(im_folder + fld + '/' + img)
                im = np.asarray(im)

                im_h, im_w, _ = im.shape

                if im_h > opt.min_size and im_w > opt.min_size:

                    ann = face_detector(np.expand_dims(im, axis=0)).numpy()[0]
                    lm_align = np.array([[ann[4] * im_w, ann[5] * im_h],
                                         [ann[6] * im_w, ann[7] * im_h],
                                         [ann[8] * im_w, ann[9] * im_h],
                                         [ann[10] * im_w, ann[11] * im_h],
                                         [ann[12] * im_w, ann[13] * im_h]],
                                        dtype=np.float32)

                    im_aligned = norm_crop(im, lm_align, image_size=opt.im_size, shrink_factor=opt.shrink_factor)
                    if not os.path.isdir(im_target + fld):
                        os.mkdir(im_target + fld)
                    Image.fromarray(im_aligned).save(im_target + fld + '/' + img)
            except Exception as e:
                print(e)
    print("[*] Done.")
