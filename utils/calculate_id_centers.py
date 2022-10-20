import torch
from torchvision import transforms
from arcface_model.resnet import iresnet50

import numpy as np
import os
import argparse

from PIL import Image
from tqdm import tqdm


def main(opt):
    device = torch.device(opt.device_id if torch.cuda.is_available() else "cpu")

    eval_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Resize(112),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    cosface_state_dict = torch.load(opt.cosface_path)
    CosFace = iresnet50()
    CosFace.load_state_dict(cosface_state_dict)
    CosFace.eval()
    CosFace.to(device)

    if opt.mode == 'centers':
        data_path = opt.test_data_dir

        final_id_array_arc = []
        id_list = []

        for vdx, video_id in tqdm(enumerate(os.listdir(data_path)), total=opt.num_ids):

            id_array_arc = np.ones(shape=(len(os.listdir(data_path + video_id)), 512))

            for idx, img in enumerate(os.listdir(data_path + video_id)):
                i_reg = Image.open(data_path + video_id +  "/" + img)
                i_arc = eval_transform(i_reg).unsqueeze(0).to(device)

                id_arc = CosFace(i_arc)

                id_array_arc[idx] = id_arc.cpu().detach().numpy()[0]

            final_id_array_arc[vdx] = id_array_arc.mean(axis=0)
            id_list.append(video_id)

        final_id_array_arc = np.asarray(final_id_array_arc)
        id_list = np.asarray(id_list)

        np.save("results/fixed_ff_identity_centers_cosface.npy", final_id_array_arc)
        np.save("results/fixed_ff_id_list_centers.npy", id_list)

    elif opt.mode == 'spread':
        data_path = opt.test_data_dir

        final_id_array_arc = []
        id_list = []

        for vdx, video_id in tqdm(enumerate(os.listdir(data_path)), total=opt.num_ids):

            id_array_arc = np.ones(shape=(len(os.listdir(data_path + video_id)), 512))

            for idx, img in enumerate(os.listdir(data_path + video_id)):
                i_reg = Image.open(data_path + video_id + "/" + img)
                i_arc = eval_transform(i_reg).unsqueeze(0).to(device)

                id_arc = CosFace(i_arc)

                id_array_arc[idx] = id_arc.cpu().detach().numpy()[0]

                final_id_array_arc.append(id_array_arc.mean(axis=0))
                id_list.append(video_id)

        id_list = np.asarray(id_list)
        final_id_array_arc = np.asarray(final_id_array_arc)
        np.save("results/fixed_ff_identity_spread_cosface.npy", final_id_array_arc)
        np.save("results/fixed_ff_id_list_spread.npy", id_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data_dir', type=str, default="D:/fixed_forensic_v2/",
                        help='Path to test data. Structure: data_folder/id_0, id_1, ..., id_n/img_0, img_1, ..., img_n')
    parser.add_argument('--mode', type=str, default='spread',
                        help='centers creates an average id vector per identity,'
                             'spread creates an id vector per image')
    parser.add_argument('--num_ids', type=int, default=883,
                        help='device to use')
    parser.add_argument('--device_id', type=str, default='cuda:0',
                        help='device to use')
    parser.add_argument('--cosface_path', type=str, default='../arcface_model/glint360k_cosface_r50/backbone.pth',
                        help='path to the cosface model')
    opt = parser.parse_args()
    main(opt)





