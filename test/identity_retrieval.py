import argparse

import numpy as np
import os
import torch
from torchvision import transforms

from arcface_model.resnet import iresnet50
from PIL import Image
from tqdm import tqdm


def nearest_cosine_distance(u, v):
    u_n = u / np.linalg.norm(u, axis=-1, keepdims=True)
    v_n = v / np.linalg.norm(v, axis=-1, keepdims=True)

    d = np.sum(u_n * v_n, axis=-1)

    return np.argmax(d), d[np.argmax(d)]


def main(opt):

    if not os.path.isdir(opt.test_data_dir + "/results/"):
        os.mkdir(opt.test_data_dir + "/results/")

    device = torch.device(opt.device_id if torch.cuda.is_available() else "cpu")

    eval_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Resize(112),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    cosface_state_dict = torch.load(opt.cosface_path)
    CosFace = iresnet50()
    CosFace.load_state_dict(cosface_state_dict)
    CosFace.eval()
    CosFace.to(device)

    arc_face_mean_identities = None
    identity_list = None
    if opt.mode == 'centers':
        arc_face_mean_identities = np.load("../miscellaneous/fixed_ff_identity_centers_cosface.npy")
        identity_list = np.load("../miscellaneous/fixed_ff_id_list.npy")

    elif opt.mode == 'spread':
        arc_face_mean_identities = np.load("../miscellaneous/fixed_ff_identity_spread_cosface.npy")
        identity_list = np.load("../miscellaneous/fixed_ff_id_list_spread.npy")

    num_identities = os.listdir(opt.test_data_dir + "/change/")

    cos_retrieval = 0

    i = 0

    distance_list = []
    distance_match = []
    identity_match = []
    for identity in tqdm(num_identities):

        num_faces = os.listdir(opt.test_data_dir + "/change/" + identity)

        for face in num_faces:

            try:
                i_reg_change = Image.open(opt.test_data_dir + "/change/" + identity + '/' + face)
                i_arc_change = eval_transform(i_reg_change).unsqueeze(0).to(device)

                i_arc = CosFace(i_arc_change)
                i_arc = i_arc.cpu().detach().numpy()

                d, dis = nearest_cosine_distance(arc_face_mean_identities, i_arc)

                distance_list.append(dis)

                meta = np.load(opt.test_data_dir + "/meta/" + str(i) + ".npy")
                if identity_list[d] == meta[-1]:
                    cos_retrieval += 1
                    distance_match.append(dis)
                    identity_match.append(meta[-1])
                    print("MATCH! -", identity_list[d], meta[-1], 'distance -', dis)

                if (i + 1) % 100 == 0:
                    print(cos_retrieval / (i + 1))
                    print(identity_list[d], meta[-1], 'distance -', dis)

                i += 1
            except Exception as e:
                print(e)

    print("ID retrieval:", cos_retrieval / len(num_faces))

    if opt.mode == 'centers':
        np.save(opt.test_data_dir + "/results/" + "cosface_retrieval_center.npy", np.asarray([cos_retrieval]))
        np.save(opt.test_data_dir + "/results/" + "cosface_distance_to_closest.npy", np.asarray(distance_list))
        np.save(opt.test_data_dir + "/results/" + "cosface_distance_for_successful_retrievals.npy",
                np.asarray(distance_list))
        np.save(opt.test_data_dir + "/results/" + "identity_num_for_successful_retrievals.npy",
                np.asarray(distance_list))
    else:
        np.save(opt.test_data_dir + "/results/" + "cosface_retrieval_spread.npy", np.asarray([cos_retrieval]))
        np.save(opt.test_data_dir + "/results/" + "cosface_distance_to_closest.npy", np.asarray(distance_list))
        np.save(opt.test_data_dir + "/results/" + "cosface_distance_for_successful_retrievals.npy",
                np.asarray(distance_list))
        np.save(opt.test_data_dir + "/results/" + "identity_num_for_successful_retrievals.npy",
                np.asarray(distance_list))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data_dir',
                        type=str,
                        default='I:/Datasets/FaceForensic/anonymization_evaluation/facedancer_hq_1/',
                        help='Path to test data. Structure: data_folder/id_0, id_1, ..., id_n/img_0, img_1, ..., img_n')

    parser.add_argument('--mode', type=str, default='spread',
                        help='centers creates an average id vector per identity,'
                             'spread creates an id vector per image')
    parser.add_argument('--device_id', type=str, default='cuda:1',
                        help='device to use')
    parser.add_argument('--cosface_path', type=str, default='../arcface_model/glint360k_cosface_r50/backbone.pth',
                        help='path to the cosface model')
    options = parser.parse_args()
    main(options)


