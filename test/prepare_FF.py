
import numpy as np
import cv2
import argparse

from tqdm import tqdm
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str,
                             default="C:/Users/Berge001/PycharmProjects/anonymizer/temporal/"
                                     "face_forensics/original_sequences/youtube/c23/videos/",
                             help='path to FaceForensic++ videos')
    parser.add_argument('--out_dir', type=str,
                             default="I:/Datasets/FaceForensic/frames/",
                             help='path to save video frames')

    opt = parser.parse_args()

    # Prepare to load video

    video_list = os.listdir(opt.data_dir)

    for vid_id in tqdm(video_list, total=1000):

        cap = cv2.VideoCapture(opt.data_dir + vid_id)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if not os.path.isdir(opt.out_dir + vid_id[:3]):
            os.mkdir(opt.out_dir + vid_id[:3])

        for i in range(10):
            frame_index = np.random.randint(total_frames)

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            _, image = cap.read()

            cv2.imwrite(opt.out_dir + vid_id[:3] + '/' + str(i) + '.jpg', image)

            im = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


