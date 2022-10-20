import json
import numpy as np
import cv2
import math

import tensorflow as tf
from tensorflow.keras.models import model_from_json
from networks.layers import AdaIN, AdaptiveAttention

from skimage import transform as trans
from scipy.signal import convolve2d
from skimage.color import rgb2yuv, yuv2rgb

from PIL import Image


def save_model_internal(model, path, name, num):
    json_model = model.to_json()
    with open(path + name + '.json', "w") as json_file:
        json_file.write(json_model)

    model.save_weights(path + name + '_' + str(num) + '.h5')


def load_model_internal(path, name, num):
    with open(path + name + '.json', 'r') as json_file:
        model_dict = json_file.read()

    mod = model_from_json(model_dict, custom_objects={'AdaIN': AdaIN, 'AdaptiveAttention': AdaptiveAttention})
    mod.load_weights(path + name + '_' + str(num) + '.h5')

    return mod


def save_training_meta(state_dict, path, num):
    with open(path + str(num) + '.json', 'w') as json_file:
        json.dump(state_dict, json_file, indent=2)


def load_training_meta(path, num):
    with open(path + str(num) + '.json', 'r') as json_file:
        state_dict = json.load(json_file)
    return state_dict


def log_info(sw, results_dict, iteration):
    with sw.as_default():
        for key in results_dict.keys():
            tf.summary.scalar(key, results_dict[key], step=iteration)


# facial alignment, taken from https://github.com/deepinsight/insightface
src1 = np.array([[51.642, 50.115], [57.617, 49.990], [35.740, 69.007],
                 [51.157, 89.050], [57.025, 89.702]],
                dtype=np.float32)
# <--left
src2 = np.array([[45.031, 50.118], [65.568, 50.872], [39.677, 68.111],
                 [45.177, 86.190], [64.246, 86.758]],
                dtype=np.float32)

# ---frontal
src3 = np.array([[39.730, 51.138], [72.270, 51.138], [56.000, 68.493],
                 [42.463, 87.010], [69.537, 87.010]],
                dtype=np.float32)

# -->right
src4 = np.array([[46.845, 50.872], [67.382, 50.118], [72.737, 68.111],
                 [48.167, 86.758], [67.236, 86.190]],
                dtype=np.float32)

# -->right profile
src5 = np.array([[54.796, 49.990], [60.771, 50.115], [76.673, 69.007],
                 [55.388, 89.702], [61.257, 89.050]],
                dtype=np.float32)

src = np.array([src1, src2, src3, src4, src5])
src_map = {112: src, 224: src * 2}

# Left eye, right eye, nose, left mouth, right mouth
arcface_src = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32)

arcface_src = np.expand_dims(arcface_src, axis=0)


def extract_face(img, bb, absolute_center, mode='arcface', extention_rate=0.05, debug=False):
    """Extract face from image given a bounding box"""
    # bbox
    x1, y1, x2, y2 = bb + 60
    adjusted_absolute_center = (absolute_center[0] + 60, absolute_center[1] + 60)
    if debug:
        print(bb + 60)
        x1, y1, x2, y2 = bb
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.circle(img, absolute_center, 1, (255, 0, 255), 2)
        Image.fromarray(img).show()
        x1, y1, x2, y2 = bb + 60
    # Pad image in case face is out of frame
    padded_img = np.zeros(shape=(248, 248, 3), dtype=np.uint8)
    padded_img[60:-60, 60:-60, :] = img

    if debug:
        cv2.rectangle(padded_img, (x1, y1), (x2, y2), (0, 255, 255), 3)
        cv2.circle(padded_img, adjusted_absolute_center, 1, (255, 255, 255), 2)
        Image.fromarray(padded_img).show()

    y_len = abs(y1 - y2)
    x_len = abs(x1 - x2)

    new_len = (y_len + x_len) // 2

    extension = int(new_len * extention_rate)

    x_adjust = (x_len - new_len) // 2
    y_adjust = (y_len - new_len) // 2

    x_1_adjusted = x1 + x_adjust - extension
    x_2_adjusted = x2 - x_adjust + extension

    if mode == 'arcface':
        y_1_adjusted = y1 - extension
        y_2_adjusted = y2 - 2 * y_adjust + extension
    else:
        y_1_adjusted = y1 + 2 * y_adjust - extension
        y_2_adjusted = y2 + extension

    move_x = adjusted_absolute_center[0] - (x_1_adjusted + x_2_adjusted) // 2
    move_y = adjusted_absolute_center[1] - (y_1_adjusted + y_2_adjusted) // 2

    x_1_adjusted = x_1_adjusted + move_x
    x_2_adjusted = x_2_adjusted + move_x
    y_1_adjusted = y_1_adjusted + move_y
    y_2_adjusted = y_2_adjusted + move_y

    # print(y_1_adjusted, y_2_adjusted, x_1_adjusted, x_2_adjusted)

    return padded_img[y_1_adjusted:y_2_adjusted, x_1_adjusted:x_2_adjusted]


def euclidean_distance(a, b):
    x1 = a[0]; y1 = a[1]
    x2 = b[0]; y2 = b[1]
    return np.sqrt(((x2 - x1) * (x2 - x1)) + ((y2 - y1) * (y2 - y1)))


def align_face(img, landmarks, debug=False):
    nose, right_eye, left_eye = landmarks

    left_eye_x = left_eye[0]
    left_eye_y = left_eye[1]

    right_eye_x = right_eye[0]
    right_eye_y = right_eye[1]

    center_eye = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)

    if left_eye_y < right_eye_y:
        point_3rd = (right_eye_x, left_eye_y)
        direction = -1
    else:
        point_3rd = (left_eye_x, right_eye_y)
        direction = 1

    if debug:
        cv2.circle(img, point_3rd, 1, (255, 0, 0), 1)
        cv2.circle(img, center_eye, 1, (255, 0, 0), 1)

        cv2.line(img, right_eye, left_eye, (0, 0, 0), 1)
        cv2.line(img, left_eye, point_3rd, (0, 0, 0), 1)
        cv2.line(img, right_eye, point_3rd, (0, 0, 0), 1)

    a = euclidean_distance(left_eye, point_3rd)
    b = euclidean_distance(right_eye, left_eye)
    c = euclidean_distance(right_eye, point_3rd)

    cos_a = (b * b + c * c - a * a) / (2 * b * c)

    angle = np.arccos(cos_a)

    angle = (angle * 180) / np.pi

    if direction == -1:
        angle = 90 - angle
        ang = math.radians(direction * angle)
    else:
        ang = math.radians(direction * angle)
        angle = 0 - angle

    M = cv2.getRotationMatrix2D((64, 64), angle, 1)
    new_img = cv2.warpAffine(img, M, (128, 128),
                            flags=cv2.INTER_CUBIC)

    rotated_nose = (int((nose[0] - 64) * np.cos(ang) - (nose[1] - 64) * np.sin(ang) + 64),
                    int((nose[0] - 64) * np.sin(ang) + (nose[1] - 64) * np.cos(ang) + 64))

    rotated_center_eye = (int((center_eye[0] - 64) * np.cos(ang) - (center_eye[1] - 64) * np.sin(ang) + 64),
                          int((center_eye[0] - 64) * np.sin(ang) + (center_eye[1] - 64) * np.cos(ang) + 64))

    abolute_center = (rotated_center_eye[0], (rotated_nose[1] + rotated_center_eye[1]) // 2)

    if debug:
        cv2.circle(new_img, rotated_nose, 1, (0, 0, 255), 1)
        cv2.circle(new_img, rotated_center_eye, 1, (0, 0, 255), 1)
        cv2.circle(new_img, abolute_center, 1, (0, 0, 255), 1)

    return new_img, abolute_center


def estimate_norm(lmk, image_size=112, mode='arcface', shrink_factor=1.0):
    assert lmk.shape == (5, 2)
    tform = trans.SimilarityTransform()
    lmk_tran = np.insert(lmk, 2, values=np.ones(5), axis=1)
    min_M = []
    min_index = []
    min_error = float('inf')
    src_factor = image_size / 112
    if mode == 'arcface':
        src = arcface_src * shrink_factor + (1 - shrink_factor) * 56
        src = src * src_factor
    else:
        src = src_map[image_size] * src_factor
    for i in np.arange(src.shape[0]):
        tform.estimate(lmk, src[i])
        M = tform.params[0:2, :]
        results = np.dot(M, lmk_tran.T)
        results = results.T
        error = np.sum(np.sqrt(np.sum((results - src[i])**2, axis=1)))
        #         print(error)
        if error < min_error:
            min_error = error
            min_M = M
            min_index = i
    return min_M, min_index


def inverse_estimate_norm(lmk, t_lmk, image_size=112, mode='arcface', shrink_factor=1.0):
    assert lmk.shape == (5, 2)
    tform = trans.SimilarityTransform()
    lmk_tran = np.insert(lmk, 2, values=np.ones(5), axis=1)
    min_M = []
    min_index = []
    min_error = float('inf')
    src_factor = image_size / 112
    if mode == 'arcface':
        src = arcface_src * shrink_factor + (1 - shrink_factor) * 56
        src = src * src_factor
    else:
        src = src_map[image_size] * src_factor
    for i in np.arange(src.shape[0]):
        tform.estimate(t_lmk, lmk)
        M = tform.params[0:2, :]
        results = np.dot(M, lmk_tran.T)
        results = results.T
        error = np.sum(np.sqrt(np.sum((results - src[i])**2, axis=1)))
        #         print(error)
        if error < min_error:
            min_error = error
            min_M = M
            min_index = i
    return min_M, min_index


def norm_crop(img, landmark, image_size=112, mode='arcface', shrink_factor=1.0):
    """
    Align and crop the image based of the facial landmarks in the image. The alignment is done with
    a similarity transformation based of source coordinates.
    :param img: Image to transform.
    :param landmark: Five landmark coordinates in the image.
    :param image_size: Desired output size after transformation.
    :param mode: 'arcface' aligns the face for the use of Arcface facial recognition model. Useful for
    both facial recognition tasks and face swapping tasks.
    :param shrink_factor: Shrink factor that shrinks the source landmark coordinates. This will include more border
    information around the face. Useful when you want to include more background information when performing face swaps.
    The lower the shrink factor the more of the face is included. Default value 1.0 will align the image to be ready
    for the Arcface recognition model, but usually omits part of the chin. Value of 0.0 would transform all source points
    to the middle of the image, probably rendering the alignment procedure useless.

    If you process the image with a shrink factor of 0.85 and then want to extract the identity embedding with arcface,
    you simply do a central crop of factor 0.85 to yield same cropped result as using shrink factor 1.0. This will
    reduce the resolution, the recommendation is to processed images to output resolutions higher than 112 is using
    Arcface. This will make sure no information is lost by resampling the image after central crop.
    :return: Returns the transformed image.
    """
    M, pose_index = estimate_norm(landmark, image_size, mode, shrink_factor=shrink_factor)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    return warped


def transform_landmark_points(M, points):
    lmk_tran = np.insert(points, 2, values=np.ones(5), axis=1)
    transformed_lmk = np.dot(M, lmk_tran.T)
    transformed_lmk = transformed_lmk.T

    return transformed_lmk


def multi_convolver(image, kernel, iterations):
    if kernel == "Sharpen":
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
    elif kernel == "Unsharp_mask":
        kernel = np.array([[1, 4, 6, 4, 1],
                           [4, 16, 24, 16, 1],
                           [6, 24, -476, 24, 1],
                           [4, 16, 24, 16, 1],
                           [1, 4, 6, 4, 1]]) * (-1 / 256)
    elif kernel == "Blur":
        kernel = (1 / 16.0) * np.array([[1., 2., 1.],
                                        [2., 4., 2.],
                                        [1., 2., 1.]])
    for i in range(iterations):
        image = convolve2d(image, kernel, 'same', boundary='fill', fillvalue = 0)
    return image


def convolve_rgb(image, kernel, iterations=1):
    img_yuv = rgb2yuv(image)
    img_yuv[:, :, 0] = multi_convolver(img_yuv[:, :, 0], kernel,
                                       iterations)
    final_image = yuv2rgb(img_yuv)

    return final_image.astype('float32')


def generate_mask_from_landmarks(lms, im_size):
    blend_mask_lm = np.zeros(shape=(im_size, im_size, 3), dtype='float32')

    # EYES
    blend_mask_lm = cv2.circle(blend_mask_lm,
                               (int(lms[0][0]), int(lms[0][1])), 12, (255, 255, 255), 30)
    blend_mask_lm = cv2.circle(blend_mask_lm,
                               (int(lms[1][0]), int(lms[1][1])), 12, (255, 255, 255), 30)
    blend_mask_lm = cv2.circle(blend_mask_lm,
                               (int((lms[0][0] + lms[1][0]) / 2), int((lms[0][1] + lms[1][1]) / 2)),
                               16, (255, 255, 255), 65)

    # NOSE
    blend_mask_lm = cv2.circle(blend_mask_lm,
                               (int(lms[2][0]), int(lms[2][1])), 5, (255, 255, 255), 5)
    blend_mask_lm = cv2.circle(blend_mask_lm,
                               (int((lms[0][0] + lms[1][0]) / 2), int(lms[2][1])), 16, (255, 255, 255), 100)

    # MOUTH
    blend_mask_lm = cv2.circle(blend_mask_lm,
                               (int(lms[3][0]), int(lms[3][1])), 6, (255, 255, 255), 30)
    blend_mask_lm = cv2.circle(blend_mask_lm,
                               (int(lms[4][0]), int(lms[4][1])), 6, (255, 255, 255), 30)

    blend_mask_lm = cv2.circle(blend_mask_lm,
                               (int((lms[3][0] + lms[4][0]) / 2), int((lms[3][1] + lms[4][1]) / 2)),
                               16, (255, 255, 255), 40)
    return blend_mask_lm


def display_distance_text(im, distance, lms, im_w, im_h, scale=2):
    blended_insert = cv2.putText(im, str(distance)[:4],
                                 (int(lms[4] * im_w * 0.5), int(lms[5] * im_h * 0.8)),
                                 cv2.FONT_HERSHEY_SIMPLEX, scale * 0.5, (0.08, 0.16, 0.08), int(scale * 2))
    blended_insert = cv2.putText(blended_insert, str(distance)[:4],
                                 (int(lms[4] * im_w * 0.5), int(lms[5] * im_h * 0.8)),
                                 cv2.FONT_HERSHEY_SIMPLEX, scale*  0.5, (0.3, 0.7, 0.32), int(scale * 1))
    return blended_insert


def get_lm(annotation, im_w, im_h):
    lm_align = np.array([[annotation[4] * im_w, annotation[5] * im_h],
                         [annotation[6] * im_w, annotation[7] * im_h],
                         [annotation[8] * im_w, annotation[9] * im_h],
                         [annotation[10] * im_w, annotation[11] * im_h],
                         [annotation[12] * im_w, annotation[13] * im_h]],
                        dtype=np.float32)
    return lm_align
