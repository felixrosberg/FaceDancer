from retinaface.anchor import decode_tf, prior_box_tf
import tensorflow as tf


def extract_detections(bbox_regressions, landm_regressions, classifications, image_sizes, iou_th=0.4, score_th=0.02):
    min_sizes = [[16, 32], [64, 128], [256, 512]]
    steps = [8, 16, 32]
    variances = [0.1, 0.2]
    preds = tf.concat(  # [bboxes, landms, landms_valid, conf]
        [bbox_regressions,
         landm_regressions,
         tf.ones_like(classifications[:, 0][..., tf.newaxis]),
         classifications[:, 1][..., tf.newaxis]], 1)
    priors = prior_box_tf(image_sizes, min_sizes, steps, False)
    decode_preds = decode_tf(preds, priors, variances)

    selected_indices = tf.image.non_max_suppression(
        boxes=decode_preds[:, :4],
        scores=decode_preds[:, -1],
        max_output_size=tf.shape(decode_preds)[0],
        iou_threshold=iou_th,
        score_threshold=score_th)

    out = tf.gather(decode_preds, selected_indices)

    return out

