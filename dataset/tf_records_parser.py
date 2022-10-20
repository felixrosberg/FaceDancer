import tensorflow as tf


image_feature_description = {
    'face': tf.io.FixedLenFeature([], tf.string),
    }


def get_tf_dataset(tfrecords_paths, im_size=256, batchsize=10, repeat=False):
    def decode_img(img):
        img = tf.image.decode_png(img, channels=3)
        return (tf.image.resize(img, [im_size, im_size]) - 127.5) / 127.5

    def _parse_image_function(example_proto):
        data_dict = tf.io.parse_single_example(example_proto, image_feature_description)
        face = decode_img(data_dict['face'])
        return face

    files_targets = tf.io.matching_files(tfrecords_paths).numpy()
    files_target = tf.random.shuffle(files_targets)
    shards_target = tf.data.Dataset.from_tensor_slices(files_target)
    dataset_target = shards_target.interleave(tf.data.TFRecordDataset)
    dataset_target = dataset_target.shuffle(buffer_size=1000)
    dataset_target = dataset_target.map(map_func=_parse_image_function,
                                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset_target = dataset_target.batch(batchsize, drop_remainder=True)

    files_sources = tf.io.matching_files(tfrecords_paths).numpy()
    files_source = tf.random.shuffle(files_sources)
    shards_source = tf.data.Dataset.from_tensor_slices(files_source)
    dataset_source = shards_source.interleave(tf.data.TFRecordDataset)
    dataset_source = dataset_source.shuffle(buffer_size=1000)
    dataset_source = dataset_source.map(map_func=_parse_image_function,
                                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset_source = dataset_source.batch(batchsize, drop_remainder=True)

    dataset_total = tf.data.Dataset.zip((dataset_target, dataset_source))
    dataset_total.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    if repeat:
        dataset_total = dataset_total.repeat()

    return dataset_total
