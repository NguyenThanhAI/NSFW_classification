import tensorflow as tf


def _bytes_feature(value):
    if not isinstance(value, list):
        value = [value]

    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _int64_feature(value):
    if  not isinstance(value, list):
        value = [value]

    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def image_to_tfexample(image_data, label, width, height, channels):
    return tf.train.Example(features=tf.train.Features(feature={'image': _bytes_feature(image_data),
                                                                'label': _int64_feature(label),
                                                                'width': _int64_feature(width),
                                                                'height': _int64_feature(height),
                                                                'channels': _int64_feature(channels)}))
