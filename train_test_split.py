import os
import argparse
import itertools
from operator import itemgetter

from tqdm import tqdm

import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold

import cv2

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

import tfrecord_utils

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--positive_dir", type=str, default=None)
    parser.add_argument("--negative_dir", type=str, default=None)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--save_dir", type=str, default=None)

    args = parser.parse_args()

    return args


def enumerate_images(dir):
    images_list = []
    for dirs, _, images in os.walk(dir):
        if "unsup" in dirs:
            continue
        for image in images:
            images_list.append(os.path.join(dirs, image))

    return images_list


if __name__ == '__main__':
    args = get_args()

    positive_dir = args.positive_dir
    negative_dir = args.negative_dir
    n_splits = args.n_splits
    save_dir = args.save_dir

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    negative_images_list = enumerate_images(negative_dir)
    negative_list = list(map(lambda x: (x, 0), negative_images_list))
    print("negative list: {}".format(len(negative_list)))

    positive_images_list = enumerate_images(positive_dir)
    positive_list = list(map(lambda x: (x, 1), positive_images_list))
    print("positive list: {}".format(len(positive_list)))

    records = negative_list + positive_list
    images = list(map(lambda x: x[0], records))
    labels = list(map(lambda x: x[1], records))

    fold_splitter = StratifiedKFold(n_splits=n_splits, shuffle=True)
    folds = fold_splitter.split(images, labels)

    sess = tf.Session()
    image_phl = tf.placeholder(shape=[None, None, None], dtype=tf.uint8)
    jpg_encoded = tf.image.encode_jpeg(image_phl, quality=100)

    for i, (train_indices, test_indices) in enumerate(folds):
        train_fold = itemgetter(*train_indices)(records)
        val_fold = itemgetter(*test_indices)(records)

        print("Training")

        for key, group in itertools.groupby(sorted(train_fold, key=lambda x: x[1]), lambda x: x[1]):
            print("{}: {}".format(key, len(list(group)) / len(train_fold)))

        with tf.python_io.TFRecordWriter(os.path.join(save_dir, "fold_{}_training_nsfw.tfrecord".format(i + 1))) as training_tfrecord_writer:
            for element in tqdm(train_fold):
                image, label = element

                img = cv2.imread(image)
                img = img[:, :, ::-1]

                height, width, channels = img.shape

                jpg_string = sess.run(jpg_encoded, feed_dict={image_phl: img})

                example = tfrecord_utils.image_to_tfexample(image_data=jpg_string,
                                                            label=label,
                                                            width=width,
                                                            height=height,
                                                            channels=channels)

                training_tfrecord_writer.write(example.SerializeToString())

        print("Validation")

        for key, group in itertools.groupby(sorted(val_fold, key=lambda x: x[1]), lambda x: x[1]):
            print("{}: {}".format(key, len(list(group)) / len(val_fold)))

        with tf.python_io.TFRecordWriter(os.path.join(save_dir, "fold_{}_validation_nsfw.tfrecord".format(i + 1))) as validation_tfrecord_writer:
            for element in tqdm(val_fold):
                image, label = element

                img = cv2.imread(image)
                img = img[:, :, ::-1]

                height, width, channels = img.shape

                jpg_string = sess.run(jpg_encoded, feed_dict={image_phl: img})

                example = tfrecord_utils.image_to_tfexample(image_data=jpg_string,
                                                            label=label,
                                                            width=width,
                                                            height=height,
                                                            channels=channels)

                validation_tfrecord_writer.write(example.SerializeToString())
