import os
import argparse

import numpy as np
import cv2

import tensorflow as tf
from tensorflow import keras


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--target_size", type=int, default=300)
    parser.add_argument("--num_classes", type=int, default=2)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()

    CLASSES = {0: "Negative", 1: "Positive"}

    img = cv2.imread(args.image_path)
    rgb_img = img[:, :, ::-1].copy()

    inputs = keras.Input(shape=(args.target_size, args.target_size, 3))

    pretrained_model = keras.applications.EfficientNetB3(input_shape=(args.target_size, args.target_size, 3),
                                                         include_top=False, weights=None)
    # pretrained_model = keras.applications.ResNet50(input_shape=(TARGET_HEIGHT, TARGET_WIDTH, 3), include_top=False, weights=None)
    x = pretrained_model(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    outputs = keras.layers.Dense(args.num_classes)(x)
    outputs = tf.nn.softmax(outputs)
    model = keras.Model(inputs, outputs)

    model.summary()
    model.load_weights(args.checkpoint, by_name=True)

    resized_img = cv2.resize(img, (args.target_size, args.target_size))

    pred = model.predict(np.expand_dims(resized_img, 0))

    pred = list(map(lambda x: dict(zip(CLASSES.values(), x)), pred))

    print("Prediction: {}".format(pred))

    cv2.imshow("Input", img)
    cv2.waitKey(0)
