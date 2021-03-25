import os
import argparse
import numpy as np
import cv2

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras

import imgaug as ia
from imgaug import augmenters as iaa


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--tfrecord_dir", type=str,
                        default=r"F:\Kaggle\Data\Cassava_Leaf_Disease_Classification\tfrecords")
    parser.add_argument("--tfrecord_pattern", type=str, default="fold_1")
    parser.add_argument("--model_name", type=str, default="efficientnet_b7")
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--checkpoint_file", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default="tensorboard")
    parser.add_argument("--min_learning_rate", type=float, default=1e-6)
    parser.add_argument("--max_learning_rate", type=float, default=1e-2)
    parser.add_argument("--step_size", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--first_time_weight", type=str, default="imagenet")
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--num_train_elements", type=int, default=17115)
    parser.add_argument("--num_val_elements", type=int, default=4282)
    parser.add_argument("--target_height", type=int, default=300)
    parser.add_argument("--target_width", type=int, default=300)

    args = parser.parse_args()

    return args


sometimes = lambda aug: iaa.Sometimes(0.8, aug)

_augment_seq = iaa.Sequential(
    [
        # apply the following augmenters to most images
        iaa.Fliplr(0.5), # horizontally flip 50% of all images
        iaa.Flipud(0.5), # vertically flip 20% of all images
        # crop images by -5% to 10% of their height/width
        sometimes(iaa.CropAndPad(
                    percent=(-0.2, 0.2),
                    pad_mode=ia.ALL,
                    pad_cval=(0, 255)
                )),
        sometimes(iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
            rotate=(-45, 45), # rotate by -45 to +45 degrees
            shear=(-16, 16), # shear by -16 to +16 degrees
            order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
            cval=(0, 255), # if mode is constant, use a cval between 0 and 255
            mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        )),
        # execute 0 to 5 of the following (less important) augmenters per image
        # don't execute all of them, as that would often be way too strong
        iaa.SomeOf((2, 7),
            [
                #sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
                iaa.OneOf([
                    iaa.GaussianBlur((0, 1.0)), # blur images with a sigma between 0 and 3.0
                    iaa.AverageBlur(k=3), # blur image using local means with kernel sizes between 2 and 7
                    iaa.MedianBlur(k=3), # blur image using local medians with kernel sizes between 2 and 7
                ]),
                #iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
                #iaa.Emboss(alpha=(0, 0.5), strength=(0, 0.5)), # emboss images
                # search either for all edges or for directed edges,
                # blend the result with the original image using a blobby mask
                #iaa.SimplexNoiseAlpha(iaa.OneOf([
                #    iaa.EdgeDetect(alpha=(0.5, 1.0)),
                #    iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                #])),
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.02*255), per_channel=0.5), # add gaussian noise to images
                #iaa.OneOf([
                #    iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
                #    iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                #]),
                #iaa.CoarseDropout((0.03, 0.15), size_percent=(0.1, 0.2), per_channel=0.2),
                #iaa.Invert(0.05, per_channel=True), # invert color channels
                iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                #iaa.AddToHueAndSaturation((-20, 20)), # change hue and saturation
                # either change the brightness of the whole image (sometimes
                # per channel) or change the brightness of subareas
                iaa.OneOf([
                    iaa.Multiply((0.5, 1.5), per_channel=0.5),
                    iaa.FrequencyNoiseAlpha(
                        exponent=(-4, 0),
                        first=iaa.Multiply((0.5, 1.5), per_channel=True),
                        second=iaa.LinearContrast((0.5, 2.0))
                    )
                ]),
                #iaa.LinearContrast((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast
                iaa.Grayscale(alpha=(0.0, 1.0)),
                #sometimes(iaa.ElasticTransformation(alpha=(2.0, 10.0), sigma=0.25)), # move pixels locally around (with random strengths) # Nên bỏ đi
                sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))), # sometimes move parts of the image around
                sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1))),
                #sometimes(iaa.Jigsaw(nb_rows=(3, 5), nb_cols=(3, 5)))
            ],
            random_order=True
        )
    ],
    random_order=True
)


def _augment_fn(image: tf.Tensor, label: tf.Tensor):
    image = tf.numpy_function(_augment_seq.augment_images,
                              [image],
                              image.dtype)
    image = tf.cast(image, dtype=tf.float32)
    return image, label

keys_to_features = {'image': tf.io.FixedLenFeature([], tf.string),
                    'label': tf.io.FixedLenFeature([], tf.int64),
                    'width': tf.io.FixedLenFeature([], tf.int64),
                    'height': tf.io.FixedLenFeature([], tf.int64),
                    'channels': tf.io.FixedLenFeature([], tf.int64)}


def _parse_fn(data_record, target_height=300, target_width=300, num_classes=2):
  features = keys_to_features
  sample = tf.io.parse_single_example(data_record, features)
  image = tf.image.decode_jpeg(sample["image"])
  image = tf.cast(image, dtype=tf.uint8)
  height = sample["height"]
  width = sample["width"]
  image.set_shape(shape=[height, width, 3])
  image = tf.image.resize(image, size=[target_height, target_width], method=tf.image.ResizeMethod.BICUBIC, antialias=True)
  image = tf.cast(image, dtype=tf.uint8)
  label = sample["label"]
  #label = tf.one_hot(label, num_classes, 1)

  return image, label


def get_dataset(tfrecord_path, batch_size, target_height=300, target_width=300, is_training=True):
    dataset = tf.data.TFRecordDataset([tfrecord_path])
    dataset = dataset.map(lambda x: _parse_fn(x, target_height=target_height, target_width=target_width),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if is_training:
        #dataset = dataset.flat_map(lambda image, label: tf.data.Dataset.from_tensors((image, label)).repeat(
        #    oversample_classes(image, label, num_classes=5)))
        #dataset = dataset.filter(undersampling_filter)
        dataset = dataset.shuffle(512)
        dataset = dataset.repeat()
    dataset = dataset.batch(batch_size, drop_remainder=True)
    if is_training:
        dataset = dataset.map(_augment_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    #dataset = dataset.unbatch()
    #dataset = dataset.repeat(num_epochs)

    return dataset


def get_model(args):
    inputs = keras.Input(shape=(args.target_height, args.target_width, 3))
    if args.model_name.lower() == "efficientnet_b0":
        model = keras.applications.EfficientNetB0(include_top=False, weights=None, input_shape=(args.target_height, args.target_width, 3),
                                                  pooling="avg")
        x = model(inputs)
        outputs = keras.layers.Dense(args.num_classes)(x)

        return keras.Model(inputs, outputs)
    elif args.model_name.lower() == "efficientnet_b1":
        model = keras.applications.EfficientNetB1(include_top=False, weights=None, input_shape=(args.target_height, args.target_width, 3),
                                                  pooling="avg")
        x = model(inputs)
        outputs = keras.layers.Dense(args.num_classes)(x)

        return keras.Model(inputs, outputs)
    elif args.model_name.lower() == "efficientnet_b2":
        model = keras.applications.EfficientNetB2(include_top=False, weights=None, input_shape=(args.target_height, args.target_width, 3),
                                                  pooling="avg")
        x = model(inputs)
        outputs = keras.layers.Dense(args.num_classes)(x)

        return keras.Model(inputs, outputs)
    elif args.model_name.lower() == "efficientnet_b3":
        model = keras.applications.EfficientNetB3(include_top=False, weights=None, input_shape=(args.target_height, args.target_width, 3),
                                                  pooling="avg")
        x = model(inputs)
        outputs = keras.layers.Dense(args.num_classes)(x)

        return keras.Model(inputs, outputs)
    elif args.model_name.lower() == "efficientnet_b4":
        model = keras.applications.EfficientNetB4(include_top=False, weights=None, input_shape=(args.target_height, args.target_width, 3),
                                                  pooling="avg")
        x = model(inputs)
        outputs = keras.layers.Dense(args.num_classes)(x)

        return keras.Model(inputs, outputs)
    elif args.model_name.lower() == "efficientnet_b5":
        model = keras.applications.EfficientNetB5(include_top=False, weights=None, input_shape=(args.target_height, args.target_width, 3),
                                                  pooling="avg")
        x = model(inputs)
        outputs = keras.layers.Dense(args.num_classes)(x)

        return keras.Model(inputs, outputs)
    elif args.model_name.lower() == "efficientnet_b6":
        model = keras.applications.EfficientNetB6(include_top=False, weights=None, input_shape=(args.target_height, args.target_width, 3),
                                                  pooling="avg")
        x = model(inputs)
        outputs = keras.layers.Dense(args.num_classes)(x)

        return keras.Model(inputs, outputs)
    elif args.model_name.lower() == "efficientnet_b7":
        model = keras.applications.EfficientNetB7(include_top=False, weights=None, input_shape=(args.target_height, args.target_width, 3),
                                                  pooling="avg")
        x = model(inputs)
        outputs = keras.layers.Dense(args.num_classes)(x)

        return keras.Model(inputs, outputs)
    elif args.model_name.lower() == "densenet_121":
        model = keras.applications.DenseNet121(include_top=False, weights=None, input_shape=(args.target_height, args.target_width, 3),
                                               pooling="avg")
        x = model(inputs)
        outputs = keras.layers.Dense(args.num_classes)(x)

        return keras.Model(inputs, outputs)
    elif args.model_name.lower() == "densenet_169":
        model = keras.applications.DenseNet169(include_top=False, weights=None, input_shape=(args.target_height, args.target_width, 3),
                                               pooling="avg")
        x = model(inputs)
        outputs = keras.layers.Dense(args.num_classes)(x)

        return keras.Model(inputs, outputs)
    elif args.model_name.lower() == "densenet_201":
        model = keras.applications.DenseNet201(include_top=False, weights=None, input_shape=(args.target_height, args.target_width, 3),
                                               pooling="avg")
        x = model(inputs)
        outputs = keras.layers.Dense(args.num_classes)(x)

        return keras.Model(inputs, outputs)
    elif args.model_name.lower() == "xception":
        model = keras.applications.Xception(include_top=False, weights=None, input_shape=(args.target_height, args.target_width, 3),
                                            pooling="avg")
        x = model(inputs)
        outputs = keras.layers.Dense(args.num_classes)(x)

        return keras.Model(inputs, outputs)
    elif args.model_name.lower() == "resnet50":
        model = keras.applications.ResNet50(include_top=False, weights=None, input_shape=(args.target_height, args.target_width, 3),
                                            pooling="avg")
        x = model(inputs)
        outputs = keras.layers.Dense(args.num_classes)(x)

        return keras.Model(inputs, outputs)
    else:
        raise ValueError("Model {} not in chosen models".format(args.model_name))


def get_optimizer(args):
    learning_rate_schedule = tfa.optimizers.ExponentialCyclicalLearningRate(initial_learning_rate=args.min_learning_rate,
                                                                            maximal_learning_rate=args.max_learning_rate,
                                                                            step_size=args.step_size,
                                                                            gamma=0.9)
    if args.optimizer.lower() == "adam":
        return keras.optimizers.Adam(learning_rate=learning_rate_schedule)
    elif args.optimizer.lower() == "rms" or args.optimizer.lower()  == "rmsprop":
        return keras.optimizers.RMSprop(learning_rate=learning_rate_schedule, momentum=0.5)
    else:
        raise ValueError("Optimizer {} not in chosen optimizers".format(args.optimizer))


if __name__ == '__main__':
    args = get_args()

    print("Arguments: {}".format(args))

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    try:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu="grpc://" + os.environ["COLAB_TPU_ADDR"])
        print("Running on TPU", resolver.cluster_spec().as_dict()["worker"])
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        print("All devices: ", tf.config.list_logical_devices("TPU"))
        strategy = tf.distribute.experimental.TPUStrategy(resolver)
    except:
        print("No TPU, run on GPU or CPU")
        if tf.config.list_physical_devices("GPU"):
            strategy = tf.distribute.MirroredStrategy()
            print("Running on GPU")
        else:
            strategy = tf.distribute.get_strategy()
            print("Running on CPU")

    batch_size = args.batch_size * strategy.num_replicas_in_sync
    print("Batch size: {}".format(batch_size))

    callbacks = [keras.callbacks.TensorBoard(log_dir=args.log_dir,
                                             write_graph=False if args.checkpoint_file else True,
                                             write_images=True,
                                             update_freq=100,
                                             profile_batch=5),
                 keras.callbacks.EarlyStopping(monitor="val_sparse_categorical_accuracy", min_delta=0.001,
                                               patience=5, verbose=1, mode="max",
                                               restore_best_weights=True)]

    with strategy.scope():
        model = get_model(args)
        model.summary()
        if args.checkpoint_file is not None:
            model.load_weights(filepath=os.path.join(args.checkpoint_dir, args.checkpoint_file))
        optimizer = get_optimizer(args)
        model.compile(optimizer=optimizer,
                      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=[keras.metrics.SparseCategoricalAccuracy()])

    train_tfrecord_name = os.path.join(args.tfrecord_dir, "{}_training_nsfw.tfrecord".format(args.tfrecord_pattern))
    if args.num_train_elements:
        num_train_elements = args.num_train_elements
    else:
        num_train_elements = sum(1 for _ in tf.data.TFRecordDataset(train_tfrecord_name))
    print("num_train_elements: {}".format(num_train_elements))
    num_train_steps = num_train_elements // batch_size
    val_tfrecord_name = os.path.join(args.tfrecord_dir, "{}_validation_nsfw.tfrecord".format(args.tfrecord_pattern))
    if args.num_val_elements:
        num_val_elements = args.num_val_elements
    else:
        num_val_elements = sum(1 for _ in tf.data.TFRecordDataset(val_tfrecord_name))
    print("num_val_elements: {}".format(num_val_elements))
    num_val_steps = num_val_elements // batch_size

    train_dataset = get_dataset(tfrecord_path=train_tfrecord_name,
                                batch_size=batch_size, is_training=True,
                                target_height=args.target_height, target_width=args.target_width)

    val_dataset = get_dataset(tfrecord_path=val_tfrecord_name,
                              batch_size=batch_size, is_training=False,
                              target_height=args.target_height, target_width=args.target_width)

    model.fit(train_dataset, steps_per_epoch=num_train_steps, callbacks=callbacks, validation_data=val_dataset,
              epochs=args.num_epochs, validation_steps=num_val_steps)

    result = model.evaluate(val_dataset, steps=num_val_steps)

    print("Result: {}".format(dict(zip(model.metrics_names, result))))

    model.save_weights(
        filepath=os.path.join(args.checkpoint_dir, "model_{}.h5".format(datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))),
        save_format="h5")

    print("Save weights and exit")
