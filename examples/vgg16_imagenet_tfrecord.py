import os, pathlib
import tensorflow as tf
from tensorflow.keras import (models, layers, datasets, callbacks, optimizers,
                              initializers, regularizers)
from tensorflow.keras.utils import CustomObjectScope
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from ml_genn import Model
from ml_genn.layers import InputType
from ml_genn.norm import DataNorm, SpikeNorm
from ml_genn.utils import parse_arguments, raster_plot
from glob import glob
import numpy as np

# Learning rate schedule
def schedule(epoch, learning_rate):
    if epoch < 30:
        return 0.05
    elif epoch < 60:
        return 0.005
    else:
        return 0.0005

def initializer(shape, dtype=None):
    stddev = np.sqrt(2.0 / float(shape[0] * shape[1] * shape[3]))
    return tf.random.normal(shape, dtype=dtype, stddev=stddev)

if __name__ == '__main__':
    args = parse_arguments('VGG16 ImageNet classifier model')
    print('arguments: ' + str(vars(args)))

    # for gpu in tf.config.experimental.list_physical_devices('GPU'):
    #     tf.config.experimental.set_memory_growth(gpu, True)

    # READING INPUT FILES
    # ===================

    #train_root = pathlib.Path('/mnt/data0/train')
    train_root = pathlib.Path('/mnt/data0/validation')
    checkpoint_root = pathlib.Path('./training_checkpoints')

    def parse_buffer(buffer):
        keys_to_features = {
            "image/encoded": tf.io.FixedLenFeature((), tf.string, ''),
            "image/class/label": tf.io.FixedLenFeature([], tf.int64, -1)}
        parsed = tf.io.parse_single_example(buffer, keys_to_features)

        # get label
        label = tf.cast(tf.reshape(parsed["image/class/label"], shape=[]), dtype=tf.int32) - 1

        # decode image
        image = tf.image.decode_jpeg(tf.reshape(parsed["image/encoded"], shape=[]), channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)

        # resize small
        shape = tf.math.maximum(tf.shape(image)[0:2], 224)
        image = tf.image.resize(image, shape)

        # random crop and horizontal flip
        image = tf.image.random_crop(image, [224, 224, 3])
        image = tf.image.random_flip_left_right(image)

        # zero mean colours
        image = image - [0.485, 0.456, 0.406]

        return image, label

    def fetch(path):
        return tf.data.TFRecordDataset(path, buffer_size=8*1024*1024)

    #shard_ds = tf.data.Dataset.list_files(str(train_root/'train*'))
    shard_ds = tf.data.Dataset.list_files(str(train_root/'validation*'))
    buffer_ds = shard_ds.interleave(fetch, cycle_length=64)
    image_ds = buffer_ds.map(parse_buffer, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    image_ds = image_ds.shuffle(10000)
    image_ds = image_ds.batch(128)
    image_ds = image_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    # Create L2 regularizer
    regularizer = regularizers.l2(0.0001)

    # Create, train and evaluate TensorFlow model
    tf_model = models.Sequential([
        layers.Conv2D(64, 3, padding='same', activation='relu', use_bias=False, input_shape=(224, 224, 3),
                      kernel_initializer=initializer, kernel_regularizer=regularizer),
        layers.Dropout(0.3),
        layers.Conv2D(64, 3, padding='same', activation='relu', use_bias=False,
                      kernel_initializer=initializer, kernel_regularizer=regularizer),
        layers.AveragePooling2D(2),

        layers.Conv2D(128, 3, padding="same", activation="relu", use_bias=False,
                      kernel_initializer=initializer, kernel_regularizer=regularizer),
        layers.Dropout(0.4),
        layers.Conv2D(128, 3, padding="same", activation="relu", use_bias=False,
                      kernel_initializer=initializer, kernel_regularizer=regularizer),
        layers.AveragePooling2D(2),

        layers.Conv2D(256, 3, padding="same", activation="relu", use_bias=False,
                      kernel_initializer=initializer, kernel_regularizer=regularizer),
        layers.Dropout(0.4),
        layers.Conv2D(256, 3, padding="same", activation="relu", use_bias=False,
                      kernel_initializer=initializer, kernel_regularizer=regularizer),
        layers.Dropout(0.4),
        layers.Conv2D(256, 3, padding="same", activation="relu", use_bias=False,
                      kernel_initializer=initializer, kernel_regularizer=regularizer),
        layers.AveragePooling2D(2),

        layers.Conv2D(512, 3, padding="same", activation="relu", use_bias=False,
                      kernel_initializer=initializer, kernel_regularizer=regularizer),
        layers.Dropout(0.4),
        layers.Conv2D(512, 3, padding="same", activation="relu", use_bias=False,
                      kernel_initializer=initializer, kernel_regularizer=regularizer),
        layers.Dropout(0.4),
        layers.Conv2D(512, 3, padding="same", activation="relu", use_bias=False,
                      kernel_initializer=initializer, kernel_regularizer=regularizer),
        layers.AveragePooling2D(2),

        layers.Conv2D(512, 3, padding="same", activation="relu", use_bias=False,
                      kernel_initializer=initializer, kernel_regularizer=regularizer),
        layers.Dropout(0.4),
        layers.Conv2D(512, 3, padding="same", activation="relu", use_bias=False,
                      kernel_initializer=initializer, kernel_regularizer=regularizer),
        layers.Dropout(0.4),
        layers.Conv2D(512, 3, padding="same", activation="relu", use_bias=False,
                      kernel_initializer=initializer, kernel_regularizer=regularizer),
        layers.AveragePooling2D(2),

        layers.Flatten(),
        layers.Dense(4096, activation="relu", use_bias=False, kernel_regularizer=regularizer),
        layers.Dropout(0.5),
        layers.Dense(4096, activation="relu", use_bias=False, kernel_regularizer=regularizer),
        layers.Dropout(0.5),
        layers.Dense(1000, activation="softmax", use_bias=False, kernel_regularizer=regularizer),
    ], name='vgg16_imagenet')


    if args.reuse_tf_model:
        with CustomObjectScope({'initializer': initializer}):
            tf_model = models.load_model('training_checkpoints/checkpoint-87.hdf5')

    else:
        initial_epoch = 0

        # If there are any existing checkpoints
        existing_checkpoints = list(sorted(glob(str(checkpoint_root / 'checkpoint-*.hdf5'))))
        if len(existing_checkpoints) > 0:
            # Load model from newest ceckpoint
            newest_checkpoint_file = existing_checkpoints[-1]




        callbacks = [callbacks.LearningRateScheduler(schedule),
                     callbacks.ModelCheckpoint(checkpoint_root)]
        if args.record_tensorboard:
            callbacks.append(callbacks.TensorBoard(log_dir="logs", histogram_freq=1, profile_batch=(4,8)))

        optimizer = optimizers.SGD(lr=0.05, momentum=0.9)

        #tf_model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        #tf_model.fit(image_ds, epochs=100, callbacks=callbacks)

        #models.save_model(tf_model, 'vgg16_imagenet_tf_model', save_format='h5')



    tf_model.evaluate(image_ds)


    exit(0)





    # Create, normalise and evaluate ML GeNN model
    mlg_model = Model.convert_tf_model(tf_model, input_type=args.input_type, connectivity_type=args.connectivity_type)
    mlg_model.compile(dt=args.dt, batch_size=args.batch_size, rng_seed=args.rng_seed)

    if args.norm_method == 'data-norm':
        norm = DataNorm([x_norm], tf_model)
        norm.normalize(mlg_model)
    elif args.norm_method == 'spike-norm':
        norm = SpikeNorm([x_norm])
        norm.normalize(mlg_model, 2500)

    acc, spk_i, spk_t = mlg_model.evaluate([x_test], [y_test], 2500, save_samples=args.save_samples)

    # Report ML GeNN model results
    print('Accuracy of VGG16 ImageNet GeNN model: {}%'.format(acc[0]))
    if args.plot:
        names = ['input_nrn'] + [name + '_nrn' for name in mlg_model.layer_names]
        neurons = [mlg_model.g_model.neuron_populations[name] for name in names]
        raster_plot(spk_i, spk_t, neurons)
