import argparse
import numpy as np
import os
import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler

from utils import get_keypoints, parse_tfrecord_fn, TEMPLATE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str)
    parser.add_argument("--val_dir", type=str)
    parser.add_argument("--test_dir", type=str)
    args = parser.parse_args()

    train_dir = args.train_dir
    val_dir = args.val_dir
    test_dir = args.test_dir

    IMAGE_SIZE = 224
    BATCH_SIZE = 32
    VAL_SPLIT = 0.11
    MAX_SHIFT = 20
    MAX_BRIGHTNESS_DELTA = 0.2
    MIN_CONTRAST = 0.5
    MAX_CONTRAST = 2
    MAX_HUE_DELTA = 0.1
    MIN_QUALITY = 25
    MAX_QUALITY = 100
    MIN_SATURATION = 0.3
    MAX_SATURATION = 1.7

    autotune = tf.data.AUTOTUNE
    train_records = [os.path.join(train_dir, file) for file in os.listdir(train_dir) if os.path.splitext(file)[1] == '.tfrec']
    val_records = [os.path.join(val_dir, file) for file in os.listdir(val_dir) if os.path.splitext(file)[1] == '.tfrec']
    test_records = [os.path.join(test_dir, file) for file in os.listdir(test_dir) if os.path.splitext(file)[1] == '.tfrec']

    raw_train_dataset = tf.data.TFRecordDataset(train_records)
    raw_val_dataset = tf.data.TFRecordDataset(val_records)
    raw_test_dataset = tf.data.TFRecordDataset(test_records)

    train_dataset = raw_train_dataset.map(parse_tfrecord_fn, num_parallel_calls=autotune)
    val_dataset = raw_val_dataset.map(parse_tfrecord_fn, num_parallel_calls=autotune)
    test_dataset = raw_test_dataset.map(parse_tfrecord_fn, num_parallel_calls=autotune)

    # Resize
    train_dataset = train_dataset.map(lambda x, y: resize_images_keypoints(x, y, IMAGE_SIZE), num_parallel_calls=autotune)
    val_dataset = val_dataset.map(lambda x, y: resize_images_keypoints(x, y, IMAGE_SIZE), num_parallel_calls=autotune)
    test_dataset = test_dataset.map(lambda x, y: resize_images_keypoints(x, y, IMAGE_SIZE), num_parallel_calls=autotune)


    #  Multiply dataset by 2^n.
    n = 6
    for i in range(n):
        train_dataset = train_dataset.concatenate(train_dataset)

    # Augmentations
    train_dataset = train_dataset.map(flip_images_keypoints, num_parallel_calls=autotune)
    train_dataset = train_dataset.map(rotate_images_keypoints, num_parallel_calls=autotune)
    train_dataset = train_dataset.map(lambda x, y: shift_images_keypoints(x, y, MAX_SHIFT))
    train_dataset = train_dataset.map(lambda x, y: (tf.image.random_brightness(x, MAX_BRIGHTNESS_DELTA), y))
    train_dataset = train_dataset.map(lambda x, y: (tf.image.random_contrast(x, MIN_CONTRAST, MAX_CONTRAST), y))
    train_dataset = train_dataset.map(lambda x, y: (tf.image.random_hue(x, MAX_HUE_DELTA), y))
    train_dataset = train_dataset.map(lambda x, y: (tf.image.random_jpeg_quality(x, MIN_QUALITY, MAX_QUALITY), y))
    train_dataset = train_dataset.map(lambda x, y: (tf.image.random_saturation(x, MIN_SATURATION, MAX_SATURATION), y))

    train_dataset = train_dataset.shuffle(8*BATCH_SIZE)
    train_dataset = train_dataset.batch(BATCH_SIZE)
    train_dataset = train_dataset.prefetch(autotune)

    val_dataset = val_dataset.batch(BATCH_SIZE)
    val_dataset = val_dataset.prefetch(autotune)

    test_dataset = test_dataset.batch(BATCH_SIZE)
    test_dataset = test_dataset.prefetch(autotune)

    number_of_steps = train_dataset.cardinality().numpy()
    if number_of_steps == -2:
        number_of_steps = len(list(iter(train_dataset)))
    
    # Define callbacks
    earlystop = EarlyStopping(monitor = 'val_loss', 
                patience = 64,
                mode = 'min',
                baseline=None,
                restore_best_weights=True)
    
    base_model = keras.applications.MobileNetV2(
        include_top=False, pooling='avg', weights='imagenet', input_shape=(224, 224, 3)
    )

    # Want the last layer to be dense layer with 34 nodes
    inputs = keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name='input')
    x = base_model(inputs)
    x = keras.layers.Dropout(0.1)(x)
    outputs = keras.layers.Dense(34)(x)
    model = keras.Model(inputs, outputs)

    # mobilenetv2 training.
    LEARNING_RATE = 0.0001
    warmup = keras.optimizers.schedules.PiecewiseConstantDecay([10*number_of_steps, 20*number_of_steps], [LEARNING_RATE/100, LEARNING_RATE/10, LEARNING_RATE])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=warmup), 
        loss='mean_absolute_error',
        metrics=['mse'],
        )

    history_high_lr=model.fit(train_dataset,
                    epochs=1000,
                    validation_data=val_dataset,
                    callbacks=[
                        earlystop,
                        tf.keras.callbacks.ModelCheckpoint(
                            filepath=os.path.join("high_lr", "mobilenetv2"),
                            monitor="val_loss",
                            save_best_only=True,
                            save_weights_only=True,
                            verbose=1,
                        )
                    ]
                    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE/10),
        loss='mean_absolute_error',
        metrics=['mse'],
        )

    history_low_lr=model.fit(train_dataset,
                    epochs=1000,
                    validation_data=val_dataset,
                    callbacks=[
                        earlystop,
                        tf.keras.callbacks.ModelCheckpoint(
                            filepath=os.path.join("low_lr", "mobilenetv2"),
                            monitor="val_loss",
                            save_best_only=True,
                            save_weights_only=True,
                            verbose=1,
                        )
                    ]
                    )
    
def resize_images_keypoints(image, keypoints, size=256):
        resized_img = tf.image.resize(image, (size, size), method='nearest')
        x, y = keypoints[::2], keypoints[1::2]
        resized_trgts = [1, 2, 3, 4]
        image_shape = tf.cast(tf.shape(image)[:2], dtype=tf.int32)
        resized_x = tf.multiply(x, tf.cast(tf.divide(size, image_shape[1]), tf.float32))
        resized_y = tf.multiply(y, tf.cast(tf.divide(size, image_shape[0]), tf.float32))
        
    
        resized_trgts = tf.reshape(tf.stack((resized_x, resized_y), axis=-1), shape=(-1, 34))[0]
        return resized_img, resized_trgts

def flip_images_keypoints(image, keypoints):
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        image_shape = tf.cast(tf.shape(image)[:2], tf.float32)
        keypoints = tf.reshape(tf.stack([image_shape[1] - keypoints[::2], keypoints[1::2]], axis=-1), shape=(-1,34))[0]
    return image, keypoints

def rotate_images_keypoints(images, keypoints):
    keypoints = keypoints

    angle = tf.random.uniform((), maxval= 359)
    radians = angle * (np.pi/180)
    
    random_number = tf.random.uniform(())
    if  0 <= random_number and random_number < 0.33333:
        image = tfa.image.rotate(images, radians, fill_mode='wrap')
    elif 0.33333 <= random_number and random_number < 0.66666:
        image = tfa.image.rotate(images, radians, fill_mode="reflect")
    else:
        image = tfa.image.rotate(images, radians)

    image_shape = tf.shape(image)[:2]

    radians = angle * (np.pi/180)
    angle_cos, angle_sin = tf.math.cos(radians), -tf.math.sin(radians)
    half_width = tf.cast(image_shape[1]/2, dtype=tf.float32)
    half_height = tf.cast(image_shape[0]/2, dtype=tf.float32)

    x = keypoints[::2] - half_width
    y = keypoints[1::2] - half_height

    rot_x = (x*angle_cos) - (y*angle_sin)
    rot_y = (y*angle_cos) + (x*angle_sin)
    rot_x = rot_x+half_width
    rot_y = rot_y+half_height
    
    keypoints = tf.reshape(tf.stack([rot_x, rot_y], axis=-1), shape=(-1,34))[0]
    
    return image, keypoints

def shift_images_keypoints(image, keypoints, max_pixels=15):
    keypoints = tf.cast(keypoints, tf.float32)
    image_shape = tf.cast(tf.shape(image)[:2], tf.float32)


    translation = tf.random.uniform([2], minval=-max_pixels, maxval=max_pixels)
    
    x = keypoints[::2] + translation[0]
    y = keypoints[1::2] + translation[1]

    
    if tf.reduce_all(x < image_shape[1]) and tf.reduce_all(y < image_shape[0]):
        random_number = tf.random.uniform(())
        if  0 <= random_number and random_number < 0.33333:
            image = tfa.image.translate(image, translation, fill_mode='wrap')
        elif 0.33333 <= random_number and random_number < 0.66666:
            image = tfa.image.translate(image, translation, fill_mode='reflect')
        else:
            image = tfa.image.translate(image, translation)
        keypoints = tf.reshape(tf.stack([x, y], axis=-1), shape=(-1,34))[0]
        
    return image, keypoints
    

if __name__ == "__main__":
    main()