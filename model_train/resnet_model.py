import numpy as np
import tensorflow as tf
from tensorflow import keras
import boto3
import pandas as pd
import cv2
import os

import tensorflow_datasets as tfds

tfds.disable_progress_bar()

session = boto3.Session(
    aws_access_key_id="AKIAV3KKLC57NGTGPB7K",
    aws_secret_access_key="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",
    region_name='us-east-1'
)

s3 = session.client('s3')

# s3 = boto3.client('s3')

paginator = s3.get_paginator('list_objects_v2')
pages = paginator.paginate(Bucket='dalle2images', Prefix='fake/')
pages_real = paginator.paginate(Bucket='dalle2images', Prefix='real/')

fake_images = list()
real_images = list()

for page in pages:
    for image in page["Contents"]:
        fake_images.append(image["Key"])
    break

for page in pages_real:
    for image in page["Contents"]:
        real_images.append(image["Key"])
    break

fake_images = fake_images[1:]
real_images = real_images[1:]

# for fake_image in fake_images:
#     print(fake_image)
#     s3.download_file('dalle2images', fake_image, fake_image)

# for real_image in real_images:
#     s3.download_file('dalle2images', real_image, real_image)

train_data = list()

for img in os.listdir("fake/"):
    img_arr=cv2.imread("fake/"+img)
    img_arr=cv2.resize(img_arr,(224,224))
    train_data.append(img_arr)

for img in os.listdir("real/"):
    img_arr=cv2.imread("real/"+img)
    img_arr=cv2.resize(img_arr,(224,224))
    train_data.append(img_arr)

print(len(train_data))

train_ds, validation_ds, test_ds = tfds.load(
    "cats_vs_dogs",
    # Reserve 10% for validation and 10% for test
    split=["train[:40%]", "train[40%:50%]", "train[50%:60%]"],
    as_supervised=True,  # Include labels
)

print("Number of training samples: %d" % tf.data.experimental.cardinality(train_ds))
print(
    "Number of validation samples: %d" % tf.data.experimental.cardinality(validation_ds)
)
print("Number of test samples: %d" % tf.data.experimental.cardinality(test_ds))

size = (150, 150)

train_ds = train_ds.map(lambda x, y: (tf.image.resize(x, size), y))
validation_ds = validation_ds.map(lambda x, y: (tf.image.resize(x, size), y))
test_ds = test_ds.map(lambda x, y: (tf.image.resize(x, size), y))

labels = ["fake"]*999 + ["real"]*999
train_data = train_data / 255
dataset = tf.data.Dataset.from_tensor_slices((train_data, labels))

'''
batch_size = 32

train_ds = train_ds.cache().batch(batch_size).prefetch(buffer_size=10)
validation_ds = validation_ds.cache().batch(batch_size).prefetch(buffer_size=10)
test_ds = test_ds.cache().batch(batch_size).prefetch(buffer_size=10)

from tensorflow import keras
from tensorflow.keras import layers

data_augmentation = keras.Sequential(
    [layers.RandomFlip("horizontal"), layers.RandomRotation(0.1),]
)

base_model = keras.applications.Xception(
    weights="imagenet",  # Load weights pre-trained on ImageNet.
    input_shape=(150, 150, 3),
    include_top=False,
)  # Do not include the ImageNet classifier at the top.

# Freeze the base_model
base_model.trainable = False

# Create new model on top
inputs = keras.Input(shape=(150, 150, 3))
x = data_augmentation(inputs)  # Apply random data augmentation

# Pre-trained Xception weights requires that input be scaled
# from (0, 255) to a range of (-1., +1.), the rescaling layer
# outputs: `(inputs * scale) + offset`
scale_layer = keras.layers.Rescaling(scale=1 / 127.5, offset=-1)
x = scale_layer(x)

# The base model contains batchnorm layers. We want to keep them in inference mode
# when we unfreeze the base model for fine-tuning, so we make sure that the
# base_model is running in inference mode here.
x = base_model(x, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.2)(x)  # Regularize with dropout
outputs = keras.layers.Dense(1)(x)
model = keras.Model(inputs, outputs)

model.summary()

model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[keras.metrics.BinaryAccuracy()],
)

epochs = 20
model.fit(train_ds, epochs=epochs, validation_data=validation_ds)

'''