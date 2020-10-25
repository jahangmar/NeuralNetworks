import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers
import argparse
import random
import json

batch_size = 32
image_size = (180, 180)
validation_split = 0.2
seed = random.randint(1,1000)
epochs = 1000

parser = argparse.ArgumentParser()
parser.add_argument("directory", nargs=1)

def parse_args():
    args = parser.parse_args()
    global directory
    directory = args.directory[0]
    global image_dir
    image_dir = os.path.join(directory, "images")
    print("image_folder is {}".format(image_dir))
    global checkpoint_dir
    checkpoint_dir = os.path.join(directory, "checkpoint_best")
    return args

input_shape = (image_size[0], image_size[1], 3) 

#for generating addition training data
augmentation_layer = tf.keras.Sequential(
        [
            layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=input_shape),
            layers.experimental.preprocessing.RandomRotation(0.1),
            layers.experimental.preprocessing.RandomZoom(0.1)
        ])

def create_model(num_classes):
    model = tf.keras.Sequential([
        augmentation_layer,
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dense(num_classes)
        ])
    return model


def save_class_names(class_names):
    f = open(os.path.join(directory, 'class_names'), 'w')
    json.dump(class_names, f)
    f.close()

def load_class_names():
    f = open(os.path.join(directory, 'class_names'), 'r')
    class_names = json.load(f)
    f.close()
    return class_names
