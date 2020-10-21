import os
import tensorflow as tf
import numpy as np

import common

common.parser.add_argument("image_file", nargs=1)

args = common.parse_args()
image_file = args.image_file[0]

image = tf.keras.preprocessing.image.load_img(image_file, target_size=common.image_size)

image_array = tf.keras.preprocessing.image.img_to_array(image)
image_array = tf.expand_dims(image_array, 0)

#load model
model = tf.keras.models.load_model(common.checkpoint_dir)

predictions = model.predict(image_array)
score = tf.nn.softmax(predictions[0])

class_names = common.load_class_names()

print("Result: Image belongs to {} ({}% confidence)".format(class_names[np.argmax(score)], 100 * np.max(score)))
