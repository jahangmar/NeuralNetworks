import numpy as np
import os
import tensorflow as tf

import common

common.parse_args()


training_data = tf.keras.preprocessing.image_dataset_from_directory(
        common.image_dir,
        validation_split=common.validation_split,
        subset="training",
        seed = common.seed,
        image_size=common.image_size,
        batch_size=common.batch_size
        )


validation_data = tf.keras.preprocessing.image_dataset_from_directory(
        common.image_dir,
        validation_split=common.validation_split,
        subset="validation",
        seed = common.seed,
        image_size=common.image_size,
        batch_size=common.batch_size
        )

class_names = training_data.class_names
print ("Loaded the following classes: {}".format(class_names))

num_classes = len(class_names)

#Rescale pixels from 0..255 to 0f..1f values
norm_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
training_data = training_data.map(lambda x, y: (norm_layer(x), y))
validation_data = training_data.map(lambda x, y: (norm_layer(x), y))

#Image caching
training_data = training_data.cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
validation_data = validation_data.cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

model = common.create_model(num_classes)

model.compile(
        optimizer='adam',
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
        )

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(                
        filepath = common.checkpoint_dir,
        save_best_only=True,
        verbose=1
        )

common.save_class_names(class_names)

model.fit(training_data, validation_data=validation_data, epochs=common.epochs, callbacks=[checkpoint_callback])
