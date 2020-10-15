import tensorflow as tf
import numpy as np
import os
import time

from common import *;

parser.add_argument('--epochs', nargs=1, default=epochs, type=int)
parser.add_argument('--load_model', nargs=1, default=-1, type=int)
args = parse_args()
epochs = args.epochs[0]
load_model_num = args.load_model[0]
from common import checkpoint_dir, text, vocab, char2idx, idx2char, itext;

examples_per_epoch = len(text)

char_dataset = tf.data.Dataset.from_tensor_slices(itext)

#break text into chunks of size seq_length+1
sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

#for each chunk, form input and target sequence
def split_input_target(chunk):
    inputt = chunk[:-1]
    targett = chunk[1:]
    return inputt, targett

dataset = sequences.map(split_input_target)


dataset = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)

if load_model_num > -1:
    print ('bla')
else:
    model = build_model(
        vocab_size=len(vocab),
        embedding_dim=embedding_dim,
        rnn_units=rnn_units,
        batch_size=batch_size)


def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

model.compile(optimizer='adam', loss=loss)


checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath = os.path.join(checkpoint_dir, "ckpt_{epoch}"),
        verbose=1,
        save_weigths_only=True)

history = model.fit(dataset, epochs=epochs, callbacks=[checkpoint_callback])
