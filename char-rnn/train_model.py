import tensorflow as tf
import numpy as np
import os
import time

from common import *;

parser.add_argument('--epochs', nargs=1, default=epochs, type=int)
parser.add_argument('--seq_length', nargs=1, type=int, default=seq_length)
#parser.add_argument('--load_model', nargs=1, default=-1, type=int)
args = parse_args()
epochs = args.epochs[0]
seq_length = args.seq_length[0]
from common import checkpoint_dir, texts, vocab, char2idx, idx2char, itexts;

char_datasets = []
for itext in itexts:
    char_datasets.append(tf.data.Dataset.from_tensor_slices(itext))

#for each chunk, form input and target sequence
def split_input_target(chunk):
    inputt = chunk[:-1]
    targett = chunk[1:]
    return inputt, targett

def create_single_dataset(c_dataset):
    return c_dataset.batch(seq_length+1, drop_remainder=True).map(split_input_target)

#break each text into chunks of size seq_length+1
#but don't mix inputs
dataset = create_single_dataset(char_datasets[0])
for char_dataset in char_datasets[1:]:
    dataset = dataset.concatenate(create_single_dataset(char_dataset))

dataset = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)

if os.path.isdir(checkpoint_dir):
    print('###loading model from checkpoint')
    model = tf.keras.models.load_model(checkpoint_dir, compile=False)
else:
    print('###creating new model')
    model = build_model(
        vocab_size=len(vocab),
        embedding_dim=embedding_dim,
        rnn_units=rnn_units,
        batch_size=batch_size)


def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=0.001,
        end_learning_rate=0.00001,
        decay_steps=10000
        )

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule, amsgrad=False), loss=loss)


checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
#        filepath = os.path.join(checkpoint_dir, "ckpt_{epoch}"),
        filepath = checkpoint_dir,
        save_best_only=True,
        monitor='loss',
        mode = 'min',
        verbose=1
        )

history = model.fit(dataset, epochs=epochs, callbacks=[checkpoint_callback])
