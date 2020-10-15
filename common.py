import os;
import argparse;
import numpy as np;
import tensorflow as tf;

seq_length = 100
batch_size = 64
buffer_size = 10000

embedding_dim = 256
rnn_units = 1024

epochs=1


parser = argparse.ArgumentParser()
parser.add_argument('directory', nargs=1)

args = parser.parse_args()
directory = args.directory[0]

checkpoint_dir = os.path.join(directory, 'checkpoints')

filename = os.path.join(directory, 'input.txt')

text = open(filename, 'rb').read().decode(encoding='utf-8')
print('Length of file {} is {} characters.'.format(filename, len(text)))

vocab = sorted(set(text))
char2idx = {u:i for i,u in enumerate(vocab)}
idx2char = np.array(vocab)

itext = np.array([char2idx[c] for c in text])


def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
            tf.keras.layers.GRU(rnn_units, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'),
            tf.keras.layers.Dense(vocab_size)
        ])            
        return model


