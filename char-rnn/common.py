import os;
import sys;
import argparse;
import numpy as np;
import tensorflow as tf;

seq_length = 100#100
batch_size = 64#64
buffer_size = 10000

embedding_dim = 256#256
rnn_units = 1024#1024

epochs=10

parser = argparse.ArgumentParser()
parser.add_argument('directory', nargs=1)

def parse_args():
    args = parser.parse_args()
    directory = args.directory[0]

    global checkpoint_dir
    checkpoint_dir = os.path.join(directory, 'checkpoint_best')

    input_dir = os.path.join(directory, 'inputs')
    filenames = os.listdir(input_dir)

    global vocab, char2idx, idx2char, itexts
    global texts
    texts = []
    vocab = set()
    for filename in filenames:
        text = open(os.path.join(input_dir, filename), 'rb').read().decode(encoding='utf-8')
        texts.append(text)
        vocab = vocab | set(text)
        print('Loaded {} with {} characters.'.format(filename, len(text)))
        print('whole vocabulary has now size {}'.format(len(vocab)))


    vocab = sorted(vocab)
    char2idx = {u:i for i,u in enumerate(vocab)}
    idx2char = np.array(vocab)

    itexts = []
    for text in texts:
        itexts.append(np.array([char2idx[c] for c in text]))

    return args


def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
#            tf.keras.layers.GRU(rnn_units, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'),
            tf.keras.layers.LSTM(rnn_units, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'),
#            tf.keras.layers.LSTM(rnn_units, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'),
            tf.keras.layers.GRU(rnn_units, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'),
#            tf.keras.layers.SimpleRNN(rnn_units, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'),
            tf.keras.layers.Dense(vocab_size)
        ])            
        return model


