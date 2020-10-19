import os;
import sys;
import argparse;
import numpy as np;
import tensorflow as tf;
import re;
import json

seq_length = 20
batch_size = 32
buffer_size = 10000

embedding_dim = 256
rnn_units = 1024

epochs=10

parser = argparse.ArgumentParser()
parser.add_argument('directory', nargs=1)

def parse_args(generate_vocab=True):
    args = parser.parse_args()
    global directory
    directory = args.directory[0]

    global checkpoint_dir
    checkpoint_dir = os.path.join(directory, 'checkpoint_best')

    input_dir = os.path.join(directory, 'inputs')
    filenames = os.listdir(input_dir)

    global vocab
    global word_lists
    global iword_lists
    word_lists = []
    vocab = set()
    if generate_vocab:
        for filename in filenames:
            text = open(os.path.join(input_dir, filename), 'rb').read().decode(encoding='utf-8')
            word_list = text_to_word_list(text)
            word_lists.append(word_list)
            vocab = vocab | set(word_list)
            print('Loaded {} with {} characters and {} words.'.format(filename, len(text), len(word_list)))
            print('whole vocabulary has now size {}'.format(len(vocab)))
        vocab = sorted(vocab)
    else:
        vocab = load_vocabulary()

    set_values_from_vocab(vocab)

    iword_lists = []
    for word_list in word_lists:
        iword_lists.append(np.array([word2idx[w] for w in word_list]))

    return args

def set_values_from_vocab(vocab):
    global word2idx, idx2word
    word2idx = {u:i for i,u in enumerate(vocab)}
    idx2word = np.array(vocab)


__unclean_word_reg__ = re.compile('\s+')
__word_reg__ = re.compile('(\w+)')
__shortened_reg = re.compile("\w+'\w+")

def text_to_word_list(text):
    result = []
    words = __unclean_word_reg__.split(text)
    for w in words:
        cleaned = __word_reg__.split(w)
        
        before = ""        
        shortened = ""
        for l in cleaned:
            if shortened != "" and is_word(l):
                result[-2] = shortened + l
                del result[-1]
                shortend = ""
                continue
            elif l == "'" and is_word(before):
                shortened = before + l
            if l != '':
                result.append(l) 
            before = l

    return result

def word_list_to_text(word_list):
    result = ''
    for word in word_list:
        if is_word(word) or is_shortened_word(word):
            result = result + ' ' + word
        else:
            result = result + word
    return result


def is_word(w):
    return __word_reg__.fullmatch(w) != None


def is_shortened_word(w):
    return __shortened_reg.fullmatch(w) != None

__vocab_file = 'vocabulary.json'

def save_vocabulary(vocab):
    f = open(os.path.join(directory, __vocab_file), 'w')
    json.dump(vocab, f)
    f.close()

def load_vocabulary():
    f = open(os.path.join(directory, __vocab_file), 'r')
    v = json.load(f)
    f.close()
    return v


def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
#            tf.keras.layers.GRU(rnn_units, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'),
#            tf.keras.layers.SimpleRNN(rnn_units, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'),
            tf.keras.layers.LSTM(rnn_units, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'),
            tf.keras.layers.LSTM(rnn_units, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'),
#            tf.keras.layers.GRU(rnn_units, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'),
#            tf.keras.layers.SimpleRNN(rnn_units, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'),
            tf.keras.layers.Dense(vocab_size)
        ])            
        return model


