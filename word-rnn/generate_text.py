import numpy as np;
import tensorflow as tf;
import sys

from common import *;

parser.add_argument('start_words', nargs=1)
args = parse_args(generate_vocab=False)
start_words = args.start_words[0]

from common import checkpoint_dir, vocab, word2idx, idx2word;

def generate_text(model, start_words):
    num=100
   
    for word in start_words:
        if word not in word2idx:
            print("Error: '{}' is not part of the vocabulary".format(word))
            sys.exit();

    input_eval = [word2idx[word] for word in start_words]
    input_eval = tf.expand_dims(input_eval, 0)

    words_generated = []

    temperature = 1.0

    model.reset_states()
    for i in range(num):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)
        predicitions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

        input_eval = tf.expand_dims([predicted_id], 0)
        words_generated.append(idx2word[predicted_id])

    return (word_list_to_text(start_words + words_generated))


latest = checkpoint_dir

loaded_model = tf.keras.models.load_model(latest, compile=False)


model = build_model(len(vocab), embedding_dim, rnn_units, batch_size=1)
model.set_weights(loaded_model.get_weights())
model.build(tf.TensorShape({1, None}))

text = generate_text(model, start_words=text_to_word_list(start_words))
print(text)
