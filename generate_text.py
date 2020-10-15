import numpy as np;
import tensorflow as tf;
import sys

from common import *;

parser.add_argument('ckpt', nargs=1)
parser.add_argument('start_string', nargs=1)
args = parse_args()
start_string = args.start_string[0]
ckpt_n = args.ckpt[0]

from common import checkpoint_dir, text, vocab, char2idx, idx2char, itext;

def generate_text(model, start_string):
    num=1000
    
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    text_generated = []

    temperature = 1.0

    model.reset_states()
    for i in range(num):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)

        predicitions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(idx2char[predicted_id])

    return (start_string + ''.join(text_generated))



#latest = tf.train.latest_checkpoint(checkpoint_dir)
#TODO temporary solution since latest_checkpoint doesn't work
latest = os.path.join(checkpoint_dir, 'ckpt_' + ckpt_n)

loaded_model = tf.keras.models.load_model(latest, compile=False)


model = build_model(len(vocab), embedding_dim, rnn_units, batch_size=1)
model.set_weights(loaded_model.get_weights())
model.build(tf.TensorShape({1, None}))

text = generate_text(model, start_string=start_string)
print(text)
