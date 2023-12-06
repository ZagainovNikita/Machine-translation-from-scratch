from data import preprocess_inputs
import tensorflow as tf


def make_pred(model, enc_input, dec_input):
    return tf.argmax(model((enc_input, dec_input)), axis=-1)


def translate(input_sent):
    (enc_input, dec_input), _ = preprocess_inputs([input_sent], [""])
    for i in range(1, 250):
        pred = make_pred(enc_input, dec_input)
        dec_input[0][i] = pred[0][i]
        if dec_input[0][i] == 32:
            return dec_input
    return dec_input
