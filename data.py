import numpy as np

import tensorflow as tf
try: [tf.config.experimental.set_memory_growth(gpu, True) for gpu in tf.config.experimental.list_physical_devices("GPU")]
except: pass

from mltu.tokenizers import CustomTokenizer


from config import SEQ_SIZE, BATCH_SIZE

Dataset = tf.data.Dataset

train_input_path = 'Datasets/en-es/opus.en-es-train.en'
train_output_path = 'Datasets/en-es/opus.en-es-train.es'
valid_input_path = 'Datasets/en-es/opus.en-es-dev.en'
valid_output_path = 'Datasets/en-es/opus.en-es-dev.es'

def read_file(path):
    with open(path, 'r') as f:
        return f.read().split('\n')[:-1]

en_train = read_file(train_input_path)
es_train = read_file(train_output_path)
en_valid = read_file(valid_input_path)
es_valid = read_file(valid_output_path)

max_length = 500
train_dataset = [[en_sentence, es_sentence] for en_sentence, es_sentence in zip(en_train, es_train) if len(en_sentence) <= max_length and len(es_sentence) <= max_length]
valid_dataset = [[en_sentence, es_sentence] for en_sentence, es_sentence in zip(en_valid, es_valid) if len(en_sentence) <= max_length and len(es_sentence) <= max_length]
en_train, es_train = zip(*train_dataset)
en_valid, es_valid = zip(*valid_dataset)


tokenizer = CustomTokenizer()
detokenizer = CustomTokenizer()

tokenizer.fit_on_texts(en_train)
detokenizer.fit_on_texts(es_train)
def preprocess_inputs(data_batch, label_batch, seq_size=SEQ_SIZE):
    encoder_input = np.zeros((len(data_batch), seq_size)).astype(np.int64)
    decoder_input = np.zeros((len(label_batch), seq_size)).astype(np.int64)
    decoder_output = np.zeros((len(label_batch), seq_size)).astype(np.int64)

    data_batch_tokens = tokenizer.texts_to_sequences(data_batch)
    label_batch_tokens = detokenizer.texts_to_sequences(label_batch)

    for index, (data, label) in enumerate(zip(data_batch_tokens, label_batch_tokens)):
        encoder_input[index][:len(data)] = data
        decoder_input[index][:len(label) - 1] = label[:-1]  # Drop the [END] tokens
        decoder_output[index][:len(label) - 1] = label[1:]  # Drop the [START] tokens

    return (encoder_input, decoder_input), decoder_output


def train_generator():
    for en_sent, es_sent in zip(en_train, es_train):
        yield preprocess_inputs([en_sent], [es_sent])


def valid_generator():
    for en_sent, es_sent in zip(en_valid, es_valid):
        yield preprocess_inputs([en_sent], [es_sent])

train_dataset = Dataset.from_generator(train_generator, output_signature=(
    (tf.TensorSpec(shape=(1, SEQ_SIZE), dtype=tf.int64), tf.TensorSpec(shape=(1, SEQ_SIZE), dtype=tf.int64)),
    tf.TensorSpec(shape=(1, SEQ_SIZE), dtype=tf.int64)
)).unbatch().batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE).cache()

valid_dataset = Dataset.from_generator(valid_generator, output_signature=(
    (tf.TensorSpec(shape=(1, SEQ_SIZE), dtype=tf.int64), tf.TensorSpec(shape=(1, SEQ_SIZE), dtype=tf.int64)),
    tf.TensorSpec(shape=(1, SEQ_SIZE), dtype=tf.int64)
)).unbatch().batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE).cache()
