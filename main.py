import keras.models

from model import Transformer
from train import process_and_train
from data import train_dataset, valid_dataset, tokenizer, detokenizer
from config import *
import os

def train_transformer():
    transformer = Transformer(input_vocab_size=len(tokenizer),
                              target_vocab_size=len(detokenizer)+1,
                              encoder_input_size=SEQ_SIZE,
                              decoder_input_size=SEQ_SIZE,
                              num_layers=6,
                              d_model=512,
                              num_heads=8,
                              dff=2048,
                              dropout_rate=0.1)

    translator = process_and_train(transformer, train_dataset, valid_dataset)
    if not os.path.exists("Models"):
        os.mkdir("Models")
    tokenizer.save("Models/tokenizer.json")
    detokenizer.save("Models/detokenizer.json")
    translator.save("Models/translator.h5", save_format="h5")

def main():
    if (os.path.exists("Models/tokenizer.json")
            and os.path.exists("Models/detokenizer.json")
            and os.path.exists("Models/translator.h5")):
        train_transformer()
    transformer = keras.models.load_model("Models/translator.h5")
    tokenizer = keras.models.load_model("Models/tokenizer.json")
    detokenizer = keras.models.load_model("Model/detokenizer.json")

