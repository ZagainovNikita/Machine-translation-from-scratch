import numpy as np

import tensorflow as tf
try: [tf.config.experimental.set_memory_growth(gpu, True) for gpu in tf.config.experimental.list_physical_devices("GPU")]
except: pass

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from mltu.tensorflow.callbacks import Model2onnx, WarmupCosineDecay
from keras import Model

from mltu.tensorflow.dataProvider import DataProvider
from mltu.tokenizers import CustomTokenizer

from mltu.tensorflow.transformer.utils import MaskedAccuracy, MaskedLoss
from mltu.tensorflow.transformer.callbacks import EncDecSplitCallback
from mltu.tensorflow.transformer.layers import Encoder, Decoder

from config import SEQ_SIZE, BATCH_SIZE, LR
from data import es_train, en_train

Dataset = tf.data.Dataset

def process_and_train(model, train_dataset, valid_dataset) -> Model:
    optimizer = tf.keras.optimizers.Adam(learning_rate=LR, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    model.compile(
        loss=MaskedLoss(),
        optimizer=optimizer,
        metrics=[MaskedAccuracy()],
        run_eagerly=False)
    warmupCosineDecay = WarmupCosineDecay(
        lr_after_warmup=0.0001,
        final_lr=LR / 10,
        warmup_epochs=2,
        decay_epochs=18,
        initial_lr=LR,
    )

    early_stopper = EarlyStopping(monitor="val_masked_accuracy",
                                  patience=5,
                                  verbose=1,
                                  mode="max")
    reduceLROnPlat = ReduceLROnPlateau(monitor="val_masked_accuracy",
                                       factor=0.9,
                                       min_delta=1e-10,
                                       patience=2,
                                       verbose=1,
                                       mode="max")
    model.fit(train_dataset,
                    validation_data=[valid_dataset],
                    epochs=3,
                    callbacks=[warmupCosineDecay, reduceLROnPlat])

    return model
