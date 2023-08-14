import time
import pandas as pd
from sklearn.model_selection import train_test_split

from R import *

set_tensorflow_logging_level(3)

import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
import tensorflow_text as text  # required


def train_model(epochs: int = 19, test: bool = True):
    df = pd.read_csv(DATASET_PARSED_FILE_PATH)
    x_data = df[COL_MESSAGE]
    y_data = df[COL_IS_SPAM]

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
                                                        shuffle=True,
                                                        random_state=100,
                                                        test_size=0.1)

    # Loading Bert
    start = time.time()
    bert_preprocessor = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
    bert_encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")
    end = time.time()
    print(f"BERT loaded in {end - start}", flush=True)

    # .................  Model (Functional style)  .................
    # BERT Layers
    text_input = keras.layers.Input(shape=(), dtype=tf.string, name="text_input")
    bert_pre_layer = bert_preprocessor(text_input)
    bert_enc_layer = bert_encoder(bert_pre_layer)

    # Neural Layers
    dropout = keras.layers.Dropout(0.1, name="dropout")(bert_enc_layer['pooled_output'])
    # dense1 = keras.layers.Dense(128, activation="relu")(bert_enc_layer['pooled_output'])
    output_layer = keras.layers.Dense(1, activation='sigmoid', name="output")(dropout)

    # Input to Output functional model
    model = keras.models.Model(inputs=[text_input], outputs=[output_layer])
    model.summary()

    # Compile
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Training
    train_history = model.fit(x_train, y_train, epochs=epochs)
    model.save(BERT_MODEL_FILE_PATH)

    if test:
        # test
        metrics = model.evaluate(x_test, y_test)
        print(metrics)

    return model


def load_bert_model():
    if os.path.isfile(BERT_MODEL_FILE_PATH):
        return keras.models.load_model(BERT_MODEL_FILE_PATH, custom_objects={'KerasLayer': hub.KerasLayer})

    print("No saved bert model found. Training the model from scratch...")
    return train_model(epochs=5, test=False)



def predict_spams(model, messages, verbose: int = 0):
    pred = model.predict(messages, verbose=verbose).reshape(-1)
    return [1 if i > 0.5 else 0 for i in pred]


def main():
    train_model()


if __name__ == '__main__':
    main()
