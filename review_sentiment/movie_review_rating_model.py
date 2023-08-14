import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
from R import *


def train_rating_model():
    df = pd.read_csv(DATASET_CSV_FILE_PATH, compression=DATASET_CSV_COMPRESSION)
    sentence_encoder = create_sentence_encoder(True)

    encoded_df = df[[col for col in df.columns if col.startswith(COL_ENCODED_REVIEW_PREFIX)]]
    x_data = encoded_df
    y_data = df[COL_RATING] - 1  # To make ratings [0, 9]

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, shuffle=True, test_size=0.1)

    # Model
    model = keras.Sequential([
        keras.layers.Embedding(sentence_encoder.vocab.word_count, 16, input_length=MAX_ENCODED_REVIEW_LEN),
        keras.layers.GlobalAvgPool1D(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])

    model.summary()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Training
    train_history = model.fit(x_data, y_data, epochs=10)
    model.save(MODEL_REVIEW_RATING_FILE_PATH)

    # Testing
    test_result = model.evaluate(x_test, y_test)
    print("Test Result: ", test_result)


def predict_ratings(model, reviews, encode: bool = True, sentence_encoder: SentenceEncoder = None,
                    verbose: int = 0) -> list:
    if encode:
        enc = sentence_encoder if sentence_encoder else create_sentence_encoder(True)
        reviews = enc.encode_sentences(reviews, False)
    pred = model.predict(reviews, verbose=verbose)
    return [np.argmax(r) for r in pred]


if __name__ == '__main__':
    train_rating_model()
