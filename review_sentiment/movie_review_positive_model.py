import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tensorflow import keras

from R import *
from dataset_generator import positive_code


def train_model():
    df = pd.read_csv(DATASET_CSV_FILE_PATH, compression=DATASET_CSV_COMPRESSION)
    sentence_encoder = create_sentence_encoder(True)

    encoded_df = df[[col for col in df.columns if col.startswith(COL_ENCODED_REVIEW_PREFIX)]]
    x_data = encoded_df
    y_data = df[COL_POSITIVE]

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, shuffle=True, random_state=100, test_size=0.1)

    # Model
    model = keras.Sequential([
        keras.layers.Embedding(sentence_encoder.vocab.word_count, 24, input_length=MAX_ENCODED_REVIEW_LEN),
        keras.layers.GlobalAveragePooling1D(),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    model.summary()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Training
    train_history = model.fit(x_train, y_train, epochs=5)
    model.save("review_positive_model.h5")

    # Testing
    model = keras.models.load_model("review_positive_model.h5")

    test_result = model.evaluate(x_test, y_test)
    print("Test Result: ", test_result)

    y_pred = predict_postives(model, x_test, encode=False)
    print(classification_report(y_test, y_pred))


def predict_postives(model, reviews, encode: bool = True, sentence_encoder: SentenceEncoder = None,
                     verbose: int = 0) -> list:
    if encode:
        enc = sentence_encoder if sentence_encoder else create_sentence_encoder(True)
        reviews = enc.encode_sentences(reviews, False)
    pred = model.predict(reviews, verbose=verbose).reshape(-1)
    return [positive_code(r, 0, 1) for r in pred]
