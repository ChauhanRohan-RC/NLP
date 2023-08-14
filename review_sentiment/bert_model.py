import time
import pandas as pd
from sklearn.model_selection import train_test_split
from R import *
from utils import set_tensorflow_logging_level
set_tensorflow_logging_level(2)

import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub

print("Tensorflow imported", flush=True)

# Dataset
df = pd.read_csv(DATASET_CSV_FILE_PATH)

start = time.time()
bert_preprocessor = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
bert_encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")
end = time.time()
print(f"BERT loaded in {end - start}", flush=True)

df.groupby(COL_POSITIVE).describe()

pos_df = df[df[COL_POSITIVE] == 1]
pos_df_downsample = pos_df.sample(700)

neg_df = df[df[COL_POSITIVE] == 0]
neg_df_downsample = neg_df.sample(700)

df_downsample = pd.concat((pos_df_downsample, neg_df_downsample))
df_downsample.groupby(COL_POSITIVE).describe()

x_data = df_downsample[COL_REVIEW]
y_data = df_downsample[COL_POSITIVE]

# Train test split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, random_state=40, shuffle=True,
                                                    test_size=0.1)

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
train_history = model.fit(x_train, y_train, epochs=3, batch_size=100)
model.save("movie_review_bert.h5")

# # test
# metrics = model.evaluate(x_test, y_test)
# print(metrics)
