import sys
import os
from sentence_encoder import SentenceEncoder

FROZEN = getattr(sys, 'frozen', False)
DIR_MAIN = os.path.dirname(sys.executable) if FROZEN else (
    os.path.dirname(os.path.abspath(os.path.realpath(__file__))))

# Dataset
DIR_DATASET = os.path.join(DIR_MAIN, 'dataset')

# raw_dataset_dir -> more folders -> files with name format {id}_{rating}.txt
DIR_DATASET_RAW = os.path.join(DIR_DATASET, 'raw')
DATASET_CSV_FILE_PATH = os.path.join(DIR_DATASET, 'imdb_dataset.csv.gz')
DATASET_CSV_COMPRESSION = 'gzip'
VOCAB_FILE_PATH = os.path.join(DIR_DATASET, 'imdb_vocabulary.pickle')

COL_REVIEW = 'review'
COL_RATING = 'rating'
COL_POSITIVE = 'positive'
COL_ENCODED_REVIEW_PREFIX = 'enc_'

MAX_ENCODED_REVIEW_LEN = 300


def create_sentence_encoder(load_vocab: bool) -> SentenceEncoder:
    enc = SentenceEncoder(MAX_ENCODED_REVIEW_LEN)
    if load_vocab:
        enc.load_vocab(VOCAB_FILE_PATH)
    return enc


# Models
DIR_MODELS = os.path.join(DIR_MAIN, "models")
MODEL_REVIEW_RATING_FILE_PATH = os.path.join(DIR_MODELS, "review_rating_model.h5")
MODEL_REVIEW_POSITIVE_FILE_PATH = os.path.join(DIR_MODELS, "review_positive_model.h5")
