import sys
import os

def set_tensorflow_logging_level(level: int):  # level [0, 3] from most to least verbose
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(level)


FROZEN = getattr(sys, 'frozen', False)
DIR_MAIN = os.path.dirname(sys.executable) if FROZEN else (
    os.path.dirname(os.path.abspath(os.path.realpath(__file__))))

# Dataset
DIR_DATASET = os.path.join(DIR_MAIN, 'dataset')
DATASET_ORIGINAL_FILE_PATH = os.path.join(DIR_DATASET, 'spam_original.csv')
DATASET_PARSED_FILE_PATH = os.path.join(DIR_DATASET, 'spam_parsed.csv')

COL_CATEGORY = 'category'
COL_IS_SPAM = 'is_spam'
COL_MESSAGE = 'message'

TAG_SPAM = 'spam'
TAG_NOT_SPAM = 'ham'

# Models
DIR_MODELS = os.path.join(DIR_MAIN, "models")
BERT_MODEL_FILE_PATH = os.path.join(DIR_MODELS, "spam_bert_model.h5")

