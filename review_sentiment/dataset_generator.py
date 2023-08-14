import os.path

import pandas as pd

from R import *
from utils import get_lerp_factor


def parse_review_test(review_text: str) -> str:
    return (review_text
            .replace('<br />', ' ')
            .replace('<br>', ' ')
            .replace('</br>', ' ')
            .replace('<br/>', ' ')
            .strip())


def is_rating_positive(rating: int | float, start: int | float, end: int | float) -> bool:
    return get_lerp_factor(start, end, rating) > 0.5


def positive_code(rating: int | float, start: int | float, end: int | float) -> int:
    return 1 if is_rating_positive(rating, start, end) else 0


def create_entry(dir_, file_name, sentence_encoder: SentenceEncoder) -> dict:
    name = os.path.splitext(file_name)[0]
    id_, rating = name.split("_")
    rating = int(rating)

    with open(os.path.join(dir_, file_name), 'r', encoding="utf-8") as f:
        text = parse_review_test(f.read())
        dic = {
            COL_REVIEW: text,
            COL_RATING: rating,
            COL_POSITIVE: positive_code(rating, 1, 10),
        }

        encoded = sentence_encoder.encode_sentence(text)

        for i, v in enumerate(encoded):
            dic[COL_ENCODED_REVIEW_PREFIX + str(i)] = v

        return dic


def create_dataset() -> pd.DataFrame:
    sentence_encoder = create_sentence_encoder(False)
    entries = []

    for dir_ in os.listdir(DIR_DATASET_RAW):
        dir_ = os.path.join(DIR_DATASET_RAW, dir_)
        for file_name in os.listdir(dir_):
            entries.append(create_entry(dir_, file_name, sentence_encoder))

    df = pd.DataFrame(entries)
    df.to_csv(DATASET_CSV_FILE_PATH, index=False, compression=DATASET_CSV_COMPRESSION)  # Save dataset
    sentence_encoder.save_vocab(VOCAB_FILE_PATH)  # Save vocabulary

    print(f"Saved dataset file to <{DATASET_CSV_FILE_PATH}>")
    print(df.info())
    return df


def main():
    create_dataset()


if __name__ == '__main__':
    main()
