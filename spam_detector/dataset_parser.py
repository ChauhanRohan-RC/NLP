import pandas as pd
from R import *


def parse_dataset():
    # Dataset
    df = pd.read_csv(DATASET_ORIGINAL_FILE_PATH)
    print(df.groupby(COL_CATEGORY).describe())

    # Adding spam column
    df[COL_IS_SPAM] = df[COL_CATEGORY].apply(lambda x: 1 if x == TAG_SPAM else 0)

    df_spam = df[df[COL_IS_SPAM] == 1]
    df_ham = df[df[COL_IS_SPAM] == 0]

    if df_spam.shape[0] < df_ham.shape[0]:
        # Imbalanced dataset. Much more ham than spam

        df_ham_down_sampled = df_ham.sample(df_spam.shape[0])
        df_parsed = pd.concat([df_spam, df_ham_down_sampled])

    elif df_spam.shape[0] > df_ham.shape[0]:
        # Imbalanced dataset. Much more spam than ham

        df_spam_down_sampled = df_spam.sample(df_ham.shape[0])
        df_parsed = pd.concat([df_spam_down_sampled, df_ham])
    else:
        df_parsed = df

    print("\nParsed DataFrame")
    print(df_parsed.groupby(COL_CATEGORY).describe())
    df_parsed.to_csv(DATASET_PARSED_FILE_PATH, index=False)


if __name__ == '__main__':
    parse_dataset()
