from collections.abc import Iterable

import numpy as np
import pickle


def text_to_word_sequence(
        input_text,
        filters='!"\'#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
        lower=True,
        split=" ",
):
    """Converts a text to a sequence of words (or tokens).
    This function transforms a string of text into a list of words
    while ignoring `filters` which include punctuations by default.

    >>> sample_text = 'This is a sample sentence.'
    >>> text_to_word_sequence(sample_text)
    ['this', 'is', 'a', 'sample', 'sentence']

    Args:
        input_text: Input text (string).
        filters: list (or concatenation) of characters to filter out, such as
            punctuation. Default: ``'!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\\t\\n'``,
              includes basic punctuation, tabs, and newlines.
        lower: boolean. Whether to convert the input to lowercase.
        split: str. Separator for word splitting.

    Returns:
        A list of words (or tokens).
    """
    if lower:
        input_text = input_text.lower()

    translate_dict = {c: split for c in filters}
    translate_map = str.maketrans(translate_dict)
    input_text = input_text.translate(translate_map)

    seq = input_text.split(split)
    return [i for i in seq if i]


class Vocabulary:
    PAD_TOKEN = "<PAD>"
    PAD_INDEX = 0

    START_TOKEN = "<START>"
    START_INDEX = 1

    END_TOKEN = "<END>"
    END_INDEX = 2

    UNKNOWN_TOKEN = "<UNK>"
    UNKNOWN_INDEX = 3

    VOCAB_START_INDEX = 10

    @staticmethod
    def inverse_dict(dict_: dict):
        return dict(((v, k) for k, v in dict_.items()))

    def __init__(self):
        self._vocab_counter = self.VOCAB_START_INDEX

        # word -> index
        self._vocab: dict = {
            self.PAD_TOKEN: self.PAD_INDEX,
            self.START_TOKEN: self.START_INDEX,
            self.END_TOKEN: self.END_INDEX,
            self.UNKNOWN_TOKEN: self.UNKNOWN_INDEX,
        }

        # index -> word
        self._inverse_vocab: dict = self.inverse_dict(self._vocab)

    def __getstate__(self):
        return self._vocab_counter, self._vocab

    def __setstate__(self, state):
        self._vocab_counter, self._vocab = state
        self._inverse_vocab = self.inverse_dict(self._vocab)

    @property
    def word_count(self):
        return len(self._vocab)

    @property
    def vocab_counter(self):
        return self._vocab_counter

    def word_from_index(self, index: int, def_value: str | None) -> str:
        return self._inverse_vocab.get(index, def_value)

    def word_index(self, w: str, update_vocab: bool = True) -> int:
        if w in self._vocab:
            return self._vocab[w]

        if update_vocab:
            idx = self._vocab_counter
            self._vocab_counter += 1
            self._vocab[w] = idx
            self._inverse_vocab[idx] = w
            return idx

        return self.UNKNOWN_INDEX

    def copy(self):
        copy = self.__class__()
        copy._vocab_counter = self._vocab_counter
        copy._vocab = self._vocab.copy()
        copy._inverse_vocab = self._inverse_vocab.copy()
        return copy

    def __repr__(self):
        return f"Vocabulary(counter={self._vocab_counter}, vocab={self._vocab})"


class SentenceEncoder:

    @staticmethod
    def load_vocabulary(file_path) -> Vocabulary:
        with open(file_path, "rb") as file:
            return pickle.load(file)

    def __init__(self, max_seq_length: int, sentence_splitter=None):
        self.max_seq_length = max_seq_length
        self.sentence_splitter = sentence_splitter
        self._vocab: Vocabulary = Vocabulary()

    def save_vocab(self, file_path):
        with open(file_path, "wb+") as file:
            pickle.dump(self._vocab, file)

    def load_vocab(self, file_path):
        self._vocab = self.load_vocabulary(file_path)

    @property
    def vocab(self) -> Vocabulary:
        return self._vocab

    def encode_word_seq(self, words_sequence: list[str], update_vocab: bool = True) -> np.ndarray:
        encoded = np.full(shape=(self.max_seq_length,), fill_value=self._vocab.PAD_INDEX)

        if len(words_sequence) > 0:
            encoded[0] = self._vocab.START_INDEX

            wi = 1
            for wi, w in enumerate(words_sequence, start=1):
                if wi >= self.max_seq_length - 1:
                    break
                encoded[wi] = self._vocab.word_index(w, update_vocab=update_vocab)

            encoded[wi] = self._vocab.END_INDEX

        return encoded

    def encode_sentence(self, sentence: str, update_vocab: bool = True) -> np.ndarray:
        splitter = self.sentence_splitter if self.sentence_splitter else text_to_word_sequence
        seq = splitter(sentence)
        return self.encode_word_seq(seq, update_vocab=update_vocab)

    def encode_sentences(self, sentences: Iterable[str], update_vocab: bool = True) -> np.ndarray:
        res = []
        for s in sentences:
            res.append(self.encode_sentence(s, update_vocab=update_vocab))
        return np.array(res)
