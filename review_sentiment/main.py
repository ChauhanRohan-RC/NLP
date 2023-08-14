import os.path

print("\n========================  Sentiment Analysis  ======================")
print("Loading Models...")

from utils import set_tensorflow_logging_level

set_tensorflow_logging_level(3)

from R import create_sentence_encoder, MODEL_REVIEW_POSITIVE_FILE_PATH, MODEL_REVIEW_RATING_FILE_PATH
from movie_review_rating_model import predict_ratings
from movie_review_positive_model import predict_postives
import keras

from tkinter import Tk, filedialog

model_positive = keras.models.load_model(MODEL_REVIEW_POSITIVE_FILE_PATH)
model_rating = keras.models.load_model(MODEL_REVIEW_RATING_FILE_PATH)
sentence_encoder = create_sentence_encoder(True)


def predict_rating_and_positive(review_: str):
    encoded_review = sentence_encoder.encode_sentence(review_, update_vocab=False)
    encoded_review.resize((1, *encoded_review.shape))  # wrap in an array

    pred_rating = predict_ratings(model_rating, encoded_review, encode=False)[0]
    pred_positive = predict_postives(model_positive, encoded_review, encode=False)[0]

    return pred_rating, pred_positive


def print_results(review_text: str):
    print(">> Analysing review...")
    rat, pos = predict_rating_and_positive(review_text)
    # print(f"Rating: {rat}/10")
    print(f">>> Sentiment: {'Positive' if pos else 'Negative'}\n")


def cmd_review_text():
    while True:
        review = input("AI (Enter Review) > ")
        if not review:
            continue

        if review in ('exit', 'quit', 'back'):
            break

        print_results(review)
        break


def cmd_review_file():
    print("> Please select a review file from the dialog...")
    file_path = filedialog.askopenfilename(initialdir="C;\\", title="Choose Review File",
                                           filetypes=(('Text File', '*.txt'), ('All files', '*.*')))

    if not file_path:
        print(">> INFO: No file selected!")
        return

    print(f">> Loading file '{os.path.basename(file_path)}'..")
    try:
        with open(file_path, 'r') as f:
            review_text = f.read()
    except Exception as e:
        print(f"> ERR: Failed to load file '{file_path}' -> {e}")
    else:
        print_results(review_text)


win = Tk()
win.withdraw()

print(f'\nCOMMANDS\n\texit/quit: exit console\n\ttext/review/r: enter a review on the console\n\tfile/f: input a review file\n')

while True:
    cmd = input("AI> ")
    if not cmd:
        continue
    elif cmd in ('exit', 'quit'):
        break
    elif cmd in ('file', 'f'):
        cmd_review_file()
    elif cmd in ('text', 'review', 'rev', 'r'):
        cmd_review_text()
    else:
        print(f"Invalid command {cmd}")

win.destroy()
quit()
