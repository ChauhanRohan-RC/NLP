import numpy as np
from R import *

print("\n========================  Spam Detector  ======================")
print("Loading BERT Model...")
set_tensorflow_logging_level(2)
import bert_model
from tkinter import Tk, filedialog

model = bert_model.load_bert_model()


def predict_spam(msg_: str):
    pred = bert_model.predict_spams(model, np.array([msg_]))
    return pred[0]


def print_results(msg_: str):
    print(">> Analysing message...")
    is_spam = predict_spam(msg_)
    # print(f"Rating: {rat}/10")
    print(f">>> Prediction: {'SPAM' if is_spam else 'NOT A SPAM'}\n")


def cmd_msg_text():
    while True:
        msg_ = input("AI (Enter Message) > ")
        if not msg_:
            continue

        if msg_ in ('exit', 'quit', 'back'):
            break

        print_results(msg_)
        break


def cmd_msg_file():
    print("> Please select a message file from the dialog...")
    file_path = filedialog.askopenfilename(initialdir="C;\\", title="Choose Message File",
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

print(f'\nCOMMANDS\n\texit/quit: exit console\n\ttext/msg: enter a message on the console\n\tfile/f: input a message file\n')

while True:
    cmd = input("AI> ")
    if not cmd:
        continue
    elif cmd in ('exit', 'quit'):
        break
    elif cmd in ('file', 'f'):
        cmd_msg_file()
    elif cmd in ('text', 'msg', 'message'):
        cmd_msg_text()
    else:
        print(f"Invalid command {cmd}")

win.destroy()
quit()
