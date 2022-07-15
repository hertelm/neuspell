import argparse

from neuspell import CnnlstmChecker, SclstmChecker, NestedlstmChecker, BertChecker, ElmosclstmChecker, \
    BertsclstmChecker, SclstmbertChecker, SclstmelmoChecker
from neuspell.corrector import Corrector
from neuspell.seq_modeling.helpers import bert_tokenize_for_valid_examples
from neuspell.seq_modeling.helpers import sclstm_tokenize
from neuspell.commons import spacy_tokenizer


CHECKERS = {
    "CnnlstmChecker": CnnlstmChecker,
    "SclstmChecker": SclstmChecker,
    "NestedlstmChecker": NestedlstmChecker,
    "BertChecker": BertChecker,
    "ElmosclstmChecker": ElmosclstmChecker,
    "BertsclstmChecker": BertsclstmChecker,
    "SclstmbertChecker": SclstmbertChecker,
    "SclstmelmoChecker": SclstmelmoChecker
}


def load_checker(checker_name) -> Corrector:
    checker_class = CHECKERS[checker_name]
    checker = checker_class()
    checker.from_pretrained()
    return checker


def read_file(path):
    with open(path) as f:
        lines = f.read().splitlines()
    return lines


def interactive_sequences():
    while True:
        sequence = input("> ")
        yield sequence


def tokenize(checker, sequence):
    if isinstance(checker, BertChecker):
        tokens = bert_tokenize_for_valid_examples([sequence], [sequence])[0][0].split()
    else:
        tokens = [token for token in spacy_tokenizer(sequence).split() if len(token) > 0]
    return tokens


def postprocess(sequence, prediction):
    tokens = sequence.split()
    predicted_tokens = prediction.split()
    for i, token in enumerate(tokens):
        if "'" in token:
            predicted_tokens[i] = token
    return " ".join(predicted_tokens)


def predict(checker, sequence):
    tokens = tokenize(checker, sequence)
    space_positions = set()
    pos = 0
    for i, token in enumerate(tokens):
        if pos < len(sequence) and sequence[pos] == ' ':
            space_positions.add(i)
            pos += 1
        t_pos = 0
        while t_pos < len(token) and pos < len(sequence):
            if sequence[pos] != ' ':
                t_pos += 1
            pos += 1
    result = checker.correct(sequence)
    result_tokens = result.split()
    result_sequence = ""
    for i, token in enumerate(result_tokens):
        if i in space_positions:
            result_sequence += ' '
        result_sequence += token
    result_sequence = postprocess(sequence, result_sequence)
    return result_sequence


def main(args):
    checker = load_checker(args.checker)
    if args.in_file:
        sequences = read_file(args.in_file)
    else:
        sequences = interactive_sequences()
    if args.out_file:
        out_file = open(args.out_file, "w")
    else:
        out_file = None
    for sequence in sequences:
        predicted = predict(checker, sequence)
        print(predicted)
        if out_file:
            out_file.write(predicted)
            out_file.write("\n")
    if out_file:
        out_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", dest="checker", type=str,
                        choices=list(CHECKERS), required=True)
    parser.add_argument("-f", dest="in_file", type=str, default=None)
    parser.add_argument("-o", dest="out_file", type=str, default=None)
    args = parser.parse_args()
    main(args)
