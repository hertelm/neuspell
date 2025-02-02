import argparse
import time

from neuspell import BertChecker, SclstmChecker
from neuspell.corrector import Corrector
from neuspell.seq_modeling.helpers import bert_tokenize_for_valid_examples
from neuspell.seq_modeling.helpers import sclstm_tokenize
from neuspell.commons import spacy_tokenizer


CHECKERS = {
    "BertChecker",
    "SclstmelmoChecker"
}


def load_checker(checker_name) -> Corrector:
    if checker_name == "BertChecker":
        checker = BertChecker()
    else:
        checker = SclstmChecker()
        checker = checker.add_("elmo", at="output")
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


def postprocess_sequence(sequence, prediction):
    tokens = sequence.split()
    predicted_tokens = prediction.split()
    for i, token in enumerate(tokens):
        for j in range(1, len(token) - 1):
            if token[j] == "'" or token[j] == " ":
                predicted_tokens[i] = token
                break
    return " ".join(predicted_tokens)


def preprocess_tokens(sequence, tokens):
    pos = 0
    for t_i, token in enumerate(tokens):
        for c_i, char in enumerate(token):
            if sequence[pos] == " ":
                tokens[t_i] = tokens[t_i][:c_i] + " " + tokens[t_i][c_i:]
                pos += 1
            pos += 1
        if pos < len(sequence) and sequence[pos] == " ":
            pos += 1
    return tokens


def postprocess_tokens(tokens, predicted_tokens):
    for i, token in enumerate(tokens):
        if not token.isalpha():
            predicted_tokens[i] = token
    return predicted_tokens


def predict(checker, sequence):
    tokens = tokenize(checker, sequence)
    tokens = preprocess_tokens(sequence, tokens)
    space_positions = set()
    pos = 0
    for i, token in enumerate(tokens):
        if pos < len(sequence) and sequence[pos] == ' ':
            space_positions.add(i)
            pos += 1
        pos += len(token)
    result = checker.correct(sequence)
    result_tokens = result.split()
    postprocess_tokens(tokens, result_tokens)
    result_sequence = ""
    for i, token in enumerate(result_tokens):
        if i in space_positions:
            result_sequence += ' '
        result_sequence += token
    result_sequence = postprocess_sequence(sequence, result_sequence)
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
    total_runtime = 0
    for sequence in sequences:
        start_time = time.time()
        predicted = predict(checker, sequence)
        total_runtime += time.time() - start_time
        print(predicted)
        if out_file:
            out_file.write(predicted)
            out_file.write("\n")
    if out_file:
        out_file.write(str(total_runtime) + "\n")
        out_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", dest="checker", type=str,
                        choices=list(CHECKERS), required=True)
    parser.add_argument("-f", dest="in_file", type=str, default=None)
    parser.add_argument("-o", dest="out_file", type=str, default=None)
    args = parser.parse_args()
    main(args)
