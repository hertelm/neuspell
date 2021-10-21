import argparse

from neuspell import CnnlstmChecker, SclstmChecker, NestedlstmChecker, BertChecker, ElmosclstmChecker, \
    BertsclstmChecker, SclstmbertChecker, SclstmelmoChecker
from neuspell.corrector import Corrector


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


def interactive_sequences():
    while True:
        sequence = input("> ")
        yield sequence


def main(args):
    checker = load_checker(args.checker)
    if args.in_file:
        checker.correct_from_file(src=args.in_file, dest=args.out_file)
    else:
        for sequence in interactive_sequences():
            predicted = checker.correct(sequence)
            print(predicted)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", dest="checker", type=str,
                        choices=list(CHECKERS), required=True)
    parser.add_argument("-f", dest="in_file", type=str, default=None)
    parser.add_argument("-o", dest="out_file", type=str, default=None)
    args = parser.parse_args()
    main(args)
