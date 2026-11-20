"""Command-line entry for LingSante-Pulse CGM workflows."""

import argparse
import importlib
import sys

COMMANDS = {
    "extract": "inference.pipeline",
    "train": "training.public",
    "evaluate": "evaluation.fixed",
    "evaluate-nested": "evaluation.nested",
    "downstream": "downstream",
    "export": "checkpoints",
    "prepare-data": "data.builders.public_pretraining_corpus",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    sys.argv = [parser.prog + " " + args.command, *args.arguments]
    importlib.import_module("lingsante_pulse.cgm." + COMMANDS[args.command]).main()


if __name__ == "__main__":
    main()
