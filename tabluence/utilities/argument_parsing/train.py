import argparse


def get_train_parser() -> argparse.ArgumentParser:
    """
    Returns the argument parser for the train command.
    """
    parser = argparse.ArgumentParser("train")
    parser.add_argument(
        "config", help="Path to the config file.", type=str)
    parser.add_argument("--device", default=-1, help="Device to use.", type=int)
    parser.add_argument("--seed", default=42, help="Random seed.", type=int)
    parser.add_argument("--clean", action="store_true", help="Restart training and erase previous checkpoint contents.")
    parser.add_argument("--eval", action="store_true", help="Evaluate the model (the latest checkpoint must exist in "
                                                            "the folder, and it will be used.")
    return parser
