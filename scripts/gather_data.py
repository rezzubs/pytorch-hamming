import argparse

from hamming_utils import Data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--bit_error_rate", type=float, required=True)
    parser.add_argument("-i", "--iterations", type=int, required=True)
    parser.add_argument("-a", "--autosave", type=int, default=1, required=False)
    parser.add_argument("-d", "--data_path", type=str, required=False)
    parser.add_argument("-p", "--protected", action="store_true", required=False)

    args = parser.parse_args()

    data = Data.load(args.data_path or "./")

    data.record_n(
        args.iterations, args.bit_error_rate, args.protected, autosave=args.autosave
    )


if __name__ == "__main__":
    main()
