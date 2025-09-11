from hamming_utils import Data
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", metavar="DATA_PATH", type=str)

    args = parser.parse_args()

    Data.load(args.path).overview()


if __name__ == "__main__":
    main()
