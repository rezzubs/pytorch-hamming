import argparse

from hamming_utils._data import MetaData
import torch

from hamming_utils import Data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--bit_error_rate", type=float, required=True)
    parser.add_argument("-i", "--iterations", type=int, required=True)
    parser.add_argument("-a", "--autosave", type=int, default=1, required=False)
    parser.add_argument("-d", "--data-path", type=str, default="./")
    parser.add_argument("-p", "--protected", action="store_true", required=False)
    parser.add_argument("-c", "--cuda", action="store_true", required=False)
    parser.add_argument(
        "--f16",
        help="Use 16 bit precision for the model instead of the default 32",
        action="store_true",
        required=False,
    )
    parser.add_argument(
        "--data-buffer-size",
        help="Chunk size for data to be encoded - power of two",
        type=int,
        default=64,
    )

    args = parser.parse_args()

    buffer_size = None if not args.protected else args.data_buffer_size
    dtype = "f32" if not args.f16 else "f16"
    data = Data.load(args.data_path or "./", MetaData(buffer_size, dtype))

    device_str = "cpu"
    if args.cuda:
        if torch.cuda.is_available():
            device_str = "cuda:0"
        else:
            print(f"Cuda is not available. Falling back to {device_str}")

    device = torch.device(device_str)

    data.record_n(
        args.iterations,
        args.bit_error_rate,
        args.protected,
        args.f16,
        args.data_path,
        autosave=args.autosave,
        device=device,
        data_buffer_size=args.data_buffer_size,
    )


if __name__ == "__main__":
    main()
