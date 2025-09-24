import argparse

from hamming_utils._data import MetaData
import torch

from hamming_utils import Data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bit_error_rate", type=float, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--autosave", type=int, default=1, required=False)
    parser.add_argument("--output-path", type=str, default="./")
    parser.add_argument("--cuda", action="store_true", required=False)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument(
        "--f16",
        help="Use 16 bit precision for the model instead of the default 32",
        action="store_true",
        required=False,
    )
    parser.add_argument(
        "--protected-buffer-size",
        help="Chunk size for data to be encoded - power of two",
        type=int,
        required=False,
    )

    args = parser.parse_args()

    dtype = "f32" if not args.f16 else "f16"
    data = Data.load(
        args.output_path or "./",
        MetaData(args.protected_buffer_size, dtype, args.model, args.dataset),
    )

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
        args.f16,
        args.output_path,
        args.protected_buffer_size,
        autosave=args.autosave,
        device=device,
    )


if __name__ == "__main__":
    main()
