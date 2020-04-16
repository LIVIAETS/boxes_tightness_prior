#!/usr/bin/env python3.7

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np


def main(args) -> None:
    print(f"Reporting on {len(args.folders)} folders.")

    main_metric: str = args.metrics[0]

    best_epoch: List[int] = display_metric(args, main_metric, args.folders, args.axises)
    for metric in args.metrics[1:]:
        display_metric(args, metric, args.folders, args.axises, best_epoch)


def display_metric(args, metric: str, folders: List[str], axises: Tuple[int], best_epoch: List[int] = None):
    print(f"{metric} (classes {axises})")

    if not best_epoch:
        get_epoch = True
        best_epoch = [0] * len(folders)
    else:
        get_epoch = False

    for i, folder in enumerate(folders):
        file: Path = Path(folder, metric).with_suffix(".npy")
        data: np.ndarray = np.load(file)[:, :, axises]  # Epoch, sample, classes
        averages: np.ndarray = data.mean(axis=(1, 2))
        stds: np.ndarray = data.std(axis=(1, 2))

        if get_epoch:
            if args.mode == "max":
                best_epoch[i] = np.argmax(averages)
            elif args.mode == "min":
                best_epoch[i] = np.argmin(averages)

        val: float
        val_std: float
        if args.mode in ['max', 'min']:
            val = averages[best_epoch[i]]
            val_std = stds[best_epoch[i]]
        else:
            val = averages[-args.last_n_epc:].mean()
            val_std = averages[-args.last_n_epc:].std()

        print(f"\t{Path(folder).name}: {val:.{args.precision}f} ({val_std:.{args.precision}f}) at epoch {best_epoch[i]}")

    return best_epoch


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Plot data over time')
    parser.add_argument('--folders', type=str, required=True, nargs='+', help="The folders containing the file")
    parser.add_argument('--metrics', type=str, required=True, nargs='+')
    parser.add_argument('--axises', type=int, required=True, nargs='+')
    parser.add_argument('--mode', type=str, default='max', choices=['max', 'min', 'avg'])
    parser.add_argument('--last_n_epc', type=int, default=1)
    parser.add_argument('--precision', type=int, default=4)

    args = parser.parse_args()

    print(args)

    return args


if __name__ == "__main__":
    main(get_args())
