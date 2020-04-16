#!/usr/bin/env python3.8

import re
import argparse
from pathlib import Path
from functools import partial
from multiprocessing import cpu_count
from typing import Dict, List, Match, Pattern

import torch
import numpy as np
from tqdm import tqdm
from torch import Tensor
from torch.utils.data import DataLoader
from medpy.metric.binary import hd

from utils import map_
from utils import dice_batch
from dataloader import SliceDataset, PatientSampler, custom_collate
from dataloader import png_transform, gt_transform


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Compute metrics over time on saved predictions')
    parser.add_argument('--basefolder', type=str, required=True, help="The folder containing the predicted epochs")
    parser.add_argument('--gt_folder', type=str)
    parser.add_argument('--metrics', type=str, nargs='+', required=True)
    parser.add_argument("--grp_regex", type=str, required=True)
    parser.add_argument('--num_classes', type=int, required=True)
    parser.add_argument("--debug", action="store_true", help="Dummy for compatibility")

    parser.add_argument("--n_epoch", type=int, default=-1)
    args = parser.parse_args()

    print(args)

    return args


def main() -> None:
    args = get_args()

    iterations_paths: List[Path] = sorted(Path(args.basefolder).glob("iter*"))
    # print(iterations_paths)
    print(f">>> Found {len(iterations_paths)} epoch folders")

    # Handle gracefully if not all folders are there (early stop)
    EPC: int = args.n_epoch if args.n_epoch >= 0 else len(iterations_paths)
    K: int = args.num_classes

    # Get the patient number, and image names, from the GT folder
    gt_path: Path = Path(args.gt_folder)
    names: List[str] = map_(lambda p: str(p.name), gt_path.glob("*"))
    n_img: int = len(names)

    grouping_regex: Pattern = re.compile(args.grp_regex)
    stems: List[str] = [Path(filename).stem for filename in names]  # avoid matching the extension
    matches: List[Match] = map_(grouping_regex.match, stems)  # type: ignore
    patients: List[str] = [match.group(1) for match in matches]

    unique_patients: List[str] = list(set(patients))
    n_patients: int = len(unique_patients)

    print(f">>> Found {len(unique_patients)} unique patients out of {n_img} images ; regex: {args.grp_regex}")
    # from pprint import pprint
    # pprint(unique_patients)

    # First, quickly assert all folders have the same numbers of predited images
    n_img_epoc: List[int] = [len(list(Path(p, "val").glob("*.png"))) for p in iterations_paths]
    assert len(set(n_img_epoc)) == 1
    assert all(len(list(Path(p, "val").glob("*.png"))) == n_img for p in iterations_paths)

    metrics: Dict['str', Tensor] = {}
    if '3d_dsc' in args.metrics:
        metrics['3d_dsc'] = torch.zeros((EPC, n_patients, K), dtype=torch.float32)
        print(f">> Will compute {'3d_dsc'} metric")
    if '3d_hausdorff' in args.metrics:
        metrics['3d_hausdorff'] = torch.zeros((EPC, n_patients, K), dtype=torch.float32)
        print(f">> Will compute {'3d_hausdorff'} metric")

    gen_dataset = partial(SliceDataset,
                          transforms=[png_transform, gt_transform, gt_transform],
                          are_hots=[False, True, True],
                          K=K,
                          in_memory=False,
                          bounds_generators=[(lambda *a: torch.zeros(K, 1, 2)) for _ in range(1)],
                          box_prior=False,
                          box_priors_arg='{}',
                          dimensions=2)
    data_loader = partial(DataLoader,
                          num_workers=cpu_count(),
                          pin_memory=False,
                          collate_fn=custom_collate)

    # Will replace live dataset.folders and call again load_images to update dataset.files
    print(gt_path, gt_path, Path(iterations_paths[0], 'val'))
    dataset: SliceDataset = gen_dataset(names, [gt_path, gt_path, Path(iterations_paths[0], 'val')])
    sampler: PatientSampler = PatientSampler(dataset, args.grp_regex, shuffle=False)
    dataloader: DataLoader = data_loader(dataset, batch_sampler=sampler)

    current_path: Path
    for e, current_path in enumerate(iterations_paths):
        dataset.folders = [gt_path, gt_path, Path(current_path, 'val')]
        dataset.files = SliceDataset.load_images(dataset.folders, dataset.filenames, False)

        print(f">>> Doing epoch {str(current_path)}")

        for i, data in enumerate(tqdm(dataloader, leave=None)):
            target: Tensor = data["gt"]
            prediction: Tensor = data["labels"][0]

            assert target.shape == prediction.shape

            if '3d_dsc' in args.metrics:
                dsc: Tensor = dice_batch(target, prediction)
                assert dsc.shape == (K,)

                metrics['3d_dsc'][e, i, :] = dsc
            if '3d_hausdorff' in args.metrics:
                np_pred: np.ndarray = prediction[:, 1, :, :].cpu().numpy()
                np_target: np.ndarray = target[:, 1, :, :].cpu().numpy()

                if np_pred.sum() > 0:
                    hd_: float = hd(np_pred, np_target)

                    metrics["3d_hausdorff"][e, i, 1] = hd_
                else:
                    x, y, z = np_pred.shape
                    metrics["3d_hausdorff"][e, i, 1] = (x**2 + y**2 + z**2)**0.5

        for metric in args.metrics:
            # For now, hardcode the fact we care about class 1 only
            print(f">> {metric}: {metrics[metric][e].mean(dim=0)[1]:.04f}")

    k: str
    el: Tensor
    for k, el in metrics.items():
        np.save(Path(args.basefolder, f"val_{k}.npy"), el.cpu().numpy())


if __name__ == '__main__':
    main()
