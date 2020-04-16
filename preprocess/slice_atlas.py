#!/usr/bin/env python3.7


import random
import argparse
import warnings
from pathlib import Path
from functools import partial
from typing import Callable, List, Tuple

import numpy as np
import nibabel as nib
from skimage.io import imsave
from numpy import unique as uniq

from utils import map_, mmap_, center_pad, augment


def norm_arr(img: np.ndarray) -> np.ndarray:
    casted = img.astype(np.float32)
    shifted = casted - casted.min()
    norm = shifted / shifted.max()
    res = 255 * norm

    assert 0 == res.min(), res.min()
    assert res.max() == 255, res.max()

    return res.astype(np.uint8)


def fuse_labels(t1: np.ndarray, id_: str, acq: Path, nib_obj) -> np.ndarray:
    gt: np.ndarray = np.zeros_like(t1, dtype=np.uint8)
    gt1: np.ndarray = np.zeros_like(t1, dtype=np.uint8)
    assert gt.dtype == np.uint8

    labels: List[Path] = list(acq.glob(f"{id_}_LesionSmooth_*stx.nii.gz"))
    assert len(labels) >= 1, (acq, id_)

    label_path: Path
    label: np.ndarray
    for label_path in labels:
        label_obj = nib.load(str(label_path))
        label = np.asarray(label_obj.dataobj)

        assert sanity_label(label, t1, label_obj.header.get_zooms(), nib_obj.header.get_zooms(), label_path)

        binary_label: np.ndarray = (label > 0).astype(np.uint8)
        binary_label1: np.ndarray = (label > 1).astype(np.uint8)
        assert binary_label.dtype == np.uint8, binary_label.dtype
        assert set(uniq(binary_label)) <= set([0, 1])

        gt |= binary_label  # logical OR if labels overlap
        gt1 |= binary_label1  # logical OR if labels overlap
        # gt += binary_label
    assert set(uniq(gt)) <= set([0, 1])
    assert gt.dtype == np.uint8

    return gt, gt1


def sanity_t1(t1, x, y, z, dx, dy, dz) -> bool:
    assert t1.dtype in [np.float32], t1.dtype
    assert -0.0003 <= t1.min(), t1.min()
    assert t1.max() <= 100.0001, t1.max()

    assert 1 <= dx <= 1, dx
    assert 1 <= dy <= 1, dy
    assert 1 <= dz <= 1, dz

    assert x != y, (x, y)
    assert x != z or y != z, (x, y, z)
    assert x in [197], x
    assert y in [233], y
    assert z in [189], z

    return True


def sanity_label(label, t1, resolution, t1_resolution, label_path) -> bool:
    # assert False
    assert label.shape == t1.shape
    assert resolution == t1_resolution

    assert label.dtype in [np.float64], label.dtype

    # print(str(label_path))
    # if "31898" in str(label_path):
    #     print(label_path, uniq(label))

    # > 0 means disease
    labels_allowed = [[0.0, 0.9999999997671694],
                      [0., 254.9999999406282],
                      [0., 0.9999999997671694, 253.99999994086102, 254.9999999406282],
                      [0.0, 0.9999999997671694, 1.9999999995343387, 252.99999994109385, 253.99999994086102, 254.9999999406282]]

    # assert set(uniq(label)) in set(labels_allowed), (set(uniq(label)), label_path)
    matches: List[bool] = [set(uniq(label)) == set(allowed) for allowed in labels_allowed]
    assert any(matches), (set(uniq(label)), label_path)

    return True


def slice_patient(id_: str, dest_path: Path, source_path: Path, shape: Tuple[int, int],
                  n_augment: int):
    id_path: Path = Path(source_path, id_)

    for acq in id_path.glob("t0*"):
        acq_id: int = int(acq.name[1:])
        # print(id, acq, acq_id)

        t1_path: Path = Path(acq, f"{id_}_t1w_deface_stx.nii.gz")
        nib_obj = nib.load(str(t1_path))
        t1: np.ndarray = np.asarray(nib_obj.dataobj)
        # dx, dy, dz = nib_obj.header.get_zooms()
        x, y, z = t1.shape

        assert sanity_t1(t1, *t1.shape, *nib_obj.header.get_zooms())

        # gt: np.ndarray = fuse_labels(t1, id_, acq, nib_obj)
        gt, gt1 = fuse_labels(t1, id_, acq, nib_obj)

        norm_img: np.ndarray = norm_arr(t1)

        for idz in range(z):
            padded_img: np.ndarray = center_pad(norm_img[:, :, idz], shape)
            padded_gt: np.ndarray = center_pad(gt[:, :, idz], shape)
            padded_gt1: np.ndarray = center_pad(gt1[:, :, idz], shape)
            assert padded_img.shape == padded_gt.shape == shape

            for k in range(n_augment + 1):
                arrays: List[np.ndarray] = [padded_img, padded_gt, padded_gt1]

                augmented_arrays: List[np.ndarray]
                if k == 0:
                    augmented_arrays = arrays[:]
                else:
                    augmented_arrays = map_(np.asarray, augment(*arrays))

                subfolders: List[str] = ["img", "gt", "gt1"]
                assert len(augmented_arrays) == len(subfolders)
                for save_subfolder, data in zip(subfolders,
                                                augmented_arrays):
                    filename = f"{id_}_{acq_id}_{idz}_{k}.png"

                    save_path: Path = Path(dest_path, save_subfolder)
                    save_path.mkdir(parents=True, exist_ok=True)

                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=UserWarning)
                        imsave(str(Path(save_path, filename)), data)


def get_splits(id_list: str, retains: int, fold: int) -> Tuple[List[str], List[str]]:
    id_file: Path = Path(id_list)

    ids: List[str]
    with open(id_file, 'r') as f:
        ids = f.read().split()

    print(f"Founds {len(ids)} in the id list")
    assert len(ids) > retains

    random.shuffle(ids)  # Shuffle before to avoid any problem if the patients are sorted in any way
    validation_slice = slice(fold * retains, (fold + 1) * retains)
    validation_ids: List[str] = ids[validation_slice]
    assert len(validation_ids) == retains

    training_ids: List[str] = [e for e in ids if e not in validation_ids]
    assert (len(training_ids) + len(validation_ids)) == len(ids)

    return training_ids, validation_ids


def main(args: argparse.Namespace):
    src_path: Path = Path(args.source_dir)
    dest_path: Path = Path(args.dest_dir)

    # Assume the cleaning up is done before calling the script
    assert src_path.exists()
    assert not dest_path.exists()

    training_ids: List[str]
    validation_ids: List[str]
    training_ids, validation_ids = get_splits(args.id_list, args.retains, args.fold)

    split_ids: List[str]
    for mode, split_ids in zip(["train", "val"], [training_ids, validation_ids]):
        dest_mode: Path = Path(dest_path, mode)
        print(f"Slicing {len(split_ids)} pairs to {dest_mode}")

        pfun: Callable = partial(slice_patient,
                                 dest_path=dest_mode,
                                 source_path=src_path,
                                 shape=tuple(args.shape),
                                 n_augment=args.n_augment if mode == "train" else 0)
        mmap_(pfun, split_ids)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Slicing parameters')
    parser.add_argument('--source_dir', type=str, required=True)
    parser.add_argument('--dest_dir', type=str, required=True)
    parser.add_argument('--id_list', type=str, required=True)

    # parser.add_argument('--img_dir', type=str, default="IMG")
    # parser.add_argument('--gt_dir', type=str, default="GT")
    parser.add_argument('--shape', type=int, nargs="+", default=[256, 256])
    parser.add_argument('--retains', type=int, default=25, help="Number of retained patient for the validation data")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--n_augment', type=int, default=0,
                        help="Number of augmentation to create per image, only for the training set")
    args = parser.parse_args()
    random.seed(args.seed)

    print(args)

    return args


if __name__ == "__main__":
    main(get_args())
