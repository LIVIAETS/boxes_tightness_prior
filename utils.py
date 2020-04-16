#!/usr/bin/env python3.7

import argparse
from pathlib import Path
from operator import add
from collections import namedtuple
from multiprocessing.pool import Pool
from random import random, uniform, randint
from functools import lru_cache, partial, reduce

from typing import Any, Callable, Iterable, List, Set, Tuple, TypeVar, Union, cast

import torch
import numpy as np
import scipy as sp
from tqdm import tqdm
from torch import einsum
from torch import Tensor
from skimage import measure
from skimage.io import imsave
from PIL import Image, ImageOps
from medpy.metric.binary import hd
from scipy.ndimage import distance_transform_edt as distance


colors = ["c", "r", "g", "b", "m", 'y', 'k', 'chartreuse', 'coral', 'gold', 'lavender',
          'silver', 'tan', 'teal', 'wheat', 'orchid', 'orange', 'tomato']

# functions redefinitions
tqdm_ = partial(tqdm, ncols=175,
                leave=False,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [' '{rate_fmt}{postfix}]')

A = TypeVar("A")
B = TypeVar("B")
T = TypeVar("T", Tensor, np.ndarray)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def map_(fn: Callable[[A], B], iter: Iterable[A]) -> List[B]:
    return list(map(fn, iter))


def mmap_(fn: Callable[[A], B], iter: Iterable[A]) -> List[B]:
    return Pool().map(fn, iter)


def uc_(fn: Callable) -> Callable:
    return partial(uncurry, fn)


def uncurry(fn: Callable, args: List[Any]) -> Any:
    return fn(*args)


def id_(x):
    return x


def flatten_(to_flat: Iterable[Iterable[A]]) -> List[A]:
    return [e for l in to_flat for e in l]


def flatten__(to_flat):
    if type(to_flat) != list:
        return [to_flat]

    return [e for l in to_flat for e in flatten__(l)]


def depth(e: List) -> int:
    """
    Compute the depth of nested lists
    """
    if type(e) == list and e:
        return 1 + depth(e[0])

    return 0


def compose(fns, init):
    return reduce(lambda acc, f: f(acc), fns, init)


def compose_acc(fns, init):
    return reduce(lambda acc, f: acc + [f(acc[-1])], fns, [init])


# f . g (x) := f(g(x))
def c(*fns: Callable[[Any], Any]) -> Callable[[Any], Any]:
    return reduce(lambda acc, fn: (lambda x: acc(fn(x))), fns, lambda x: x)


# fns
def soft_size(a: Tensor) -> Tensor:
    return torch.einsum("bk...->bk", a)[..., None]


def batch_soft_size(a: Tensor) -> Tensor:
    return torch.einsum("bk...->k", a)[..., None]


# Assert utils
def uniq(a: Tensor) -> Set:
    return set(torch.unique(a.cpu()).numpy())


def sset(a: Tensor, sub: Iterable) -> bool:
    return uniq(a).issubset(sub)


def eq(a: Tensor, b) -> bool:
    return torch.eq(a, b).all()


def simplex(t: Tensor, axis=1) -> bool:
    _sum = cast(Tensor, t.sum(axis).type(torch.float32))
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones)


def one_hot(t: Tensor, axis=1) -> bool:
    return simplex(t, axis) and sset(t, [0, 1])


# # Metrics and shitz
def meta_dice(sum_str: str, label: Tensor, pred: Tensor, smooth: float = 1e-8) -> Tensor:
    assert label.shape == pred.shape
    assert one_hot(label)
    assert one_hot(pred)

    inter_size: Tensor = einsum(sum_str, [intersection(label, pred)]).type(torch.float32)
    sum_sizes: Tensor = (einsum(sum_str, [label]) + einsum(sum_str, [pred])).type(torch.float32)

    dices: Tensor = (2 * inter_size + smooth) / (sum_sizes + smooth)

    return dices


dice_coef = partial(meta_dice, "bk...->bk")
dice_batch = partial(meta_dice, "bk...->k")  # used for 3d dice


def intersection(a: Tensor, b: Tensor) -> Tensor:
    assert a.shape == b.shape
    assert sset(a, [0, 1])
    assert sset(b, [0, 1])
    return a & b


def union(a: Tensor, b: Tensor) -> Tensor:
    assert a.shape == b.shape
    assert sset(a, [0, 1])
    assert sset(b, [0, 1])
    return a | b


def inter_sum(a: Tensor, b: Tensor) -> Tensor:
    return einsum("bk...->bk", intersection(a, b).type(torch.float32))


def union_sum(a: Tensor, b: Tensor) -> Tensor:
    return einsum("bk...->bk", union(a, b).type(torch.float32))


def hausdorff(preds: Tensor, target: Tensor, spacing: Tensor = None) -> Tensor:
    assert preds.shape == target.shape
    assert one_hot(preds)
    assert one_hot(target)

    B, K, *img_shape = preds.shape

    if not spacing:
        D: int = len(img_shape)
        spacing = torch.ones((B, D), dtype=torch.float32)

    assert spacing.shape == (B, len(img_shape))

    res = torch.zeros((B, K), dtype=torch.float32, device=preds.device)
    n_pred = preds.cpu().numpy()
    n_target = target.cpu().numpy()
    n_spacing = spacing.cpu().numpy()

    for b in range(B):
        # if K == 2:
        #     res[b, :] = hd(n_pred[b, 1], n_target[b, 1], voxelspacing=n_spacing[b])
        #     continue

        for k in range(K):
            if not n_pred[b, k].any() and not n_target[b, k].any():
                res[b, k] = 0
                continue
            elif not n_pred[b, k].any() or not n_target[b, k].any():
                res[b, k] = 0
                continue

            res[b, k] = hd(n_pred[b, k], n_target[b, k], voxelspacing=n_spacing[b])

    return res


def iIoU(pred: Tensor, target: Tensor) -> Tensor:
    IoUs = inter_sum(pred, target) / (union_sum(pred, target) + 1e-10)
    assert IoUs.shape == pred.shape[:2]

    return IoUs


# switch between representations
def probs2class(probs: Tensor) -> Tensor:
    b, _, *img_shape = probs.shape
    assert simplex(probs)

    res = probs.argmax(dim=1)
    assert res.shape == (b, *img_shape)

    return res


def class2one_hot(seg: Tensor, K: int) -> Tensor:
    # Breaking change but otherwise can't deal with both 2d and 3d
    # if len(seg.shape) == 3:  # Only w, h, d, used by the dataloader
    #     return class2one_hot(seg.unsqueeze(dim=0), K)[0]

    assert sset(seg, list(range(K))), (uniq(seg), K)

    b, *img_shape = seg.shape  # type: Tuple[int, ...]

    device = seg.device
    res = torch.zeros((b, K, *img_shape), dtype=torch.int32, device=device).scatter_(1, seg[:, None, ...], 1)

    assert res.shape == (b, K, *img_shape)
    assert one_hot(res)

    return res


def probs2one_hot(probs: Tensor) -> Tensor:
    _, K, *_ = probs.shape
    assert simplex(probs)

    res = class2one_hot(probs2class(probs), K)
    assert res.shape == probs.shape
    assert one_hot(res)

    return res


def one_hot2dist(seg: np.ndarray, resolution: Tuple[float, float, float] = None) -> np.ndarray:
    assert one_hot(torch.tensor(seg), axis=0)
    K: int = len(seg)

    res = np.zeros_like(seg)
    for k in range(K):
        posmask = seg[k].astype(np.bool)

        if posmask.any():
            negmask = ~posmask
            res[k] = distance(negmask, sampling=resolution) * negmask \
                - (distance(posmask, sampling=resolution) - 1) * posmask
        # The idea is to leave blank the negative classes
        # since this is one-hot encoded, another class will supervise that pixel

    return res


BoxCoords = namedtuple("BoxCoords", ["x", "y", "w", "h"])


# def one_hot2boxcoords(seg: Tensor) -> List[List[BoxCoords]]:
#     K, W, H = seg.shape

#     assert one_hot(seg, axis=0)

#     res: List[List[BoxCoords]] = []

#     for k in range(K):
#         class_coords = binary2boxcoords(seg[k])

#         res.append(class_coords.copy())
#     assert len(res) == K

#     return res


def binary2boxcoords(seg: Tensor) -> List[BoxCoords]:
    assert sset(seg, [0, 1])
    _, __ = seg.shape  # dirty way to ensure the 2d shape

    blobs: np.ndarray
    n_blob: int
    blobs, n_blob = measure.label(seg.cpu().numpy(), background=0, return_num=True)

    assert set(np.unique(blobs)) <= set(range(0, n_blob + 1)), np.unique(blobs)

    class_coords: List[BoxCoords] = []
    for b in range(1, n_blob + 1):
        blob_mask: np.ndarray = blobs == b

        assert blob_mask.dtype == np.bool, blob_mask.dtype
        # assert set(np.unique(blob_mask)) == set([0, 1])

        coords = np.argwhere(blob_mask)

        x1, y1 = coords.min(axis=0)
        x2, y2 = coords.max(axis=0)

        class_coords.append(BoxCoords(x1, y1, x2 - x1, y2 - y1))

    assert len(class_coords) == n_blob

    return class_coords


def boxcoords2masks_bounds(boxes: List[BoxCoords], shape: Tuple[int, int], d: int) -> Tuple[Tensor, Tensor]:
    '''
        For nested list, can just iterate over this function
    '''

    masks_list: List[Tensor] = []
    bounds_list: List[float] = []

    box: BoxCoords
    for box in boxes:
        for i in range(box.w // d):
            mask = torch.zeros(shape, dtype=torch.float32)
            mask[box.x + i * d:box.x + (i + 1) * d, box.y:box.y + box.h + 1] = 1
            masks_list.append(mask)
            bounds_list.append(d)

        if box.w % d:
            mask = torch.zeros(shape, dtype=torch.float32)
            mask[box.x + box.w - (box.w % d):box.x + box.w + 1, box.y:box.y + box.h + 1] = 1
            masks_list.append(mask)
            bounds_list.append(box.w % d)

        for j in range(box.h // d):
            mask = torch.zeros(shape, dtype=torch.float32)
            mask[box.x:box.x + box.w + 1, box.y + j * d:box.y + (j + 1) * d] = 1
            masks_list.append(mask)
            bounds_list.append(d)

        if box.h % d:
            mask = torch.zeros(shape, dtype=torch.float32)
            mask[box.x:box.x + box.w + 1, box.y + box.h - (box.h % d):box.y + box.h + 1] = 1
            masks_list.append(mask)
            bounds_list.append(box.h % d)

    bounds = torch.tensor(bounds_list, dtype=torch.float32) if bounds_list else torch.zeros((0,), dtype=torch.float32)
    masks = torch.stack(masks_list) if masks_list else torch.zeros((0, *shape), dtype=torch.float32)
    assert masks.shape == (len(masks_list), *shape)
    assert masks.dtype == torch.float32
    assert bounds.shape == (len(masks_list),)

    return masks, bounds


# def boxcoords2bounds(boxes: List[List[BoxCoords]], d: int) -> Tensor:
#     K: int = len(boxes)

#     __class_bounds: List[Tensor] = []

#     for k in range(K):
#         class_boxes: List[BoxCoords] = boxes[k]

#         box: BoxCoords
#         bounds: List[int] = []
#         for box in class_boxes:
#             for i in range(box.w // d):
#                 bounds.append(d)

#             if box.w % d:
#                 bounds.append(box.w % d)

#             for j in range(box.h // d):
#                 bounds.append(d)

#             if box.h % d:
#                 bounds.append(box.h % d)

#         class_bounds = torch.tensor(bounds)

#         assert class_bounds.dtype == torch.float32

#         __class_bounds.append(class_bounds)

#     N: int = max(e.shape[0] for e in __class_bounds)

#     res = torch.zeros((K, N), dtype=torch.float32)
#     for k in range(K):
#         n: int = __class_bounds[k].shape[0]
#         res[k, :n] = __class_bounds[k][:]

#     return res


# Misc utils
def save_images(segs: Tensor, names: Iterable[str], root: str, mode: str, iter: int) -> None:
    for seg, name in zip(segs, names):
        save_path = Path(root, f"iter{iter:03d}", mode, name).with_suffix(".png")
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if len(seg.shape) == 2:
            imsave(str(save_path), seg.cpu().numpy().astype(np.uint8))
        elif len(seg.shape) == 3:
            np.save(str(save_path), seg.cpu().numpy())
        else:
            raise ValueError("How did you get here")


def augment(*arrs: Union[np.ndarray, Image.Image], rotate_angle: float = 45,
            flip: bool = True, mirror: bool = True,
            rotate: bool = True, scale: bool = False) -> List[Image.Image]:
    imgs: List[Image.Image] = map_(Image.fromarray, arrs) if isinstance(arrs[0], np.ndarray) else list(arrs)

    if flip and random() > 0.5:
        imgs = map_(ImageOps.flip, imgs)
    if mirror and random() > 0.5:
        imgs = map_(ImageOps.mirror, imgs)
    if rotate and random() > 0.5:
        angle: float = uniform(-rotate_angle, rotate_angle)
        imgs = map_(lambda e: e.rotate(angle), imgs)
    if scale and random() > 0.5:
        scale_factor: float = uniform(1, 1.2)
        w, h = imgs[0].size  # Tuple[int, int]
        nw, nh = int(w * scale_factor), int(h * scale_factor)  # Tuple[int, int]

        # Resize
        imgs = map_(lambda i: i.resize((nw, nh)), imgs)

        # Now need to crop to original size
        bw, bh = randint(0, nw - w), randint(0, nh - h)  # Tuple[int, int]

        imgs = map_(lambda i: i.crop((bw, bh, bw + w, bh + h)), imgs)
        assert all(i.size == (w, h) for i in imgs)

    return imgs


def augment_arr(*arrs_a: np.ndarray, rotate_angle: float = 45,
                flip: bool = True, mirror: bool = True,
                rotate: bool = True, scale: bool = False,
                noise: bool = False, noise_loc: float = 0.5, noise_lambda: float = 0.1) -> List[np.ndarray]:
    arrs = list(arrs_a)  # manoucherie type check

    if flip and random() > 0.5:
        arrs = map_(np.flip, arrs)
    if mirror and random() > 0.5:
        arrs = map_(np.fliplr, arrs)
    if noise and random() > 0.5:
        mask: np.ndarray = np.random.laplace(noise_loc, noise_lambda, arrs[0].shape)
        arrs = map_(partial(add, mask), arrs)
        arrs = map_(lambda e: (e - e.min()) / (e.max() - e.min()), arrs)
    # if random() > 0.5:
    #     orig_shape = arrs[0].shape

    #     angle = random() * 90 - 45
    #     arrs = map_(lambda e: sp.ndimage.rotate(e, angle, order=1), arrs)

    #     arrs = get_center(orig_shape, *arrs)

    return arrs


def get_center(shape: Tuple, *arrs: np.ndarray) -> List[np.ndarray]:
    """ center cropping """
    def g_center(arr):
        if arr.shape == shape:
            return arr

        offsets: List[int] = [(arrs - s) // 2 for (arrs, s) in zip(arr.shape, shape)]

        if 0 in offsets:
            return arr[[slice(0, s) for s in shape]]

        res = arr[[slice(d, -d) for d in offsets]][[slice(0, s) for s in shape]]  # Deal with off-by-one errors
        assert res.shape == shape, (res.shape, shape, offsets)

        return res

    return [g_center(arr) for arr in arrs]


def center_pad(arr: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
    assert len(arr.shape) == len(target_shape)

    diff: List[int] = [(nx - x) for (x, nx) in zip(arr.shape, target_shape)]
    pad_width: List[Tuple[int, int]] = [(w // 2, w - (w // 2)) for w in diff]

    res = np.pad(arr, pad_width)
    assert res.shape == target_shape, (res.shape, target_shape)

    return res
