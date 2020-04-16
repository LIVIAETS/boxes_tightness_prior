#!/usr/env/bin python3.7

from functools import reduce
from operator import mul, add
from typing import List, Tuple, cast

import torch
import numpy as np
from torch import Tensor, einsum

from utils import simplex, one_hot


class CrossEntropy():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        self.nd: str = kwargs["nd"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, target: Tensor, _: Tensor, __) -> Tensor:
        assert simplex(probs) and simplex(target)

        log_p: Tensor = (probs[:, self.idc, ...] + 1e-10).log()
        mask: Tensor = cast(Tensor, target[:, self.idc, ...].type(torch.float32))

        loss = - einsum(f"bk{self.nd},bk{self.nd}->", mask, log_p)
        loss /= mask.sum() + 1e-10

        return loss


class AbstractConstraints():
    def __init__(self, **kwargs):
        self.idc: List[int] = kwargs["idc"]
        self.nd: str = kwargs["nd"]
        self.C = len(self.idc)
        self.__fn__ = getattr(__import__('utils'), kwargs['fn'])

        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def penalty(self, z: Tensor) -> Tensor:
        raise NotImplementedError

    def __call__(self, probs: Tensor, target: Tensor, bounds: Tensor, _) -> Tensor:
        assert simplex(probs)  # and simplex(target)  # Actually, does not care about second part
        assert probs.shape == target.shape

        # b, _, w, h = probs.shape  # type: Tuple[int, int, int, int]
        b: int
        b, _, *im_shape = probs.shape
        _, _, k, two = bounds.shape  # scalar or vector
        assert two == 2

        value: Tensor = cast(Tensor, self.__fn__(probs[:, self.idc, ...]))
        lower_b = bounds[:, self.idc, :, 0]
        upper_b = bounds[:, self.idc, :, 1]

        assert value.shape == (b, self.C, k), value.shape
        assert lower_b.shape == upper_b.shape == (b, self.C, k), lower_b.shape

        upper_z: Tensor = cast(Tensor, (value - upper_b).type(torch.float32)).flatten()
        lower_z: Tensor = cast(Tensor, (lower_b - value).type(torch.float32)).flatten()

        upper_penalty: Tensor = reduce(add, (self.penalty(e) for e in upper_z))
        lower_penalty: Tensor = reduce(add, (self.penalty(e) for e in lower_z))

        res: Tensor = upper_penalty + lower_penalty

        loss: Tensor = res.sum() / reduce(mul, im_shape)
        assert loss.requires_grad == probs.requires_grad  # Handle the case for validation

        return loss


class LogBarrierLoss(AbstractConstraints):
    def __init__(self, **kwargs):
        self.t: float = kwargs["t"]
        super().__init__(**kwargs)

    def penalty(self, z: Tensor) -> Tensor:
        assert z.shape == ()

        if z <= - 1 / self.t**2:
            return - torch.log(-z) / self.t
        else:
            return self.t * z + -np.log(1 / (self.t**2)) / self.t + 1 / self.t


class BoxPrior():
    def __init__(self, **kwargs):
        self.idc: List[int] = kwargs["idc"]

        self.t: float = kwargs["t"]

        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def barrier(self, z: Tensor) -> Tensor:
        assert z.shape == ()

        if z <= - 1 / self.t**2:
            return - torch.log(-z) / self.t
        else:
            return self.t * z + -np.log(1 / (self.t**2)) / self.t + 1 / self.t

    def __call__(self, probs: Tensor, _: Tensor, __: Tensor,
                 box_prior: List[List[Tuple[Tensor, Tensor]]]) -> Tensor:
        assert simplex(probs)

        B: int = probs.shape[0]
        assert len(box_prior) == B

        sublosses = []
        for b in range(B):
            for k in self.idc:
                masks, bounds = box_prior[b][k]

                sizes: Tensor = einsum('wh,nwh->n', probs[b, k], masks)

                assert sizes.shape == bounds.shape == (masks.shape[0],), (sizes.shape, bounds.shape, masks.shape)
                shifted: Tensor = bounds - sizes

                init = torch.zeros((), dtype=torch.float32, requires_grad=probs.requires_grad, device=probs.device)
                sublosses.append(reduce(add, (self.barrier(v) for v in shifted), init))

        loss: Tensor = reduce(add, sublosses)

        assert loss.dtype == torch.float32
        assert loss.shape == (), loss.shape

        return loss


class NegSizeLoss():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        self.t: float = kwargs["t"]
        self.nd: str = kwargs["nd"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def penalty(self, z: Tensor) -> Tensor:
        assert z.shape == ()

        if z <= - 1 / self.t**2:
            return - torch.log(-z) / self.t
        else:
            return self.t * z + -np.log(1 / (self.t**2)) / self.t + 1 / self.t

    def __call__(self, probs: Tensor, target: Tensor, _: Tensor, __) -> Tensor:
        assert simplex(probs) and simplex(target)

        b: int
        b, _, *im_shape = probs.shape

        probs_m: Tensor = probs[:, self.idc, ...]
        target_m: Tensor = cast(Tensor, target[:, self.idc, ...].type(torch.float32))

        nd: str = self.nd
        # Compute the size for each class, masked by the target pixels (where target ==1)
        masked_sizes: Tensor = einsum(f"bk{nd},bk{nd}->bk", probs_m, target_m).flatten()

        # We want that size to be <= so no shift is needed
        res: Tensor = reduce(add, (self.penalty(e) for e in masked_sizes))  # type: ignore

        loss: Tensor = res / reduce(mul, im_shape)
        assert loss.shape == ()
        assert loss.requires_grad == probs.requires_grad  # Handle the case for validation

        return loss
