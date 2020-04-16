#!/usr/bin/env python3.7

from typing import Any, Callable, List, Tuple, Union
from operator import add
from functools import partial

from utils import map_, uc_


class DummyScheduler(object):
    def __call__(self, epoch: int, optimizer: Any, loss_fns: List[List[Callable]], loss_weights: List[List[float]]) \
            -> Tuple[float, List[List[Callable]], List[List[float]]]:
        return optimizer, loss_fns, loss_weights


class MultiplyT():
    def __init__(self, target_loss: Union[str, List[str]], mu: float):
        if isinstance(target_loss, str):
            target_loss = [target_loss]
        self.target_loss: List[str] = target_loss
        self.mu: float = mu

    def __call__(self, epoch: int, optimizer: Any, loss_fns: List[List[Callable]], loss_weights: List[List[float]]) \
            -> Tuple[float, List[List[Callable]], List[List[float]]]:
        def update(loss: Any):
            if loss.__class__.__name__ in self.target_loss:
                loss.t *= self.mu

            return loss

        return optimizer, map_(lambda l: map_(update, l), loss_fns), loss_weights
