#!/usr/bin/env python3.8

import io
import ctypes
import argparse
from copy import copy
from pathlib import Path
from functools import partial
from shutil import copytree, rmtree
from multiprocessing import cpu_count
from multiprocessing import RawArray
from multiprocessing.pool import Pool
from typing import Any, Callable, Dict, List, Tuple, Union, cast

import torch
import numpy as np
import torch.nn.functional as F
import pydensecrf.densecrf as dcrf
from PIL import Image
from tqdm import tqdm
from torch import einsum
from torch import Tensor
from torch.utils.data import DataLoader

from utils import map_, simplex, probs2class
from main import do_epoch
from dataloader import SliceDataset, PatientSampler, custom_collate, unnormalized_color_transform
from dataloader import png_transform, gt_transform


crf_vars = {}


def init_crf_vars(u_raw, img_raw, final_raw):
    crf_vars['u'] = u_raw
    crf_vars['img'] = img_raw
    crf_vars['final'] = final_raw


def setup(args, n_class: int) -> Tuple[Any, Any, Any, List[Callable], List[float]]:
    print("\n>>> Setting up")
    cpu: bool = args.cpu or not torch.cuda.is_available()
    device = torch.device("cpu") if cpu else torch.device("cuda")

    net_class = getattr(__import__('networks'), args.network)
    net = net_class(1, n_class).to(device)
    net.init_weights()
    net.to(device)

    optimizer: Any  # disable an error for the optmizer (ADAM and SGD not same type)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.l_rate, betas=(0.9, 0.99), amsgrad=False)

    nd: str = "wh"

    loss_fns: List[Callable] = []
    loss_class = getattr(__import__('losses'), "CrossEntropy")
    loss_fns.append(loss_class(**{'idc': [0, 1]}, nd=nd))

    loss_weights: List[float] = [1]

    return net, optimizer, device, loss_fns, loss_weights


def crf_post_process(i: int, B: int, W: int, H: int, K: int) -> None:
    u_np = np.frombuffer(crf_vars['u'], dtype=np.float32).reshape(B, K, H, W)
    img_np = np.frombuffer(crf_vars['img'], dtype=np.uint8).reshape(B, H, W, 3)
    final_np = np.frombuffer(crf_vars['final'], dtype=np.float32).reshape(B, K, H, W)

    U: np.ndarray = u_np[i]
    uintimage: np.ndarray = img_np[i]

    d = dcrf.DenseCRF2D(W, H, K)

    flat_U: np.ndarray = U.reshape((K, -1))
    d.setUnaryEnergy(flat_U)

    d.addPairwiseGaussian(sxy=1, compat=3)

    # Will add the edge sensitivity tomorrow
    # im = np.ascontiguousarray(np.rollaxis(uintimage, 0, 3), dtype=np.uint8)
    im = uintimage
    assert im.shape == (H, W, 3), im.shape
    d.addPairwiseBilateral(sxy=10, srgb=20, rgbim=im, compat=10)

    Q = d.inference(5)

    final_np[i] = np.array(Q).reshape(K, H, W)


class DeepCutDataset(SliceDataset):
    def __init__(self, root: str, train_mode: bool, K=2, in_memory: bool = False, crf_batch: int = 4,
                 debug=False, batch_size=1, img_size=(256, 256)) -> None:
        filenames: List[str] = map_(lambda p: str(p.name), Path(root, 'img').glob("*"))
        folders: List[Path] = [Path(root, "img"), Path(root, "gt"), Path(root, "box")]
        are_hots: List[bool] = [False, True, True]

        self.crf_batch: int = crf_batch
        self.train_mode: bool = train_mode

        self.img_size = tuple(img_size)
        self.batch_size = batch_size

        super().__init__(filenames, folders, are_hots,
                         K=K,
                         transforms=[png_transform, gt_transform, gt_transform],
                         bounds_generators=[(lambda *a: torch.zeros(K, 1, 2)) for _ in range(1)],
                         debug=debug, debug_size=100)

        # self.uint_transform = unnormalized_color_transform()
        self.orig_boxes: List = [copy(e) for e in self.files[2]]
        assert len(self.orig_boxes) == len(self.filenames), (len(self.orig_boxes), len(self.filenames))

        self.loader: DataLoader = DataLoader(self, shuffle=False, drop_last=False, pin_memory=True,
                                             collate_fn=custom_collate, batch_size=4)
        from torch.utils.data import SequentialSampler
        assert isinstance(self.loader.sampler, SequentialSampler), self.loader._index_sampler  # type: ignore

    def __getitem__(self, index: int) -> Dict[str, Union[str,
                                                         Tensor,
                                                         List[Tensor],
                                                         List[Tuple[slice, ...]],
                                                         List[Tuple[Tensor, Tensor]]]]:
        res = super().__getitem__(index)

        if self.train_mode:
            image = Image.open(self.files[0][index])  # Index 0 for image
            res["uint_images"] = unnormalized_color_transform((1, 1), self.K)(image)

            orig_box = Image.open(self.orig_boxes[index])
            res["orig_boxes"] = gt_transform((1, 1), self.K)(orig_box)

        return res

    def update_labels(self, net, device, axis: int = 2, savedir: str = "") -> None:
        print(f"> Updating {len(self)} labels...")
        assert self.train_mode

        # w_, h_ = self.img_size
        h_, w_ = self.img_size
        u_raw = RawArray(ctypes.c_float, self.batch_size * self.K * h_ * w_)
        u_np = np.frombuffer(u_raw, dtype=np.float32).reshape(self.batch_size, self.K, h_, w_)

        img_raw = RawArray(ctypes.c_char, self.batch_size * 3 * h_ * w_)
        img_np = np.frombuffer(img_raw, dtype=np.uint8).reshape(self.batch_size, h_, w_, 3)

        final_raw = RawArray(ctypes.c_float, self.batch_size * self.K * h_ * w_)
        final_np = np.frombuffer(final_raw, dtype=np.float32).reshape(self.batch_size, self.K, h_, w_)

        pool = Pool(self.batch_size, initializer=init_crf_vars, initargs=(u_raw, img_raw, final_raw))

        i = 0
        with torch.no_grad():
            for data in tqdm(self.loader):
                imgs: Tensor = data['images'].to(device)
                uint_images: Tensor = data['uint_images']
                orig_boxes: Tensor = data['orig_boxes']

                logits: Tensor = net(imgs)
                probs: Tensor = F.softmax(logits, dim=1)

                assert simplex(probs)

                final_probs = torch.zeros_like(probs)
                U: np.ndarray = (-probs.log()).cpu().numpy()
                assert U.dtype == np.float32
                # del probs

                B, K, W, H = U.shape
                u_np[:B, ...] = U[:B, ...]

                assert B <= self.batch_size
                assert K == self.K
                assert (W, H) == self.img_size, (W, H, self.img_size)
                assert uint_images.shape == (B, W, H, 3), (uint_images.shape, (B, W, H, 3))

                proc_fn: Callable = partial(crf_post_process, B=self.batch_size, W=W, H=H, K=K)

                uintimages: np.ndarray = uint_images.numpy()
                img_np[:B, ...] = uintimages[:B, ...]

                pool.map(proc_fn, range(B))

                final_probs = torch.tensor(final_np[:B, ...])
                assert simplex(final_probs)

                proposals: Tensor = probs2class(final_probs)
                assert proposals.shape == (B, W, H), proposals.shape
                assert orig_boxes.shape == (B, K, W, H), orig_boxes.shape
                # We do not want the proposals to overflow outside the box
                final_proposals: Tensor = einsum("bwh,bwh->bwh", orig_boxes[:, 1, :, :], proposals)

                # for b in range(B):
                #     im = imgs[b, 0].cpu().numpy()
                #     gt = data['gt'][b, 1].cpu().numpy()

                #     import matplotlib.pyplot as plt
                #     from mpl_toolkits.axes_grid1 import ImageGrid

                #     fig = plt.figure()
                #     fig.clear()

                #     grid = ImageGrid(fig, 211, nrows_ncols=(1, 5))

                #     grid[0].imshow(im, cmap="gray")
                #     grid[0].imshow(gt, cmap='jet', alpha=.5, vmax=1)

                #     grid[1].imshow(im, cmap="gray")
                #     grid[1].imshow(orig_boxes[b, 1].cpu().numpy(), cmap='jet', alpha=.5, vmax=1)

                #     grid[2].imshow(im, cmap="gray")
                #     grid[2].imshow(probs[b, 1].cpu().numpy(), cmap='jet', alpha=.5, vmax=1)

                #     grid[3].imshow(im, cmap="gray")
                #     grid[3].imshow(final_probs[b, 1], cmap='jet', alpha=.5, vmax=1)

                #     grid[4].imshow(im, cmap="gray")
                #     grid[4].imshow(final_proposals[b], cmap='jet', alpha=.5, vmax=1)
                #     plt.show()

                # And now the interesting part, replace live the stored images
                for b in range(B):
                    # for proposal in final_proposals:  # type: ignore
                    proposal = final_proposals[b]
                    assert proposal.shape == self.img_size

                    pil_img = Image.fromarray(proposal.cpu().numpy().astype('uint8'), mode='L')
                    buffer_ = io.BytesIO()
                    pil_img.save(buffer_, format='PNG')

                    if savedir:
                        pil_img.save(str(Path(savedir, data["filenames"][b]).with_suffix(".png")))

                    # print(buffer_)

                    self.files[axis][i] = buffer_
                    i += 1

        assert i == len(self)


def main() -> None:
    args: argparse.Namespace = get_args()

    n_class: int = args.n_class
    lr: float = args.l_rate
    savedir: str = args.workdir
    n_epoch: int = args.n_epoch

    loss_fns: List[Callable]
    loss_weights: List[float]
    net, optimizer, device, loss_fns, loss_weights = setup(args, n_class)

    train_loader: DataLoader
    val_loaders: DataLoader
    train_set = DeepCutDataset(str(Path(args.dataset, "train")), True, in_memory=args.in_memory,
                               batch_size=args.batch_size, debug=args.debug, img_size=args.img_size)
    val_set = DeepCutDataset(str(Path(args.dataset, "val")), False, in_memory=args.in_memory,
                             debug=args.debug, img_size=args.img_size)
    val_sampler = PatientSampler(val_set, args.grp_regex, shuffle=False)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              drop_last=False, pin_memory=not args.cpu,
                              num_workers=min(cpu_count(), args.batch_size + 5), collate_fn=custom_collate)
    val_loader = DataLoader(val_set, batch_sampler=val_sampler,
                            pin_memory=not args.cpu,
                            num_workers=min(cpu_count(), args.batch_size + 5), collate_fn=custom_collate)

    n_tra: int = sum(len(tr_lo.dataset) for tr_lo in [train_loader])  # Number of images in dataset
    l_tra: int = sum(len(tr_lo) for tr_lo in [train_loader])  # Number of iteration per epc: different if batch_size > 1
    n_val: int = sum(len(vl_lo.dataset) for vl_lo in [val_loader])
    l_val: int = sum(len(vl_lo) for vl_lo in [val_loader])
    n_loss: int = max(map(len, [loss_fns]))

    best_dice: Tensor = cast(Tensor, torch.zeros(1).type(torch.float32))
    best_epoch: int = 0
    metrics: Dict[str, Tensor] = {"val_dice": torch.zeros((n_epoch, n_val, n_class)).type(torch.float32),
                                  "val_loss": torch.zeros((n_epoch, l_val, len(loss_fns))).type(torch.float32),
                                  "tra_dice": torch.zeros((n_epoch, n_tra, n_class)).type(torch.float32),
                                  "tra_loss": torch.zeros((n_epoch, l_tra, n_loss)).type(torch.float32),
                                  "val_3d_dsc": torch.zeros((n_epoch, l_val, n_class)).type(torch.float32)}

    for i in range(n_epoch):
        # Do training and validation loops
        tra_loss, tra_dice, _, tra_mIoUs, _ = do_epoch("train", net, device, [train_loader], i,
                                                       [loss_fns], [loss_weights], n_class,
                                                       savedir=savedir if args.save_train else "",
                                                       optimizer=optimizer,
                                                       metric_axis=args.metric_axis)
        with torch.no_grad():
            val_loss, val_dice, val_hausdorff, val_mIoUs, val_3d_dsc = do_epoch("val", net, device, [val_loader], i,
                                                                                [loss_fns],
                                                                                [loss_weights],
                                                                                n_class,
                                                                                savedir=savedir,
                                                                                metric_axis=args.metric_axis,
                                                                                compute_3d_dice=True)
        # Sort and save the metrics
        for k in metrics:
            assert metrics[k][i].shape == eval(k).shape, (metrics[k][i].shape, eval(k).shape, k)
            metrics[k][i] = eval(k)

        for k, e in metrics.items():
            np.save(Path(savedir, f"{k}.npy"), e.cpu().numpy())

        # Update the training set labels and reinit network
        if i > 0 and not (i % 10):
            newlabels_dir = Path(savedir, f"iter{i:03d}", "newlabels")
            newlabels_dir.mkdir(exist_ok=True, parents=True)
            train_set.update_labels(net, device, savedir=str(newlabels_dir))
            net.init_weights()
            print(">>> Network reinitialized")

        # Save model if better
        current_dice: Tensor = val_dice[:, args.metric_axis].mean()
        if current_dice > best_dice:
            best_epoch = i
            best_dice = current_dice
            if val_hausdorff is not None:
                best_hausdorff = val_hausdorff[:, args.metric_axis].mean()
            if val_3d_dsc is not None:
                best_3d_dsc = val_3d_dsc[:, args.metric_axis].mean()

            with open(Path(savedir, "best_epoch.txt"), 'w') as f:
                f.write(str(i))
            best_folder = Path(savedir, "best_epoch")
            if best_folder.exists():
                rmtree(best_folder)
            copytree(Path(savedir, f"iter{i:03d}"), Path(best_folder))
            torch.save(net, Path(savedir, "best.pkl"))

        # if args.schedule and (i > (best_epoch + 20)):
        if args.schedule and (i % (best_epoch + 20) == 0):  # Yeah, ugly but will clean that later
            for param_group in optimizer.param_groups:
                lr *= 0.5
                param_group['lr'] = lr
                print(f'>> New learning Rate: {lr}')

        if i > 0 and not (i % 5):
            maybe_3d = f', 3d_DSC: {best_3d_dsc:.3f}'
            print(f">> Best results at epoch {best_epoch}: DSC: {best_dice:.3f}{maybe_3d}")

    # Because displaying the results at the end is actually convenient
    maybe_3d = f', 3d_DSC: {best_3d_dsc:.3f}'
    print(f">> Best results at epoch {best_epoch}: DSC: {best_dice:.3f}{maybe_3d}")
    for metric in metrics:
        if "val" in metric or "loss" in metric:  # Do not care about training values, nor the loss (keep it simple)
            print(f"\t{metric}: {metrics[metric][best_epoch].mean(dim=0)}")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument("--network", type=str, required=True, help="The network to use")
    parser.add_argument("--workdir", type=str, required=True)
    parser.add_argument("--n_class", type=int, required=True)
    parser.add_argument("--metric_axis", type=int, nargs='*', required=True, help="Classes to display metrics. \
        Display only the average of everything if empty")

    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--cpu", action='store_true')
    parser.add_argument("--in_memory", action='store_true')
    parser.add_argument("--save_train", action='store_true')
    parser.add_argument("--schedule", action='store_true')

    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--img_size', type=int, nargs=2, default=[256, 256])
    parser.add_argument('--n_epoch', nargs='?', type=int, default=200,
                        help='# of the epochs')
    parser.add_argument('--l_rate', nargs='?', type=float, default=5e-4,
                        help='Learning Rate')
    parser.add_argument("--grp_regex", type=str, default=None)

    args = parser.parse_args()
    print("\n", args)

    return args


if __name__ == "__main__":
    main()
