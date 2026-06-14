# Dependencies imports
import torch
import torch.nn.functional as F
from typing import List, Tuple

# Division of full volume into sub-cubes
def split_volume(
    volume: torch.Tensor,
    sub_size: int = 64,
) -> Tuple[List[torch.Tensor], List[Tuple[int, int, int]]]:

    b, c, d, h, w = volume.shape
    assert b == 1, f"split_volume expects batch size 1, got {b}"
    assert d % sub_size == 0, f"Depth {d} not divisible by sub_size {sub_size}"
    assert h % sub_size == 0, f"Height {h} not divisible by sub_size {sub_size}"
    assert w % sub_size == 0, f"Width {w} not divisible by sub_size {sub_size}"

    sub_cubes = []
    coords = []

    for d_start in range(0, d, sub_size):
        for h_start in range(0, h, sub_size):
            for w_start in range(0, w, sub_size):
                sub_cube = volume[
                    :, :,
                    d_start : d_start + sub_size,
                    h_start : h_start + sub_size,
                    w_start : w_start + sub_size,
                ]
                sub_cubes.append(sub_cube)
                coords.append((d_start, h_start, w_start))

    return sub_cubes, coords

# Reassemble the sub-cubes into the full volume
def reassemble_volume(
    sub_preds: List[torch.Tensor],
    coords: List[Tuple[int, int, int]],
    full_size: Tuple[int, int, int],
    sub_size: int = 64,
) -> torch.Tensor:

    b, nclass = sub_preds[0].shape[0], sub_preds[0].shape[1]
    d, h, w = full_size

    full_pred = torch.zeros(
        b, nclass, d, h, w,
        dtype=sub_preds[0].dtype,
        device=sub_preds[0].device,
    )

    for sub_pred, (d_start, h_start, w_start) in zip(sub_preds, coords):
        full_pred[
            :, :,
            d_start : d_start + sub_size,
            h_start : h_start + sub_size,
            w_start : w_start + sub_size,
        ] = sub_pred

    return full_pred