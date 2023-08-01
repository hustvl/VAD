# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>

"""This module evaluates the forecasted trajectories against the ground truth."""

import math
from typing import Dict, List, Optional

import numpy as np
import torch

LOW_PROB_THRESHOLD_FOR_METRICS = 0.05


def get_ade(forecasted_trajectory: torch.Tensor, gt_trajectory: torch.Tensor) -> float:
    """Compute Average Displacement Error.
    Args:
        forecasted_trajectory: Predicted trajectory with shape [fut_ts, 2]
        gt_trajectory: Ground truth trajectory with shape [fut_ts, 2]
    Returns:
        ade: Average Displacement Error
    """
    pred_len = forecasted_trajectory.shape[0]
    ade = float(
        sum(
            torch.sqrt(
                (forecasted_trajectory[i, 0] - gt_trajectory[i, 0]) ** 2
                + (forecasted_trajectory[i, 1] - gt_trajectory[i, 1]) ** 2
            )
            for i in range(pred_len)
        )
        / pred_len
    )
    return ade

def get_best_preds(
    forecasted_trajectory: torch.Tensor,
    gt_trajectory: torch.Tensor
) -> float:
    """Compute min Average Displacement Error.
    Args:
        forecasted_trajectory: Predicted trajectory with shape [k, fut_ts, 2]
        gt_trajectory: Ground truth trajectory with shape [fut_ts, 2]
        gt_fut_masks: Ground truth traj mask with shape (fut_ts)
    Returns:
        best_forecasted_trajectory: Predicted trajectory with shape [fut_ts, 2]
    """

    # [k, fut_ts]
    dist = torch.linalg.norm(gt_trajectory[None] - forecasted_trajectory, dim=-1)
    dist = dist[..., -1]
    dist[torch.isnan(dist)] = 0
    min_mode_idx = torch.argmin(dist, dim=-1)

    return forecasted_trajectory[min_mode_idx]

def get_fde(forecasted_trajectory: torch.Tensor, gt_trajectory: torch.Tensor) -> float:
    """Compute Final Displacement Error.
    Args:
        forecasted_trajectory: Predicted trajectory with shape [fut_ts, 2]
        gt_trajectory: Ground truth trajectory with shape [fut_ts, 2]
    Returns:
        fde: Final Displacement Error
    """
    fde = float(
        torch.sqrt(
            (forecasted_trajectory[-1, 0] - gt_trajectory[-1, 0]) ** 2
            + (forecasted_trajectory[-1, 1] - gt_trajectory[-1, 1]) ** 2
        )
    )
    return fde
