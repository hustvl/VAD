import math
import mmcv
import torch
from torch import nn as nn
from mmdet.models import weighted_loss
from mmdet.models.builder import LOSSES


@LOSSES.register_module()
class PlanMapBoundLoss(nn.Module):
    """Planning constraint to push ego vehicle away from the lane boundary.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
        map_thresh (float, optional): confidence threshold to filter map predictions.
        lane_bound_cls_idx (float, optional): lane_boundary class index.
        dis_thresh (float, optional): distance threshold between ego vehicle and lane bound.
        point_cloud_range (list, optional): point cloud range.
    """

    def __init__(
        self,
        reduction='mean',
        loss_weight=1.0,
        map_thresh=0.5,
        lane_bound_cls_idx=2,
        dis_thresh=1.0,
        point_cloud_range=[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0],
        perception_detach=False
    ):
        super(PlanMapBoundLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.map_thresh = map_thresh
        self.lane_bound_cls_idx = lane_bound_cls_idx
        self.dis_thresh = dis_thresh
        self.pc_range = point_cloud_range
        self.perception_detach = perception_detach

    def forward(self,
                ego_fut_preds,
                lane_preds,
                lane_score_preds,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            ego_fut_preds (Tensor): [B, fut_ts, 2]
            lane_preds (Tensor): [B, num_vec, num_pts, 2]
            lane_score_preds (Tensor): [B, num_vec, 3]
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        if self.perception_detach:
            lane_preds = lane_preds.detach()
            lane_score_preds = lane_score_preds.detach()

        # filter lane element according to confidence score and class
        not_lane_bound_mask = lane_score_preds[..., self.lane_bound_cls_idx] < self.map_thresh
        # denormalize map pts
        lane_bound_preds = lane_preds.clone()
        lane_bound_preds[...,0:1] = (lane_bound_preds[..., 0:1] * (self.pc_range[3] -
                                self.pc_range[0]) + self.pc_range[0])
        lane_bound_preds[...,1:2] = (lane_bound_preds[..., 1:2] * (self.pc_range[4] -
                                self.pc_range[1]) + self.pc_range[1])
        # pad not-lane-boundary cls and low confidence preds
        lane_bound_preds[not_lane_bound_mask] = 1e6

        loss_bbox = self.loss_weight * plan_map_bound_loss(ego_fut_preds, lane_bound_preds,
                                                           weight=weight, dis_thresh=self.dis_thresh,
                                                           reduction=reduction, avg_factor=avg_factor)
        return loss_bbox


@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def plan_map_bound_loss(pred, target, dis_thresh=1.0):
    """Planning map bound constraint (L1 distance).

    Args:
        pred (torch.Tensor): ego_fut_preds, [B, fut_ts, 2].
        target (torch.Tensor): lane_bound_preds, [B, num_vec, num_pts, 2].
        weight (torch.Tensor): [B, fut_ts]

    Returns:
        torch.Tensor: Calculated loss [B, fut_ts]
    """
    pred = pred.cumsum(dim=-2)
    ego_traj_starts = pred[:, :-1, :]
    ego_traj_ends = pred
    B, T, _ = ego_traj_ends.size()
    padding_zeros = torch.zeros((B, 1, 2), dtype=pred.dtype, device=pred.device)  # initial position
    ego_traj_starts = torch.cat((padding_zeros, ego_traj_starts), dim=1)
    _, V, P, _ = target.size()
    ego_traj_expanded = ego_traj_ends.unsqueeze(2).unsqueeze(3)  # [B, T, 1, 1, 2]
    maps_expanded = target.unsqueeze(1)  # [1, 1, M, P, 2]
    dist = torch.linalg.norm(ego_traj_expanded - maps_expanded, dim=-1)  # [B, T, M, P]
    dist = dist.min(dim=-1, keepdim=False)[0]
    min_inst_idxs = torch.argmin(dist, dim=-1).tolist()
    batch_idxs = [[i] for i in range(dist.shape[0])]
    ts_idxs = [[i for i in range(dist.shape[1])] for j in range(dist.shape[0])]
    bd_target = target.unsqueeze(1).repeat(1, pred.shape[1], 1, 1, 1)
    min_bd_insts = bd_target[batch_idxs, ts_idxs, min_inst_idxs]  # [B, T, P, 2]
    bd_inst_starts = min_bd_insts[:, :, :-1, :].flatten(0, 2)
    bd_inst_ends = min_bd_insts[:, :, 1:, :].flatten(0, 2)
    ego_traj_starts = ego_traj_starts.unsqueeze(2).repeat(1, 1, P-1, 1).flatten(0, 2)
    ego_traj_ends = ego_traj_ends.unsqueeze(2).repeat(1, 1, P-1, 1).flatten(0, 2)

    intersect_mask = segments_intersect(ego_traj_starts, ego_traj_ends,
                                        bd_inst_starts, bd_inst_ends)
    intersect_mask = intersect_mask.reshape(B, T, P-1)
    intersect_mask = intersect_mask.any(dim=-1)
    intersect_idx = (intersect_mask == True).nonzero()

    target = target.view(target.shape[0], -1, target.shape[-1])
    # [B, fut_ts, num_vec*num_pts]
    dist = torch.linalg.norm(pred[:, :, None, :] - target[:, None, :, :], dim=-1)
    min_idxs = torch.argmin(dist, dim=-1).tolist()
    batch_idxs = [[i] for i in range(dist.shape[0])]
    ts_idxs = [[i for i in range(dist.shape[1])] for j in range(dist.shape[0])]
    min_dist = dist[batch_idxs, ts_idxs, min_idxs]
    loss = min_dist
    safe_idx = loss > dis_thresh
    unsafe_idx = loss <= dis_thresh
    loss[safe_idx] = 0
    loss[unsafe_idx] = dis_thresh - loss[unsafe_idx]

    for i in range(len(intersect_idx)):
        loss[intersect_idx[i, 0], intersect_idx[i, 1]:] = 0

    return loss


def segments_intersect(line1_start, line1_end, line2_start, line2_end):
    # Calculating the differences
    dx1 = line1_end[:, 0] - line1_start[:, 0]
    dy1 = line1_end[:, 1] - line1_start[:, 1]
    dx2 = line2_end[:, 0] - line2_start[:, 0]
    dy2 = line2_end[:, 1] - line2_start[:, 1]

    # Calculating determinants
    det = dx1 * dy2 - dx2 * dy1
    det_mask = det != 0

    # Checking if lines are parallel or coincident
    parallel_mask = torch.logical_not(det_mask)

    # Calculating intersection parameters
    t1 = ((line2_start[:, 0] - line1_start[:, 0]) * dy2 
          - (line2_start[:, 1] - line1_start[:, 1]) * dx2) / det
    t2 = ((line2_start[:, 0] - line1_start[:, 0]) * dy1 
          - (line2_start[:, 1] - line1_start[:, 1]) * dx1) / det

    # Checking intersection conditions
    intersect_mask = torch.logical_and(
        torch.logical_and(t1 >= 0, t1 <= 1),
        torch.logical_and(t2 >= 0, t2 <= 1)
    )

    # Handling parallel or coincident lines
    intersect_mask[parallel_mask] = False

    return intersect_mask


@LOSSES.register_module()
class PlanCollisionLoss(nn.Module):
    """Planning constraint to push ego vehicle away from other agents.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
        agent_thresh (float, optional): confidence threshold to filter agent predictions.
        x_dis_thresh (float, optional): distance threshold between ego and other agents in x-axis.
        y_dis_thresh (float, optional): distance threshold between ego and other agents in y-axis.
        point_cloud_range (list, optional): point cloud range.
    """

    def __init__(
        self,
        reduction='mean',
        loss_weight=1.0,
        agent_thresh=0.5,
        x_dis_thresh=1.5,
        y_dis_thresh=3.0,
        point_cloud_range = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]
    ):
        super(PlanCollisionLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.agent_thresh = agent_thresh
        self.x_dis_thresh = x_dis_thresh
        self.y_dis_thresh = y_dis_thresh
        self.pc_range = point_cloud_range

    def forward(self,
                ego_fut_preds,
                agent_preds,
                agent_fut_preds,
                agent_score_preds,
                agent_fut_cls_preds,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            ego_fut_preds (Tensor): [B, fut_ts, 2]
            agent_preds (Tensor): [B, num_agent, 2]
            agent_fut_preds (Tensor): [B, num_agent, fut_mode, fut_ts, 2]
            agent_fut_cls_preds (Tensor): [B, num_agent, fut_mode]
            agent_score_preds (Tensor): [B, num_agent, 10]
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        # filter agent element according to confidence score
        agent_max_score_preds, agent_max_score_idxs = agent_score_preds.max(dim=-1)
        not_valid_agent_mask = agent_max_score_preds < self.agent_thresh
        # filter low confidence preds
        agent_fut_preds[not_valid_agent_mask] = 1e6
        # filter not vehicle preds
        not_veh_pred_mask = agent_max_score_idxs > 4  # veh idxs are 0-4
        agent_fut_preds[not_veh_pred_mask] = 1e6
        # only use best mode pred
        best_mode_idxs = torch.argmax(agent_fut_cls_preds, dim=-1).tolist()
        batch_idxs = [[i] for i in range(agent_fut_cls_preds.shape[0])]
        agent_num_idxs = [[i for i in range(agent_fut_cls_preds.shape[1])] for j in range(agent_fut_cls_preds.shape[0])]
        agent_fut_preds = agent_fut_preds[batch_idxs, agent_num_idxs, best_mode_idxs]

        loss_bbox = self.loss_weight * plan_col_loss(ego_fut_preds, agent_preds,
                                                           agent_fut_preds=agent_fut_preds, weight=weight,
                                                           x_dis_thresh=self.x_dis_thresh,
                                                           y_dis_thresh=self.y_dis_thresh,
                                                           reduction=reduction, avg_factor=avg_factor)
        return loss_bbox


@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def plan_col_loss(
    pred,
    target,
    agent_fut_preds,
    x_dis_thresh=1.5,
    y_dis_thresh=3.0,
    dis_thresh=3.0
):
    """Planning ego-agent collsion constraint.

    Args:
        pred (torch.Tensor): ego_fut_preds, [B, fut_ts, 2].
        target (torch.Tensor): agent_preds, [B, num_agent, 2].
        agent_fut_preds (Tensor): [B, num_agent, fut_ts, 2].
        weight (torch.Tensor): [B, fut_ts, 2].
        x_dis_thresh (float, optional): distance threshold between ego and other agents in x-axis.
        y_dis_thresh (float, optional): distance threshold between ego and other agents in y-axis.
        dis_thresh (float, optional): distance threshold to filter distant agents.

    Returns:
        torch.Tensor: Calculated loss [B, fut_mode, fut_ts, 2]
    """
    pred = pred.cumsum(dim=-2)
    agent_fut_preds = agent_fut_preds.cumsum(dim=-2)
    target = target[:, :, None, :] + agent_fut_preds
    # filter distant agents from ego vehicle
    dist = torch.linalg.norm(pred[:, None, :, :] - target, dim=-1)
    dist_mask = dist > dis_thresh
    target[dist_mask] = 1e6

    # [B, num_agent, fut_ts]
    x_dist = torch.abs(pred[:, None, :, 0] - target[..., 0])
    y_dist = torch.abs(pred[:, None, :, 1] - target[..., 1])
    x_min_idxs = torch.argmin(x_dist, dim=1).tolist()
    y_min_idxs = torch.argmin(y_dist, dim=1).tolist()
    batch_idxs = [[i] for i in range(y_dist.shape[0])]
    ts_idxs = [[i for i in range(y_dist.shape[-1])] for j in range(y_dist.shape[0])]

    # [B, fut_ts]
    x_min_dist = x_dist[batch_idxs, x_min_idxs, ts_idxs]
    y_min_dist = y_dist[batch_idxs, y_min_idxs, ts_idxs]
    x_loss = x_min_dist
    safe_idx = x_loss > x_dis_thresh
    unsafe_idx = x_loss <= x_dis_thresh
    x_loss[safe_idx] = 0
    x_loss[unsafe_idx] = x_dis_thresh - x_loss[unsafe_idx]
    y_loss = y_min_dist
    safe_idx = y_loss > y_dis_thresh
    unsafe_idx = y_loss <= y_dis_thresh
    y_loss[safe_idx] = 0
    y_loss[unsafe_idx] = y_dis_thresh - y_loss[unsafe_idx]
    loss = torch.cat([x_loss.unsqueeze(-1), y_loss.unsqueeze(-1)], dim=-1)

    return loss


@LOSSES.register_module()
class PlanMapDirectionLoss(nn.Module):
    """Planning loss to force the ego heading angle consistent with lane direction.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
        theta_thresh (float, optional): angle diff thresh between ego and lane.
        point_cloud_range (list, optional): point cloud range.
    """

    def __init__(
        self,
        reduction='mean',
        loss_weight=1.0,
        map_thresh=0.5,
        dis_thresh=2.0,
        lane_div_cls_idx=0,
        point_cloud_range = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]
    ):
        super(PlanMapDirectionLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.map_thresh = map_thresh
        self.dis_thresh = dis_thresh
        self.lane_div_cls_idx = lane_div_cls_idx
        self.pc_range = point_cloud_range

    def forward(self,
                ego_fut_preds,
                lane_preds,
                lane_score_preds,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            ego_fut_preds (Tensor): [B, fut_ts, 2]
            lane_preds (Tensor): [B, num_vec, num_pts, 2]
            lane_score_preds (Tensor): [B, num_vec, 3]
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        # filter lane element according to confidence score and class
        not_lane_div_mask = lane_score_preds[..., self.lane_div_cls_idx] < self.map_thresh
        # denormalize map pts
        lane_div_preds = lane_preds.clone()
        lane_div_preds[...,0:1] = (lane_div_preds[..., 0:1] * (self.pc_range[3] -
                                self.pc_range[0]) + self.pc_range[0])
        lane_div_preds[...,1:2] = (lane_div_preds[..., 1:2] * (self.pc_range[4] -
                                self.pc_range[1]) + self.pc_range[1])
        # pad not-lane-divider cls and low confidence preds
        lane_div_preds[not_lane_div_mask] = 1e6

        loss_bbox = self.loss_weight * plan_map_dir_loss(ego_fut_preds, lane_div_preds,
                                                           weight=weight, dis_thresh=self.dis_thresh,
                                                           reduction=reduction, avg_factor=avg_factor)
        return loss_bbox


@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def plan_map_dir_loss(pred, target, dis_thresh=2.0):
    """Planning ego-map directional loss.

    Args:
        pred (torch.Tensor): ego_fut_preds, [B, fut_ts, 2].
        target (torch.Tensor): lane_div_preds, [B, num_vec, num_pts, 2].
        weight (torch.Tensor): [B, fut_ts]

    Returns:
        torch.Tensor: Calculated loss [B, fut_ts]
    """
    num_map_pts = target.shape[2]
    pred = pred.cumsum(dim=-2)
    traj_dis = torch.linalg.norm(pred[:, -1, :] - pred[:, 0, :], dim=-1)
    static_mask = traj_dis < 1.0
    target = target.unsqueeze(1).repeat(1, pred.shape[1], 1, 1, 1)

    # find the closest map instance for ego at each timestamp
    dist = torch.linalg.norm(pred[:, :, None, None, :] - target, dim=-1)
    dist = dist.min(dim=-1, keepdim=False)[0]
    min_inst_idxs = torch.argmin(dist, dim=-1).tolist()
    batch_idxs = [[i] for i in range(dist.shape[0])]
    ts_idxs = [[i for i in range(dist.shape[1])] for j in range(dist.shape[0])]
    target_map_inst = target[batch_idxs, ts_idxs, min_inst_idxs]  # [B, fut_ts, num_pts, 2]

    # calculate distance
    dist = torch.linalg.norm(pred[:, :, None, :] - target_map_inst, dim=-1)
    min_pts_idxs = torch.argmin(dist, dim=-1)
    min_pts_next_idxs = min_pts_idxs.clone()
    is_end_point = (min_pts_next_idxs == num_map_pts-1)
    not_end_point = (min_pts_next_idxs != num_map_pts-1)
    min_pts_next_idxs[is_end_point] = num_map_pts - 2
    min_pts_next_idxs[not_end_point] = min_pts_next_idxs[not_end_point] + 1
    min_pts_idxs = min_pts_idxs.tolist()
    min_pts_next_idxs = min_pts_next_idxs.tolist()
    traj_yaw = torch.atan2(torch.diff(pred[..., 1]), torch.diff(pred[..., 0]))  # [B, fut_ts-1]
    # last ts yaw assume same as previous
    traj_yaw = torch.cat([traj_yaw, traj_yaw[:, [-1]]], dim=-1)  # [B, fut_ts]
    min_pts = target_map_inst[batch_idxs, ts_idxs, min_pts_idxs]
    dist = torch.linalg.norm(min_pts - pred, dim=-1)
    dist_mask = dist > dis_thresh
    min_pts = min_pts.unsqueeze(2)
    min_pts_next = target_map_inst[batch_idxs, ts_idxs, min_pts_next_idxs].unsqueeze(2)
    map_pts = torch.cat([min_pts, min_pts_next], dim=2)
    lane_yaw = torch.atan2(torch.diff(map_pts[..., 1]).squeeze(-1), torch.diff(map_pts[..., 0]).squeeze(-1))  # [B, fut_ts]
    yaw_diff = traj_yaw - lane_yaw
    yaw_diff[yaw_diff > math.pi] =  yaw_diff[yaw_diff > math.pi] - math.pi
    yaw_diff[yaw_diff > math.pi/2] = yaw_diff[yaw_diff > math.pi/2] - math.pi
    yaw_diff[yaw_diff < -math.pi] = yaw_diff[yaw_diff < -math.pi] + math.pi
    yaw_diff[yaw_diff < -math.pi/2] = yaw_diff[yaw_diff < -math.pi/2] + math.pi
    yaw_diff[dist_mask] = 0  # loss = 0 if no lane around ego
    yaw_diff[static_mask] = 0  # loss = 0 if ego is static

    loss = torch.abs(yaw_diff)

    return loss  # [B, fut_ts]
