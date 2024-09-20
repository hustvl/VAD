import copy
from math import pi, cos, sin
import cv2
import numpy as np
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from skimage.draw import polygon
from mmdet.models import HEADS, build_loss 
from mmdet.models.dense_heads import DETRHead
from mmcv.runner import force_fp32, load_checkpoint
from mmcv.utils import TORCH_VERSION, digit_version
from mmdet.core import build_assigner, build_sampler
from mmdet3d.core.bbox.coders import build_bbox_coder
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.core.bbox.transforms import bbox_xyxy_to_cxcywh
from mmcv.cnn import Linear, bias_init_with_prob, xavier_init
from mmdet.core import (multi_apply, multi_apply, reduce_mean)
from mmdet3d.models.builder import build_backbone
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence

from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox
from projects.mmdet3d_plugin.PIPP.utils.map_utils import (
    normalize_2d_pts, normalize_2d_bbox, denormalize_2d_pts, denormalize_2d_bbox
)
from projects.mmdet3d_plugin.PIPP.utils.functional import pos2posemb2d
from projects.mmdet3d_plugin.PIPP.utils.plan_loss import segments_intersect
from shapely.geometry import LineString


# pos_idx_cnt = [0] * 256
class MLP(nn.Module):
    def __init__(self, in_channels, hidden_unit, verbose=False):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_unit),
            nn.LayerNorm(hidden_unit),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.mlp(x)
        return x

class LaneNet(nn.Module):
    def __init__(self, in_channels, hidden_unit, num_subgraph_layers):
        super(LaneNet, self).__init__()
        self.num_subgraph_layers = num_subgraph_layers
        self.layer_seq = nn.Sequential()
        for i in range(num_subgraph_layers):
            self.layer_seq.add_module(
                f'lmlp_{i}', MLP(in_channels, hidden_unit))
            in_channels = hidden_unit*2

    def forward(self, pts_lane_feats):
        '''
            Extract lane_feature from vectorized lane representation

        Args:
            pts_lane_feats: [batch size, max_pnum, pts, D]

        Returns:
            inst_lane_feats: [batch size, max_pnum, D]
        '''
        x = pts_lane_feats
        for name, layer in self.layer_seq.named_modules():
            if isinstance(layer, MLP):
                # x [bs,max_lane_num,9,dim]
                x = layer(x)
                x_max = torch.max(x, -2)[0]
                x_max = x_max.unsqueeze(2).repeat(1, 1, x.shape[2], 1)
                x = torch.cat([x, x_max], dim=-1)
        x_max = torch.max(x, -2)[0]
        return x_max


@HEADS.register_module()
class v116ADTRHead(DETRHead):
    """Head of Detr3D.
    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
        bev_h, bev_w (int): spatial shape of BEV queries.
    """
    # NOTE: already support map
    def __init__(self,
                 *args,
                 with_box_refine=False,
                 as_two_stage=False,
                 transformer=None,
                 bbox_coder=None,
                 num_cls_fcs=2,
                 code_weights=None,
                 bev_h=30,
                 bev_w=30,
                 fut_ts=6,
                 mot_fut_mode=6,
                 loss_mot_reg=dict(type='L1Loss', loss_weight=0.25),
                 loss_mot_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=0.8),
                 map_bbox_coder=None,
                 map_num_query=900,
                 map_num_classes=3,
                 map_num_vec=20,
                 map_num_pts_per_vec=2,
                 map_num_pts_per_gt_vec=2,
                 map_query_embed_type='all_pts',
                 map_transform_method='minmax',
                 map_gt_shift_pts_pattern='v0',
                 map_dir_interval=1,
                 map_code_size=None,
                 map_code_weights=None,
                 loss_map_cls=dict(
                     type='CrossEntropyLoss',
                     bg_cls_weight=0.1,
                     use_sigmoid=False,
                     loss_weight=1.0,
                     class_weight=1.0),
                 loss_map_bbox=dict(type='L1Loss', loss_weight=5.0),
                 loss_map_iou=dict(type='GIoULoss', loss_weight=2.0),
                 loss_map_pts=dict(
                    type='ChamferDistance',loss_src_weight=1.0,loss_dst_weight=1.0
                 ),
                 loss_map_dir=dict(type='PtsDirCosLoss', loss_weight=2.0),
                 tot_epoch=None,
                 mot_decoder=None,
                 mot_map_decoder=None,
                 interaction_pe_type='mlp',
                 mot_det_score=None,
                 mot_map_thresh=0.5,
                 mot_dis_thresh=0.2,
                 pe_normalization=True,
                 plan_fut_mode=256,
                 plan_fut_mode_testing=4096,
                 loss_plan_cls_col=None,
                 loss_plan_cls_bd=None,
                 loss_plan_cls_cl=None,
                 loss_plan_cls_expert=None,
                 loss_plan_reg=dict(type='L1Loss', loss_weight=0.),
                 loss_plan_bound=dict(type='PlanMapBoundLoss', loss_weight=0.),
                 loss_plan_agent_dis=dict(type='PlanAgentDisLoss', loss_weight=0.),
                 loss_plan_map_theta=dict(type='PlanMapThetaLoss', loss_weight=0.),
                 loss_tl_status_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0,
                     class_weight=None),
                 loss_tl_trigger_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0,
                     class_weight=None),
                loss_stopsign_trigger_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0,
                     class_weight=None),
                 ego_pv_decoder=None,
                 ego_agent_decoder=None,
                 ego_map_decoder=None,
                 cf_backbone=None,
                 cf_backbone_ckpt=None,
                 ego_query_thresh=None,
                 query_use_fix_pad=None,
                 ego_lcf_feat_idx=None,
                 valid_fut_ts=6,
                 plan_anchors_path='./plan_anchors_endpoint_242.npy',
                 **kwargs):

        self.bev_h = bev_h
        self.bev_w = bev_w
        self.fp16_enabled = False
        self.fut_ts = fut_ts
        self.mot_fut_mode = mot_fut_mode
        self.tot_epoch = tot_epoch
        self.mot_decoder = mot_decoder
        self.mot_map_decoder = mot_map_decoder
        self.interaction_pe_type = interaction_pe_type
        self.mot_det_score = mot_det_score
        self.mot_map_thresh = mot_map_thresh
        self.mot_dis_thresh = mot_dis_thresh
        self.pe_normalization = pe_normalization
        self.plan_fut_mode = plan_fut_mode
        self.plan_fut_mode_testing = plan_fut_mode_testing
        self.ego_pv_decoder = ego_pv_decoder
        self.ego_agent_decoder = ego_agent_decoder
        self.ego_map_decoder = ego_map_decoder
        self.ego_query_thresh = ego_query_thresh
        self.query_use_fix_pad = query_use_fix_pad
        self.ego_lcf_feat_idx = ego_lcf_feat_idx
        self.valid_fut_ts = valid_fut_ts
        self.cf_backbone = cf_backbone
        self.cf_backbone_ckpt = cf_backbone_ckpt
        self.plan_anchors = np.load(plan_anchors_path)
        self.plan_anchors = torch.from_numpy(self.plan_anchors).to(torch.float32).cuda()

        self.traj_selected_cnt = torch.zeros(self.plan_anchors.shape[0]).to(torch.float32).cuda()


        if loss_mot_cls['use_sigmoid'] == True:
            self.mot_num_cls = 1  # dont need to consider cls num here
        else:
          self.mot_num_cls = 2

        self.tl_status_num_cls = 3  # Green, Red, Yellow
        self.tl_trigger_num_cls = 1
        self.stopsign_trigger_num_cls = 5

        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        if self.as_two_stage:
            transformer['as_two_stage'] = self.as_two_stage
        if 'code_size' in kwargs:
            self.code_size = kwargs['code_size']
        else:
            self.code_size = 10
        if code_weights is not None:
            self.code_weights = code_weights
        else:
            self.code_weights = [1.0, 1.0, 1.0,
                                 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]
        if map_code_size is not None:
            self.map_code_size = map_code_size
        else:
            self.map_code_size = 10
        if map_code_weights is not None:
            self.map_code_weights = map_code_weights
        else:
            self.map_code_weights = [1.0, 1.0, 1.0,
                                 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.pc_range = self.bbox_coder.pc_range
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        self.num_cls_fcs = num_cls_fcs - 1

        self.map_bbox_coder = build_bbox_coder(map_bbox_coder)
        self.map_query_embed_type = map_query_embed_type
        self.map_transform_method = map_transform_method
        self.map_gt_shift_pts_pattern = map_gt_shift_pts_pattern
        map_num_query = map_num_vec * map_num_pts_per_vec
        self.map_num_query = map_num_query
        self.map_num_classes = map_num_classes
        self.map_num_vec = map_num_vec
        self.map_num_pts_per_vec = map_num_pts_per_vec
        self.map_num_pts_per_gt_vec = map_num_pts_per_gt_vec
        self.map_dir_interval = map_dir_interval

        if loss_map_cls['use_sigmoid'] == True:
            self.map_cls_out_channels = map_num_classes
        else:
            self.map_cls_out_channels = map_num_classes + 1

        self.map_bg_cls_weight = 0
        map_class_weight = loss_map_cls.get('class_weight', None)
        if map_class_weight is not None and (self.__class__ is v116ADTRHead):
            assert isinstance(map_class_weight, float), 'Expected ' \
                'class_weight to have type float. Found ' \
                f'{type(map_class_weight)}.'
            # NOTE following the official DETR rep0, bg_cls_weight means
            # relative classification weight of the no-object class.
            map_bg_cls_weight = loss_map_cls.get('bg_cls_weight', map_class_weight)
            assert isinstance(map_bg_cls_weight, float), 'Expected ' \
                'bg_cls_weight to have type float. Found ' \
                f'{type(map_bg_cls_weight)}.'
            map_class_weight = torch.ones(map_num_classes + 1) * map_class_weight
            # set background class as the last indice
            map_class_weight[map_num_classes] = map_bg_cls_weight
            loss_map_cls.update({'class_weight': map_class_weight})
            if 'bg_cls_weight' in loss_map_cls:
                loss_map_cls.pop('bg_cls_weight')
            self.map_bg_cls_weight = map_bg_cls_weight
        
        self.mot_bg_cls_weight = 0

        super(v116ADTRHead, self).__init__(*args, transformer=transformer, **kwargs)
        self.code_weights = nn.Parameter(torch.tensor(
            self.code_weights, requires_grad=False), requires_grad=False)
        self.map_code_weights = nn.Parameter(torch.tensor(
            self.map_code_weights, requires_grad=False), requires_grad=False)
        
        if kwargs['train_cfg'] is not None:
            assert 'map_assigner' in kwargs['train_cfg'], 'map assigner should be provided '\
                'when train_cfg is set.'
            map_assigner = kwargs['train_cfg']['map_assigner']
            assert loss_map_cls['loss_weight'] == map_assigner['cls_cost']['weight'], \
                'The classification weight for loss and matcher should be' \
                'exactly the same.'
            assert loss_map_bbox['loss_weight'] == map_assigner['reg_cost'][
                'weight'], 'The regression L1 weight for loss and matcher ' \
                'should be exactly the same.'
            assert loss_map_iou['loss_weight'] == map_assigner['iou_cost']['weight'], \
                'The regression iou weight for loss and matcher should be' \
                'exactly the same.'
            assert loss_map_pts['loss_weight'] == map_assigner['pts_cost']['weight'], \
                'The regression l1 weight for map pts loss and matcher should be' \
                'exactly the same.'

            self.map_assigner = build_assigner(map_assigner)
            # DETR sampling=False, so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.map_sampler = build_sampler(sampler_cfg, context=self)
        
        self.loss_mot_reg = build_loss(loss_mot_reg)
        self.loss_mot_cls = build_loss(loss_mot_cls)
        self.loss_map_bbox = build_loss(loss_map_bbox)
        self.loss_map_cls = build_loss(loss_map_cls)
        self.loss_map_iou = build_loss(loss_map_iou)
        self.loss_map_pts = build_loss(loss_map_pts)
        self.loss_map_dir = build_loss(loss_map_dir)
        self.loss_plan_cls_col = build_loss(loss_plan_cls_col)
        self.loss_plan_cls_bd = build_loss(loss_plan_cls_bd)
        self.loss_plan_cls_cl = build_loss(loss_plan_cls_cl)
        self.loss_plan_cls_expert = build_loss(loss_plan_cls_expert)

        self.loss_plan_reg = build_loss(loss_plan_reg)
        self.loss_plan_bound = build_loss(loss_plan_bound)
        self.loss_plan_agent_dis = build_loss(loss_plan_agent_dis)
        self.loss_plan_map_theta = build_loss(loss_plan_map_theta)
        self.loss_tl_status_cls = build_loss(loss_tl_status_cls)
        self.loss_tl_trigger_cls = build_loss(loss_tl_trigger_cls)
        self.loss_stopsign_trigger_cls = build_loss(loss_stopsign_trigger_cls)


    # NOTE: already support map
    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        cls_branch = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        mot_reg_branch = []
        for _ in range(self.num_reg_fcs):
            mot_reg_branch.append(Linear(self.embed_dims*2, self.embed_dims*2))
            mot_reg_branch.append(nn.ReLU())
        mot_reg_branch.append(Linear(self.embed_dims*2, self.fut_ts*2))
        mot_reg_branch = nn.Sequential(*mot_reg_branch)

        mot_cls_branch = []
        for _ in range(self.num_reg_fcs):
            mot_cls_branch.append(Linear(self.embed_dims*2, self.embed_dims*2))
            mot_cls_branch.append(nn.LayerNorm(self.embed_dims*2))
            mot_cls_branch.append(nn.ReLU(inplace=True))
        mot_cls_branch.append(Linear(self.embed_dims*2, self.mot_num_cls))
        mot_cls_branch = nn.Sequential(*mot_cls_branch)

        map_cls_branch = []
        for _ in range(self.num_reg_fcs):
            map_cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            map_cls_branch.append(nn.LayerNorm(self.embed_dims))
            map_cls_branch.append(nn.ReLU(inplace=True))
        map_cls_branch.append(Linear(self.embed_dims, self.map_cls_out_channels))
        map_cls_branch = nn.Sequential(*map_cls_branch)

        map_reg_branch = []
        for _ in range(self.num_reg_fcs):
            map_reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            map_reg_branch.append(nn.ReLU())
        map_reg_branch.append(Linear(self.embed_dims, self.map_code_size))
        map_reg_branch = nn.Sequential(*map_reg_branch)


        ego_query_pre_branch = []
        ego_query_pre_branch.append(Linear(self.embed_dims * self.fut_ts, self.embed_dims))
        ego_query_pre_branch.append(nn.ReLU())
        ego_query_pre_branch.append(Linear(self.embed_dims, self.embed_dims))
        self.ego_query_pre_branch = nn.Sequential(*ego_query_pre_branch)


        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        # last reg_branch is used to generate proposal from
        # encode feature map when as_two_stage is True.
        num_decoder_layers = 1
        num_map_decoder_layers = 1
        if self.transformer.decoder is not None:
            num_decoder_layers = self.transformer.decoder.num_layers
        if self.transformer.map_decoder is not None:
            num_map_decoder_layers = self.transformer.map_decoder.num_layers
        num_mot_decoder_layers = 1
        num_pred = (num_decoder_layers + 1) if \
            self.as_two_stage else num_decoder_layers
        mot_num_pred = (num_mot_decoder_layers + 1) if \
            self.as_two_stage else num_mot_decoder_layers
        map_num_pred = (num_map_decoder_layers + 1) if \
            self.as_two_stage else num_map_decoder_layers

        if self.with_box_refine:
            self.cls_branches = _get_clones(cls_branch, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
            self.mot_reg_branches = _get_clones(mot_reg_branch, mot_num_pred)
            self.mot_cls_branches = _get_clones(mot_cls_branch, mot_num_pred)
            self.map_cls_branches = _get_clones(map_cls_branch, map_num_pred)
            self.map_reg_branches = _get_clones(map_reg_branch, map_num_pred)
        else:
            self.cls_branches = nn.ModuleList(
                [cls_branch for _ in range(num_pred)])
            self.reg_branches = nn.ModuleList(
                [reg_branch for _ in range(num_pred)])
            self.mot_reg_branches = nn.ModuleList(
                [mot_reg_branch for _ in range(mot_num_pred)])
            self.mot_cls_branches = nn.ModuleList(
                [mot_cls_branch for _ in range(mot_num_pred)])
            self.map_cls_branches = nn.ModuleList(
                [map_cls_branch for _ in range(map_num_pred)])
            self.map_reg_branches = nn.ModuleList(
                [map_reg_branch for _ in range(map_num_pred)])

        if not self.as_two_stage:
            self.bev_embedding = nn.Embedding(
                self.bev_h * self.bev_w, self.embed_dims)
            self.query_embedding = nn.Embedding(self.num_query,
                                                self.embed_dims * 2)
            if self.map_query_embed_type == 'all_pts':
                self.map_query_embedding = nn.Embedding(self.map_num_query,
                                                    self.embed_dims * 2)
            elif self.map_query_embed_type == 'instance_pts':
                self.map_query_embedding = None
                self.map_instance_embedding = nn.Embedding(self.map_num_vec, self.embed_dims * 2)
                self.map_pts_embedding = nn.Embedding(self.map_num_pts_per_vec, self.embed_dims * 2)
        
        if self.mot_decoder is not None:
            self.mot_decoder = build_transformer_layer_sequence(self.mot_decoder)
            self.mot_mode_query = nn.Embedding(self.mot_fut_mode, self.embed_dims)	
            self.mot_mode_query.weight.requires_grad = True
        else:
            raise NotImplementedError('Not implement yet')

        if self.mot_map_decoder is not None:
            self.lane_encoder = LaneNet(self.embed_dims, self.embed_dims // 2, 3)
            self.mot_map_decoder = build_transformer_layer_sequence(self.mot_map_decoder)

        # self.ego_query = nn.Embedding(self.plan_fut_mode, self.embed_dims)
        # self.ego_query = pos2posemb2d(self.plan_anchors.reshape(1, self.plan_fut_mode * self.fut_ts, -1)) \
        #         .reshape(self.plan_fut_mode, self.fut_ts, -1)

        if self.ego_pv_decoder is not None:
            self.ego_pv_decoder = build_transformer_layer_sequence(self.ego_pv_decoder)
            MAXNUM_PV_TOKEN = 800
            self.pv_pos_embedding = nn.Embedding(
                MAXNUM_PV_TOKEN, self.embed_dims)

        if self.ego_agent_decoder is not None:
            self.ego_agent_decoder = build_transformer_layer_sequence(self.ego_agent_decoder)

        if self.ego_map_decoder is not None:
            self.ego_map_decoder = build_transformer_layer_sequence(self.ego_map_decoder)

        plan_reg_branch = []
        # plan_in_dim = self.embed_dims*4 + len(self.ego_lcf_feat_idx) \
        #     if self.ego_lcf_feat_idx is not None else self.embed_dims*4
        for _ in range(self.num_reg_fcs):
            plan_reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            plan_reg_branch.append(nn.ReLU())
        plan_reg_branch.append(Linear(self.embed_dims, self.fut_ts*2))
        self.plan_reg_branch = nn.Sequential(*plan_reg_branch)

        self.fus_mlp = nn.Sequential(
            nn.Linear(self.mot_fut_mode*2*self.embed_dims, self.embed_dims, bias=True),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims, bias=True))

        if self.interaction_pe_type == 'sine_mlp':
            pe_embed_mlps = nn.Sequential(
                nn.Linear(self.embed_dims, self.embed_dims*2),
                nn.ReLU(),
                nn.Linear(self.embed_dims*2, self.embed_dims),
            )
        elif self.interaction_pe_type == 'mlp':
            pe_embed_mlps = nn.Linear(2, self.embed_dims)
        else:
            raise NotImplementedError('Not implement yet')
        
        self.pe_embed_mlps = _get_clones(pe_embed_mlps, 4)

        self.ego_feat_projs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.embed_dims, self.embed_dims, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.embed_dims, self.embed_dims, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.embed_dims, self.embed_dims, bias=True),
                ), # for agent
                nn.Sequential(
                    nn.Linear(self.embed_dims, self.embed_dims, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.embed_dims, self.embed_dims, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.embed_dims, self.embed_dims, bias=True),
                ), # for map
                nn.Sequential(
                    nn.Linear(self.embed_dims, self.embed_dims, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.embed_dims, self.embed_dims, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.embed_dims, self.embed_dims, bias=True),
                ), # for traffic light
                nn.Sequential(
                    nn.Linear(2, self.embed_dims, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.embed_dims, self.embed_dims, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.embed_dims, self.embed_dims, bias=True),
                ), # for target point
                nn.Sequential(
                    nn.Linear(len(self.ego_lcf_feat_idx), self.embed_dims, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.embed_dims, self.embed_dims, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.embed_dims, self.embed_dims, bias=True),
                ), # for lcf feat
                nn.Sequential(
                    nn.Linear(6, self.embed_dims, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.embed_dims, self.embed_dims, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.embed_dims, self.embed_dims, bias=True),
                ),  # for cmdid
                nn.Sequential(
                    nn.Linear(self.embed_dims, self.embed_dims, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.embed_dims, self.embed_dims, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.embed_dims, self.embed_dims, bias=True),
                ), # for pv_feat
                nn.Sequential(
                    nn.Linear(140 * 140, self.embed_dims, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.embed_dims, self.embed_dims, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.embed_dims, self.embed_dims, bias=True),
                ), # for target point rasterized
            ]
        )

        # NOTE: add front-view feature encoder
        if self.cf_backbone is not None:
            self.cf_backbone = build_backbone(self.cf_backbone)

        tl_feats_branch = []
        tl_feats_branch.append(Linear(2048 * 10 * 13 + 6 * self.embed_dims * 5 * 7, self.embed_dims)) #10 * 13  5 * 7
        tl_feats_branch.append(nn.LayerNorm(self.embed_dims))
        tl_feats_branch.append(nn.ReLU(inplace=True))
        tl_feats_branch.append(Linear(self.embed_dims, self.embed_dims))
        tl_feats_branch.append(nn.LayerNorm(self.embed_dims))
        tl_feats_branch.append(nn.ReLU(inplace=True))
        self.tl_feats_branch = nn.Sequential(*tl_feats_branch)

        tl_status_cls_branch = []
        tl_status_cls_branch.append(Linear(self.embed_dims, self.embed_dims))
        tl_status_cls_branch.append(nn.LayerNorm(self.embed_dims))
        tl_status_cls_branch.append(nn.ReLU(inplace=True))
        tl_status_cls_branch.append(Linear(self.embed_dims, self.tl_status_num_cls))
        self.tl_status_cls_branch = nn.Sequential(*tl_status_cls_branch)

        tl_trigger_cls_branch = []
        tl_trigger_cls_branch.append(Linear(self.embed_dims, self.embed_dims))
        tl_trigger_cls_branch.append(nn.LayerNorm(self.embed_dims))
        tl_trigger_cls_branch.append(nn.ReLU(inplace=True))
        tl_trigger_cls_branch.append(Linear(self.embed_dims, self.tl_trigger_num_cls))
        self.tl_trigger_cls_branch = nn.Sequential(*tl_trigger_cls_branch)
        
        stopsign_trigger_cls_branch = []
        stopsign_trigger_cls_branch.append(Linear(self.embed_dims, self.embed_dims))
        stopsign_trigger_cls_branch.append(nn.LayerNorm(self.embed_dims))
        stopsign_trigger_cls_branch.append(nn.ReLU(inplace=True))
        stopsign_trigger_cls_branch.append(Linear(self.embed_dims, self.stopsign_trigger_num_cls))
        self.stopsign_trigger_cls_branch = nn.Sequential(*stopsign_trigger_cls_branch)

        plan_cls_col_branch = []
        # plan_cls_col_branch.append(Linear(self.embed_dims, self.embed_dims))
        # plan_cls_col_branch.append(nn.LayerNorm(self.embed_dims))
        # plan_cls_col_branch.append(nn.ReLU(inplace=True))
        plan_cls_col_branch.append(Linear(self.embed_dims, 1))
        self.plan_cls_col_branch = nn.Sequential(*plan_cls_col_branch)

        plan_cls_bd_branch = []
        # plan_cls_bd_branch.append(Linear(self.embed_dims, self.embed_dims))
        # plan_cls_bd_branch.append(nn.LayerNorm(self.embed_dims))
        # plan_cls_bd_branch.append(nn.ReLU(inplace=True))
        plan_cls_bd_branch.append(Linear(self.embed_dims, 1))
        self.plan_cls_bd_branch = nn.Sequential(*plan_cls_bd_branch)

        plan_cls_cl_branch = []
        # plan_cls_cl_branch.append(Linear(self.embed_dims, self.embed_dims))
        # plan_cls_cl_branch.append(nn.LayerNorm(self.embed_dims))
        # plan_cls_cl_branch.append(nn.ReLU(inplace=True))
        plan_cls_cl_branch.append(Linear(self.embed_dims, 1))
        self.plan_cls_cl_branch = nn.Sequential(*plan_cls_cl_branch)

        plan_cls_expert_branch = []
        plan_cls_expert_branch.append(Linear(self.embed_dims, self.embed_dims))
        plan_cls_expert_branch.append(nn.LayerNorm(self.embed_dims))
        plan_cls_expert_branch.append(nn.ReLU(inplace=True))
        plan_cls_expert_branch.append(Linear(self.embed_dims, 1))
        self.plan_cls_expert_branch = nn.Sequential(*plan_cls_expert_branch)

    # NOTE: already support map
    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)
        if self.loss_map_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.map_cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)
        if self.loss_mot_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.mot_cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)
        if self.loss_tl_status_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            nn.init.constant_(self.tl_status_cls_branch[-1].bias, bias_init)
        if self.loss_tl_trigger_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            nn.init.constant_(self.tl_trigger_cls_branch[-1].bias, bias_init)
        if self.loss_stopsign_trigger_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            nn.init.constant_(self.stopsign_trigger_cls_branch[-1].bias, bias_init)
        # if self.plan_cls_col_branch.use_sigmoid:
        #     bias_init = bias_init_with_prob(0.01)
        #     nn.init.constant_(self.plan_cls_col_branch[-1].bias, bias_init)
        # if self.plan_cls_bd_branch.use_sigmoid:
        #     bias_init = bias_init_with_prob(0.01)
        #     nn.init.constant_(self.plan_cls_bd_branch[-1].bias, bias_init)
        # for m in self.map_reg_branches:
        #     constant_init(m[-1], 0, bias=0)
        # nn.init.constant_(self.map_reg_branches[0][-1].bias.data[2:], 0.)
        if self.mot_decoder is not None:
            for p in self.mot_decoder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
            nn.init.orthogonal_(self.mot_mode_query.weight)
        if self.mot_map_decoder is not None:
            for p in self.mot_map_decoder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
            for p in self.lane_encoder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        if self.ego_pv_decoder is not None:
            for p in self.ego_pv_decoder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        if self.ego_agent_decoder is not None:
            for p in self.ego_agent_decoder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        if self.ego_map_decoder is not None:
            for p in self.ego_map_decoder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        if self.interaction_pe_type is not None:
            for emb_mlp in self.pe_embed_mlps:
                xavier_init(emb_mlp, distribution='uniform', bias=0.)
        # if self.cf_backbone is not None:
        #     load_checkpoint(self.cf_backbone, self.cf_backbone_ckpt, map_location='cpu')

    # NOTE: already support map
    # @auto_fp16(apply_to=('mlvl_feats'))
    @force_fp32(apply_to=('mlvl_feats', 'prev_bev'))
    def forward(self,
                mlvl_feats,
                img_metas,
                prev_bev=None,
                only_bev=False,
                ego_his_trajs=None,
                ego_lcf_feat=None,
                cf_img=None,
                command_wp=None,
                command_id=None,
                target_point=None,
                ego_fut_trajs=None
            ):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
            prev_bev: previous bev featues
            only_bev: only compute BEV features with encoder. 
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """
        
        if not only_bev and not self.training:
            self.plan_fut_mode = self.plan_fut_mode_testing

        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype
        object_query_embeds = self.query_embedding.weight.to(dtype)
        
        # import pdb;pdb.set_trace()
        if self.map_query_embed_type == 'all_pts':
            map_query_embeds = self.map_query_embedding.weight.to(dtype)
        elif self.map_query_embed_type == 'instance_pts':
            map_pts_embeds = self.map_pts_embedding.weight.unsqueeze(0)
            map_instance_embeds = self.map_instance_embedding.weight.unsqueeze(1)
            map_query_embeds = (map_pts_embeds + map_instance_embeds).flatten(0, 1).to(dtype)

        bev_queries = self.bev_embedding.weight.to(dtype)

        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w),
                               device=bev_queries.device).to(dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)
            
        if only_bev:  # only use encoder to obtain BEV features, TODO: refine the workaround
            return self.transformer.get_bev_features(
                mlvl_feats,
                bev_queries,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h,
                             self.real_w / self.bev_w),
                bev_pos=bev_pos,
                img_metas=img_metas,
                prev_bev=prev_bev,
            )
        else:
            outputs = self.transformer(
                mlvl_feats,
                bev_queries,
                object_query_embeds,
                map_query_embeds,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h,
                             self.real_w / self.bev_w),
                bev_pos=bev_pos,
                reg_branches=self.reg_branches if self.with_box_refine else None,  # noqa:E501
                cls_branches=self.cls_branches if self.as_two_stage else None,
                map_reg_branches=self.map_reg_branches if self.with_box_refine else None,  # noqa:E501
                map_cls_branches=self.map_cls_branches if self.as_two_stage else None,
                img_metas=img_metas,
                prev_bev=prev_bev
            )

        bev_embed, hs, init_reference, inter_references, \
            map_hs, map_init_reference, map_inter_references = outputs

        hs = hs.permute(0, 2, 1, 3)
        outputs_classes = []
        outputs_coords = []
        outputs_coords_bev = []
        outputs_mot_trajs = []
        outputs_mot_trajs_classes = []

        map_hs = map_hs.permute(0, 2, 1, 3)
        map_outputs_classes = []
        map_outputs_coords = []
        map_outputs_pts_coords = []
        map_outputs_coords_bev = []

        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.cls_branches[lvl](hs[lvl])
            tmp = self.reg_branches[lvl](hs[lvl])

            # TODO: check the shape of reference
            assert reference.shape[-1] == 3
            tmp[..., 0:2] = tmp[..., 0:2] + reference[..., 0:2]
            tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
            outputs_coords_bev.append(tmp[..., 0:2].clone().detach())
            tmp[..., 4:5] = tmp[..., 4:5] + reference[..., 2:3]
            tmp[..., 4:5] = tmp[..., 4:5].sigmoid()
            tmp[..., 0:1] = (tmp[..., 0:1] * (self.pc_range[3] -
                             self.pc_range[0]) + self.pc_range[0])
            tmp[..., 1:2] = (tmp[..., 1:2] * (self.pc_range[4] -
                             self.pc_range[1]) + self.pc_range[1])
            tmp[..., 4:5] = (tmp[..., 4:5] * (self.pc_range[5] -
                             self.pc_range[2]) + self.pc_range[2])

            # TODO: check if using sigmoid
            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        
        for lvl in range(map_hs.shape[0]):
            if lvl == 0:
                reference = map_init_reference
            else:
                reference = map_inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            map_outputs_class = self.map_cls_branches[lvl](
                map_hs[lvl].view(bs,self.map_num_vec, self.map_num_pts_per_vec,-1).mean(2)
            )
            tmp = self.map_reg_branches[lvl](map_hs[lvl])
            # TODO: check the shape of reference
            assert reference.shape[-1] == 2
            tmp[..., 0:2] += reference[..., 0:2]
            tmp = tmp.sigmoid() # cx,cy,w,h
            map_outputs_coord, map_outputs_pts_coord = self.map_transform_box(tmp)
            map_outputs_coords_bev.append(map_outputs_pts_coord.clone().detach())
            map_outputs_classes.append(map_outputs_class)
            map_outputs_coords.append(map_outputs_coord)
            map_outputs_pts_coords.append(map_outputs_pts_coord)
            
        if self.mot_decoder is not None:
            batch_size, num_agent = outputs_coords_bev[-1].shape[:2]
            # mot_query
            mot_query = hs[-1].permute(1, 0, 2)  # [A, B, D]
            mode_query = self.mot_mode_query.weight  # [mot_fut_mode, D]
            # [M, B, D], M=A*mot_fut_mode
            mot_query = (mot_query[:, None, :, :] + mode_query[None, :, None, :]).flatten(0, 1)

            if self.interaction_pe_type is not None:
                mot_coords = outputs_coords_bev[-1]  # [B, A, 2]
                mot_coords = pos2posemb2d(mot_coords, num_pos_feats=self.embed_dims // 2) if self.interaction_pe_type == 'sine_mlp' else mot_coords
                mot_pos = self.pe_embed_mlps[0](mot_coords)  # [B, A, D]
                mot_pos = mot_pos.unsqueeze(2).repeat(1, 1, self.mot_fut_mode, 1).flatten(1, 2)
                mot_pos = mot_pos.permute(1, 0, 2)  # [M, B, D]
            else:
                mot_pos = None

            if self.mot_det_score is not None:
                mot_score = outputs_classes[-1]
                max_mot_score = mot_score.max(dim=-1)[0]
                invalid_mot_idx = max_mot_score < self.mot_det_score  # [B, A]
                invalid_mot_idx = invalid_mot_idx.unsqueeze(2).repeat(1, 1, self.mot_fut_mode).flatten(1, 2)
            else:
                invalid_mot_idx = None

            mot_hs = self.mot_decoder(
                query=mot_query,
                key=mot_query,
                value=mot_query,
                query_pos=mot_pos,
                key_pos=mot_pos,
                key_padding_mask=invalid_mot_idx)

            if self.mot_map_decoder is not None:
                # map preprocess
                mot_coords = outputs_coords_bev[-1]  # [B, A, 2]
                mot_coords = mot_coords.unsqueeze(2).repeat(1, 1, self.mot_fut_mode, 1).flatten(1, 2)
                map_query = map_hs[-1].view(batch_size, self.map_num_vec, self.map_num_pts_per_vec, -1)
                map_query = self.lane_encoder(map_query)  # [B, P, pts, D] -> [B, P, D]
                map_score = map_outputs_classes[-1]
                map_pos = map_outputs_coords_bev[-1]
                map_query, map_pos, key_padding_mask = self.select_and_pad_pred_map(
                    mot_coords, map_query, map_score, map_pos,
                    map_thresh=self.mot_map_thresh, dis_thresh=self.mot_dis_thresh,
                    pe_normalization=self.pe_normalization, use_fix_pad=True)
                map_query = map_query.permute(1, 0, 2)  # [P, B*M, D]
                ca_mot_query = mot_hs.permute(1, 0, 2).flatten(0, 1).unsqueeze(0)

                # position encoding
                if self.interaction_pe_type is not None:
                    (attn_num_query, attn_batch) = ca_mot_query.shape[:2] 
                    mot_pos = torch.zeros((attn_num_query, attn_batch, 2), device=mot_hs.device)
                    mot_pos = pos2posemb2d(mot_pos, num_pos_feats=self.embed_dims // 2) if self.interaction_pe_type == 'sine_mlp' else mot_pos
                    mot_pos = self.pe_embed_mlps[1](mot_pos)
                    map_pos = map_pos.permute(1, 0, 2)
                    map_pos = pos2posemb2d(map_pos, num_pos_feats=self.embed_dims // 2) if self.interaction_pe_type == 'sine_mlp' else map_pos
                    map_pos = self.pe_embed_mlps[1](map_pos)
                else:
                    mot_pos, map_pos = None, None
                
                ca_mot_query = self.mot_map_decoder(
                    query=ca_mot_query,
                    key=map_query,
                    value=map_query,
                    query_pos=mot_pos,
                    key_pos=map_pos,
                    key_padding_mask=key_padding_mask)
            else:
                ca_mot_query = mot_hs.permute(1, 0, 2).flatten(0, 1).unsqueeze(0)

            batch_size = outputs_coords_bev[-1].shape[0]
            mot_hs = mot_hs.permute(1, 0, 2).unflatten(
                dim=1, sizes=(num_agent, self.mot_fut_mode)
            )
            ca_mot_query = ca_mot_query.squeeze(0).unflatten(
                dim=0, sizes=(batch_size, num_agent, self.mot_fut_mode)
            )
            mot_hs = torch.cat([mot_hs, ca_mot_query], dim=-1)  # [B, A, mot_fut_mode, 2D]
        else:
            raise NotImplementedError('Not implement yet')

        outputs_traj = self.mot_reg_branches[0](mot_hs)
        outputs_mot_trajs.append(outputs_traj)
        outputs_mot_class = self.mot_cls_branches[0](mot_hs)
        outputs_mot_trajs_classes.append(outputs_mot_class.squeeze(-1))

        map_outputs_classes = torch.stack(map_outputs_classes)
        map_outputs_coords = torch.stack(map_outputs_coords)
        map_outputs_pts_coords = torch.stack(map_outputs_pts_coords)
        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)
        outputs_mot_trajs = torch.stack(outputs_mot_trajs)
        outputs_mot_trajs_classes = torch.stack(outputs_mot_trajs_classes)

        # planning
        (batch, num_agent) = mot_hs.shape[:2]


        # kinodynamic filtering      
        # dynamic voca      ego_lcf_feat   ego_his_trajs   ego_fut_trajs
        # from carla_simulation.team_code_autopilot.autopilot import EgoModel
        # self.ego_model = EgoModel(dt=0.5)
        # vx, vy = ego_lcf_feat[0,0,0,:2]
        # spds = (vx**2 + vy**2).sqrt()
        # self.ego_model.forward(locs=np.array([0,0]), yaws=0, spds=spds, acts=np.array([-1, 1, 0]))       # steer, throt, brake

        Dt = 0.5
        vx, vy, ax, ay = ego_lcf_feat[0,0,0,:4]
        v_xy = torch.sqrt(vx**2 + vy**2)
        a_xy = torch.sqrt(ax**2 + ay**2)
        pred_dis = v_xy * Dt + 1 / 2 * a_xy * Dt**2
        kinodynamic_mask = (torch.norm(self.plan_anchors[:,0,:], dim=-1) - pred_dis).abs() < 100000000000
        used_index = torch.multinomial(kinodynamic_mask.float(), self.plan_fut_mode, replacement=False)
        # used_index = torch.LongTensor(random.sample(list(range(self.plan_anchors.shape[0]))[kinodynamic_mask], self.plan_fut_mode)).to(mot_hs.device)
        if self.training:
            best_match_idx = torch.linalg.norm(ego_fut_trajs[0].cumsum(dim=-2) - self.plan_anchors, dim=-1).sum(dim=-1).argmin() 
            # torch.linalg.norm(ego_traj_preds[:,:,:] - ego_fut_gt.cumsum(dim=-2)[:,:,:], dim=-1).sum(dim=-1) 
            if best_match_idx in used_index:
                pass
            else:
                used_index[-1] = best_match_idx

        self.used_plan_anchors = torch.index_select(self.plan_anchors, 0, used_index)


        # set stop traj to zero
        self.used_plan_anchors[self.used_plan_anchors[:,0].norm(dim=-1) < 1e-2] = 0.
        # fix one stop traj
        self.used_plan_anchors[0] = 0.

        _tmp = pos2posemb2d(self.used_plan_anchors.reshape(1, self.plan_fut_mode * self.fut_ts, -1), num_pos_feats=self.embed_dims // 2) \
                .reshape(self.plan_fut_mode, self.fut_ts, -1)
        ego_query = _tmp.unsqueeze(0).repeat(batch, 1, 1, 1).reshape(batch, self.plan_fut_mode, -1)


        ego_query = self.ego_query_pre_branch(ego_query)
        # ego_query = self.ego_query.weight.unsqueeze(0).repeat(batch, 1, 1)
        # ego-environment Interaction
        # ego<->agent query & pos
        agent_conf = outputs_classes[-1]
        agent_query = mot_hs.reshape(batch, num_agent, -1)
        agent_query = self.fus_mlp(agent_query) # [B, A, mot_fut_mode*2*D] -> [B, A, D]
        agent_pos = outputs_coords_bev[-1]

        agent_query, agent_pos, agent_mask = self.select_and_pad_query(
            agent_query, agent_pos, agent_conf,
            score_thresh=self.ego_query_thresh,
            use_fix_pad=self.query_use_fix_pad)

        if self.interaction_pe_type is not None:
            ego_agent_pos = torch.ones((batch, ego_query.shape[1], 2), device=ego_query.device)*0.5  # ego in the center
            ego_agent_pos = pos2posemb2d(ego_agent_pos, num_pos_feats=self.embed_dims // 2) if self.interaction_pe_type == 'sine_mlp' else ego_agent_pos
            ego_agent_pos = self.pe_embed_mlps[2](ego_agent_pos)
            agent_pos = pos2posemb2d(agent_pos, num_pos_feats=self.embed_dims // 2) if self.interaction_pe_type == 'sine_mlp' else agent_pos
            agent_pos = self.pe_embed_mlps[2](agent_pos)
            ego_agent_pos = ego_agent_pos.permute(1, 0, 2)
            agent_pos = agent_pos.permute(1, 0, 2)
        else:
            ego_agent_pos, agent_pos = None, None

        # ego <-> map query & pos
        map_query = map_hs[-1].view(batch_size, self.map_num_vec, self.map_num_pts_per_vec, -1)
        map_query = self.lane_encoder(map_query)  # [B, P, pts, D] -> [B, P, D]
        map_conf = map_outputs_classes[-1]
        map_pos = map_outputs_coords_bev[-1]
        # use the most close pts pos in each map inst as the inst's pos
        batch, num_map = map_pos.shape[:2]
        map_dis = torch.sqrt(map_pos[..., 0]**2 + map_pos[..., 1]**2)
        min_map_pos_idx = map_dis.argmin(dim=-1).flatten()  # [B*P]
        min_map_pos = map_pos.flatten(0, 1)  # [B*P, pts, 2]
        min_map_pos = min_map_pos[range(min_map_pos.shape[0]), min_map_pos_idx]  # [B*P, 2]
        min_map_pos = min_map_pos.view(batch, num_map, 2)  # [B, P, 2]
        map_query, map_pos, map_mask = self.select_and_pad_query(
            map_query, min_map_pos, map_conf,
            score_thresh=self.ego_query_thresh,
            use_fix_pad=self.query_use_fix_pad)

        if self.interaction_pe_type is not None:
            ego_map_pos = torch.ones((batch, ego_query.shape[1], 2), device=agent_query.device)*0.5  # ego in the center
            ego_map_pos = pos2posemb2d(ego_map_pos, num_pos_feats=self.embed_dims // 2) if self.interaction_pe_type == 'sine_mlp' else ego_map_pos
            ego_map_pos = self.pe_embed_mlps[3](ego_map_pos)
            map_pos = pos2posemb2d(map_pos, num_pos_feats=self.embed_dims // 2) if self.interaction_pe_type == 'sine_mlp' else map_pos
            map_pos = self.pe_embed_mlps[3](map_pos)
            ego_map_pos = ego_map_pos.permute(1, 0, 2)
            map_pos = map_pos.permute(1, 0, 2)
        else:
            ego_map_pos, map_pos = None, None

        # ego_pv_query = ego_query
        # ego <-> pv interaction
        batch, _, c_dim, _, _ = mlvl_feats[-1].shape
        attn_pv_feats = mlvl_feats[-1].permute(1, 3, 4, 0, 2).reshape(-1, batch, c_dim)
        ego_pv_query = self.ego_pv_decoder(
            query=ego_query.permute(1, 0, 2),
            key=attn_pv_feats,
            value=attn_pv_feats,
            query_pos=ego_agent_pos,
            key_pos=self.pv_pos_embedding.weight[:,None,:].repeat(1, batch, 1)[:attn_pv_feats.shape[0]]
        )
        
        # ego <-> agent interaction
        ego_agent_query = self.ego_agent_decoder(
            query=ego_pv_query,
            key=agent_query.permute(1, 0, 2),
            value=agent_query.permute(1, 0, 2),
            query_pos=ego_agent_pos,
            key_pos=agent_pos,
            key_padding_mask=agent_mask)

        # ego <-> map interaction
        ego_map_query = self.ego_map_decoder(
            query=ego_agent_query,
            key=map_query.permute(1, 0, 2),
            value=map_query.permute(1, 0, 2),
            query_pos=ego_map_pos,
            key_pos=map_pos,
            key_padding_mask=map_mask)

        # camera front feat -> embedding
        assert cf_img is not None
        # (B, 3, 320, 416) -> (B, 3, 160, 320)
        cf_img_h, cf_img_w = cf_img.shape[2:]
        crop_h = int(cf_img_h/2)
        crop_w1, crop_w2 = int(cf_img_w/4), int(cf_img_w*3/4)
        front_view_img = cf_img[:, :, :crop_h, crop_w1:crop_w2]
        cf_img_feats = self.cf_backbone(cf_img)
        if isinstance(cf_img_feats, dict):
            cf_img_feats = list(cf_img_feats.values())
        cf_img_feats = torch.cat((cf_img_feats[-1].flatten(1, 3),  mlvl_feats[-1].flatten(1, 4)), dim=-1)
        cf_img_feats = self.tl_feats_branch(cf_img_feats)
        cf_img_feats = cf_img_feats.unsqueeze(1)

        # Ego prediction
        assert self.ego_lcf_feat_idx is not None
        ego_pv_query = ego_pv_query.permute(1, 0, 2)
        ego_agent_query = ego_agent_query.permute(1, 0, 2)
        ego_map_query = ego_map_query.permute(1, 0, 2)
        ego_pv_feat = self.ego_feat_projs[6](ego_pv_query)
        ego_agent_feat = self.ego_feat_projs[0](ego_agent_query)
        ego_map_feat = self.ego_feat_projs[1](ego_map_query)
        ego_cf_feat = self.ego_feat_projs[2](cf_img_feats.clone().detach())
        # ego_wp_feat = self.ego_feat_projs[3](target_point.squeeze(1))
        if isinstance(target_point, torch.Tensor):
            _tmp_target_point = target_point.unsqueeze(1)
        else:
            _tmp_target_point = torch.tensor(target_point[None,None], device=ego_cf_feat.device)
        # range (-70m, +70m) grid_size 1m
        _tmp_rasterized_feat = torch.zeros((batch, 140, 140), dtype=torch.float32, device=ego_cf_feat.device)

        # TODO no need / 2.
        _idx = torch.floor((_tmp_target_point.clip(min=-69., max=69.) - (-70.)) / 2.).long()
        for i in range(batch):
            _tmp_rasterized_feat[i, _idx[i,0,0], _idx[i,0,1]] = 1.
        _tmp_rasterized_feat = _tmp_rasterized_feat.reshape(batch, 1, 140 * 140)
        ego_wp_feat = self.ego_feat_projs[3](_tmp_target_point)
        ego_wp_feat += 1. * self.ego_feat_projs[7](_tmp_rasterized_feat)


        # [VOID,LEFT,RIGHT,STRAIGHT,LANEFOLLOW,CHANGELANELEFT,CHANGELANERIGHT]
        if isinstance(command_id, torch.Tensor):
            cmdid_onehot = torch.zeros((batch, 1, 6), device=ego_cf_feat.device, dtype=torch.float32)
            assert command_id.max() <= 6 and command_id.min() >= 1
            for i in range(batch):
                cmdid_onehot[i, 0, command_id[i]-1] = 1.
            ego_cmdid_feat = self.ego_feat_projs[5](cmdid_onehot)
        else:
            cmdid_onehot = torch.zeros((batch, 1, 6), device=ego_cf_feat.device, dtype=torch.float32)
            assert command_id.max() <= 6 and command_id.min() >= 1
            assert batch == 1
            cmdid_onehot[0, 0, command_id - 1] = 1.
            ego_cmdid_feat = self.ego_feat_projs[5](cmdid_onehot)


        ego_status = ego_lcf_feat.squeeze(1)[..., self.ego_lcf_feat_idx]
        ego_status_feat = self.ego_feat_projs[4](ego_status)
        ego_feats = ego_agent_feat + ego_map_feat + \
                     1. * ego_wp_feat + 0. * ego_cmdid_feat + ego_status_feat + ego_cf_feat + 0. * ego_pv_feat

        outputs_ego_trajs = self.plan_reg_branch(ego_feats)
        outputs_ego_trajs = outputs_ego_trajs.reshape(outputs_ego_trajs.shape[0], 
                                                    self.plan_fut_mode, self.fut_ts, 2)
        # if self.training:                        
        #     outputs_ego_trajs = outputs_ego_trajs * 0. + self.plan_anchors[None].to(outputs_ego_trajs.device)
        # else:
        #     outputs_ego_trajs = outputs_ego_trajs * 0.  + self.centerline_trajs[None]
        outputs_ego_trajs = outputs_ego_trajs * 0. + self.used_plan_anchors[None].to(outputs_ego_trajs.device)

        # traffic light classification
        tl_status_cls_scores = self.tl_status_cls_branch(cf_img_feats)
        tl_trigger_cls_scores = self.tl_trigger_cls_branch(cf_img_feats)
        stopsign_trigger_cls_scores = self.stopsign_trigger_cls_branch(cf_img_feats)

        outputs_ego_cls_col = self.plan_cls_col_branch(ego_feats)
        outputs_ego_cls_bd = self.plan_cls_bd_branch(ego_feats)
        outputs_ego_cls_cl = self.plan_cls_cl_branch(ego_feats)
        outputs_ego_cls_expert = self.plan_cls_expert_branch(ego_feats)

        
        outs = {
            'bev_embed': bev_embed,
            'all_cls_scores': outputs_classes,
            'all_bbox_preds': outputs_coords,
            'all_mot_preds': outputs_mot_trajs.repeat(outputs_coords.shape[0], 1, 1, 1, 1),
            'all_mot_cls_scores': outputs_mot_trajs_classes.repeat(outputs_coords.shape[0], 1, 1, 1),
            'map_all_cls_scores': map_outputs_classes,
            'map_all_bbox_preds': map_outputs_coords,
            'map_all_pts_preds': map_outputs_pts_coords,
            'enc_cls_scores': None,
            'enc_bbox_preds': None,
            'map_enc_cls_scores': None,
            'map_enc_bbox_preds': None,
            'map_enc_pts_preds': None,
            'ego_fut_preds': outputs_ego_trajs,
            'tl_status_cls_scores': tl_status_cls_scores,
            'tl_trigger_cls_scores': tl_trigger_cls_scores,
            'stopsign_trigger_cls_scores': stopsign_trigger_cls_scores,
            'ego_cls_col_preds': outputs_ego_cls_col,
            'ego_cls_bd_preds': outputs_ego_cls_bd,
            'ego_cls_cl_preds': outputs_ego_cls_cl,
            'ego_cls_expert_preds': outputs_ego_cls_expert,
        }

        return outs

    def map_transform_box(self, pts, y_first=False):
        """
        Converting the points set into bounding box.

        Args:
            pts: the input points sets (fields), each points
                set (fields) is represented as 2n scalar.
            y_first: if y_fisrt=True, the point set is represented as
                [y1, x1, y2, x2 ... yn, xn], otherwise the point set is
                represented as [x1, y1, x2, y2 ... xn, yn].
        Returns:
            The bbox [cx, cy, w, h] transformed from points.
        """
        pts_reshape = pts.view(pts.shape[0], self.map_num_vec,
                                self.map_num_pts_per_vec,2)
        pts_y = pts_reshape[:, :, :, 0] if y_first else pts_reshape[:, :, :, 1]
        pts_x = pts_reshape[:, :, :, 1] if y_first else pts_reshape[:, :, :, 0]
        if self.map_transform_method == 'minmax':
            # import pdb;pdb.set_trace()

            xmin = pts_x.min(dim=2, keepdim=True)[0]
            xmax = pts_x.max(dim=2, keepdim=True)[0]
            ymin = pts_y.min(dim=2, keepdim=True)[0]
            ymax = pts_y.max(dim=2, keepdim=True)[0]
            bbox = torch.cat([xmin, ymin, xmax, ymax], dim=2)
            bbox = bbox_xyxy_to_cxcywh(bbox)
        else:
            raise NotImplementedError
        return bbox, pts_reshape

    def _get_target_single(self,
                           cls_score,
                           bbox_pred,
                           gt_labels,
                           gt_bboxes,
                           gt_attr_labels,
                           gt_bboxes_ignore=None):
        """"Compute regression and classification targets for one image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 10].
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 9) in [x,y,z,w,l,h,yaw,vx,vy] format.
            gt_labels (Tensor): Ground truth class indices for one image
                with shape (num_gts, ).
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.
        Returns:
            tuple[Tensor]: a tuple containing the following for one image.
                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """

        num_bboxes = bbox_pred.size(0)
        # assigner and sampler
        gt_mot_trajs = gt_attr_labels[:, :self.fut_ts*2]
        gt_mot_masks = gt_attr_labels[:, self.fut_ts*2:self.fut_ts*3]
        gt_bbox_c = gt_bboxes.shape[-1]
        num_gt_bbox, gt_mot_c = gt_mot_trajs.shape

        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            bbox_pred = torch.nan_to_num(bbox_pred)

        assign_result = self.assigner.assign(bbox_pred, cls_score, gt_bboxes,
                                             gt_labels, gt_bboxes_ignore)

        sampling_result = self.sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = gt_bboxes.new_full((num_bboxes,),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)[..., :gt_bbox_c]
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0

        # trajs targets
        mot_targets = torch.zeros((num_bboxes, gt_mot_c), dtype=torch.float32, device=bbox_pred.device)
        mot_weights = torch.zeros_like(mot_targets)
        mot_targets[pos_inds] = gt_mot_trajs[sampling_result.pos_assigned_gt_inds]
        mot_weights[pos_inds] = 1.0

        # Filter out invalid fut trajs
        mot_masks = torch.zeros_like(mot_targets)  # [num_bboxes, fut_ts*2]
        gt_mot_masks = gt_mot_masks.unsqueeze(-1).repeat(1, 1, 2).view(num_gt_bbox, -1)  # [num_gt_bbox, fut_ts*2]
        mot_masks[pos_inds] = gt_mot_masks[sampling_result.pos_assigned_gt_inds]
        mot_weights = mot_weights * mot_masks

        # Extra future timestamp mask for controlling pred horizon
        fut_ts_mask = torch.zeros((num_bboxes, self.fut_ts, 2),
                                   dtype=torch.float32, device=bbox_pred.device)
        fut_ts_mask[:, :self.valid_fut_ts, :] = 1.0
        fut_ts_mask = fut_ts_mask.view(num_bboxes, -1)
        mot_weights = mot_weights * fut_ts_mask

        # DETR
        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes

        return (
            labels, label_weights, bbox_targets, bbox_weights, mot_targets,
            mot_weights, mot_masks.view(-1, self.fut_ts, 2)[..., 0],
            pos_inds, neg_inds
        )

    def _map_get_target_single(self,
                           cls_score,
                           bbox_pred,
                           pts_pred,
                           gt_labels,
                           gt_bboxes,
                           gt_shifts_pts,
                           gt_bboxes_ignore=None):
        """"Compute regression and classification targets for one image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth class indices for one image
                with shape (num_gts, ).
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.
        Returns:
            tuple[Tensor]: a tuple containing the following for one image.
                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """
        # import pdb;pdb.set_trace()
        num_bboxes = bbox_pred.size(0)
        # assigner and sampler
        gt_c = gt_bboxes.shape[-1]
        # import pdb;pdb.set_trace()
        assign_result, order_index = self.map_assigner.assign(bbox_pred, cls_score, pts_pred,
                                             gt_bboxes, gt_labels, gt_shifts_pts,
                                             gt_bboxes_ignore)

        sampling_result = self.map_sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        # pts_sampling_result = self.sampler.sample(assign_result, pts_pred,
        #                                       gt_pts)

        
        # import pdb;pdb.set_trace()
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = gt_bboxes.new_full((num_bboxes,),
                                    self.map_num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)[..., :gt_c]
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0

        # pts targets
        # import pdb;pdb.set_trace()
        # pts_targets = torch.zeros_like(pts_pred)
        # num_query, num_order, num_points, num_coords
        if order_index is None:
            # import pdb;pdb.set_trace()
            assigned_shift = gt_labels[sampling_result.pos_assigned_gt_inds]
        else:
            assigned_shift = order_index[sampling_result.pos_inds, sampling_result.pos_assigned_gt_inds]
        pts_targets = pts_pred.new_zeros((pts_pred.size(0),
                        pts_pred.size(1), pts_pred.size(2)))
        pts_weights = torch.zeros_like(pts_targets)
        pts_weights[pos_inds] = 1.0

        # DETR
        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
        pts_targets[pos_inds] = gt_shifts_pts[sampling_result.pos_assigned_gt_inds,assigned_shift,:,:]
        return (labels, label_weights, bbox_targets, bbox_weights,
                pts_targets, pts_weights,
                pos_inds, neg_inds)

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_attr_labels_list,
                    gt_bboxes_ignore_list=None):
        """"Compute regression and classification targets for a batch image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            tuple: a tuple containing the following targets.
                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, mot_targets_list, mot_weights_list,
         gt_fut_masks_list, pos_inds_list, neg_inds_list) = multi_apply(
            self._get_target_single, cls_scores_list, bbox_preds_list,
            gt_labels_list, gt_bboxes_list, gt_attr_labels_list, gt_bboxes_ignore_list
         )
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
                mot_targets_list, mot_weights_list, gt_fut_masks_list, num_total_pos, num_total_neg)

    def map_get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    pts_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_shifts_pts_list,
                    gt_bboxes_ignore_list=None):
        """"Compute regression and classification targets for a batch image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            tuple: a tuple containing the following targets.
                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, pts_targets_list, pts_weights_list,
         pos_inds_list, neg_inds_list) = multi_apply(
            self._map_get_target_single, cls_scores_list, bbox_preds_list,pts_preds_list,
            gt_labels_list, gt_bboxes_list, gt_shifts_pts_list, gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, pts_targets_list, pts_weights_list,
                num_total_pos, num_total_neg)

    def loss_planning(self,
                      ego_fut_preds,
                      ego_fut_gt,
                      ego_fut_masks,
                      ego_fut_cmd,
                      lane_preds,
                      lane_score_preds,
                      agent_preds,
                      agent_fut_preds,
                      agent_score_preds,
                      agent_fut_cls_preds,
                      ego_cls_col_preds, 
                      ego_cls_bd_preds,
                      ego_cls_cl_preds,
                      ego_cls_expert_preds,
                      gt_agent_boxes,
                      gt_agent_feats,
                      gt_map_pts,
                      gt_map_labels,
                      ):
        """"Loss function for ego vehicle planning.
        Args:
            ego_fut_preds (Tensor): [B, num_cmd, fut_ts, 2]
            ego_fut_gt (Tensor): [B, fut_ts, 2]
            ego_fut_masks (Tensor): [B, fut_ts]
            ego_fut_cmd (Tensor): [B, num_cmd]
            lane_preds (Tensor): [B, num_vec, num_pts, 2]
            lane_score_preds (Tensor): [B, num_vec, 3]
            agent_preds (Tensor): [B, num_agent, 2]
            agent_fut_preds (Tensor): [B, num_agent, mot_fut_mode, fut_ts, 2]
            agent_score_preds (Tensor): [B, num_agent, 10]
            agent_fut_cls_scores (Tensor): [B, num_agent, mot_fut_mode]
            ego_cls_col_preds (Tensor): [B, num_plan_mode, 1]
            ego_cls_bd_preds (Tensor): [B, num_plan_mode, 1]
            ego_cls_cl_preds (Tensor): [B, num_plan_mode, 1]
            ego_cls_expert_preds (Tensor): [B, num_plan_mode, 1]

        Returns:
            loss_plan_cls_col (Tensor): cls col loss.
            loss_plan_cls_bd (Tensor): cls bd loss.
            loss_plan_cls_cl (Tensor): cls cl loss.
            loss_plan_cls_expert (Tensor): cls expert loss.
            loss_plan_reg (Tensor): planning l1 loss.
            loss_plan_bound (Tensor): planning map boundary loss.
            loss_plan_agent_dis (Tensor): planning agent distance loss.
            loss_plan_map_theta (Tensor): planning map theta loss.
        """

        batch = ego_fut_preds.shape[0]
        ego_fut_gt = ego_fut_gt.unsqueeze(1).repeat(1, self.plan_fut_mode, 1, 1)
        # loss_plan_l1_weight = ego_fut_cmd[..., None, None] * ego_fut_masks[:, None, :, None]
        # loss_plan_l1_weight = loss_plan_l1_weight.repeat(1, 1, 1, 2)

        # get plan cls target
        plan_col_labels, plan_bd_labels, plan_cl_labels = [], [], []
        plan_expert_labels, plan_expert_labels_weight = [], []
        for i in range(batch):
            plan_col_label = self.get_plan_col_target(
                ego_fut_preds[i].detach(),
                ego_fut_gt[i],
                gt_agent_boxes[i],
                gt_agent_feats[i])
            plan_bd_label = self.get_plan_bd_target(
                ego_fut_preds[i].detach(),
                gt_map_pts[i],
                gt_map_labels[i])
            plan_cl_label = self.get_plan_cl_target(
                ego_fut_preds[i].detach(),
                gt_map_pts[i],
                gt_map_labels[i])
            plan_expert_label, plan_expert_label_weight = self.get_plan_expert_target(
                ego_fut_preds[i].detach(),
                ego_fut_gt[i],
                ego_fut_masks[i],
                ego_cls_expert_preds[i],
                plan_col_label,
                plan_bd_label)

            plan_col_labels.append(plan_col_label)
            plan_bd_labels.append(plan_bd_label)
            plan_cl_labels.append(plan_cl_label)
            plan_expert_labels.append(plan_expert_label)
            plan_expert_labels_weight.append(plan_expert_label_weight)

        plan_col_labels = torch.stack(plan_col_labels, dim=0).to(ego_fut_preds.device)
        plan_bd_labels = torch.stack(plan_bd_labels, dim=0).to(ego_fut_preds.device)
        plan_cl_labels = torch.stack(plan_cl_labels, dim=0).to(ego_fut_preds.device)
        plan_expert_labels = torch.stack(plan_expert_labels, dim=0).to(ego_fut_preds.device)
        plan_expert_labels_weight = torch.stack(plan_expert_labels_weight,
                                                dim=0).to(ego_fut_preds.device)

        # plan collision classification loss
        loss_plan_cls_col = self.loss_plan_cls_col(
            ego_cls_col_preds.flatten(0, 1), plan_col_labels.flatten(),
            plan_col_labels.new_ones(batch*self.plan_fut_mode),
            avg_factor=batch*self.plan_fut_mode)

        # plan boundary overstepping classification loss
        loss_plan_cls_bd = self.loss_plan_cls_bd(
            ego_cls_bd_preds.flatten(0, 1), plan_bd_labels.flatten(),
            plan_bd_labels.new_ones(batch*self.plan_fut_mode),
            avg_factor=batch*self.plan_fut_mode)

        # plan centerline consistency classification loss
        plan_cl_weight = plan_cl_labels.flatten()
        plan_cl_labels_weight = (plan_cl_weight > -1.)  + (plan_cl_weight == -1.) * 0.01
        loss_plan_cls_cl = self.loss_plan_cls_cl(
            ego_cls_cl_preds.squeeze(-1).flatten(0, 1), plan_cl_labels.flatten(),
            plan_cl_labels_weight,
            # avg_factor=(plan_cl_labels > -1.).sum()
        )

        # plan expert driving behavior classification loss
        loss_plan_cls_expert = self.loss_plan_cls_expert(
            ego_cls_expert_preds.flatten(0, 1), plan_expert_labels.flatten(),
            plan_expert_labels_weight.flatten(),
            avg_factor=batch*self.plan_fut_mode)

        loss_plan_reg = (0. * ego_fut_preds).sum()
        # loss_plan_reg = self.loss_plan_reg(
        #     ego_fut_preds,
        #     ego_fut_gt,
        #     loss_plan_l1_weight
        # )

        loss_plan_bound = (0. * ego_fut_preds).sum()
        # loss_plan_bound = self.loss_plan_bound(
        #     ego_fut_preds[ego_fut_cmd==1],
        #     lane_preds,
        #     lane_score_preds,
        #     weight=ego_fut_masks
        # )

        loss_plan_agent_dis = (0. * ego_fut_preds).sum()
        # loss_plan_agent_dis = self.loss_plan_agent_dis(
        #     ego_fut_preds[ego_fut_cmd==1],
        #     agent_preds,
        #     agent_fut_preds,
        #     agent_score_preds,
        #     agent_fut_cls_preds,
        #     weight=ego_fut_masks[:, :, None].repeat(1, 1, 2)
        # )

        loss_plan_map_theta = (0. * ego_fut_preds).sum()
        # loss_plan_map_theta = self.loss_plan_map_theta(
        #     ego_fut_preds[ego_fut_cmd==1],
        #     lane_preds,
        #     lane_score_preds,
        #     weight=ego_fut_masks
        # )

        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            loss_plan_cls_col = torch.nan_to_num(loss_plan_cls_col)
            loss_plan_cls_bd = torch.nan_to_num(loss_plan_cls_bd)
            loss_plan_cls_cl = torch.nan_to_num(loss_plan_cls_cl)
            loss_plan_cls_expert = torch.nan_to_num(loss_plan_cls_expert)
            loss_plan_reg = torch.nan_to_num(loss_plan_reg)
            loss_plan_bound = torch.nan_to_num(loss_plan_bound)
            loss_plan_agent_dis = torch.nan_to_num(loss_plan_agent_dis)
            loss_plan_map_theta = torch.nan_to_num(loss_plan_map_theta)
        
        loss_plan_dict = dict()
        loss_plan_dict['loss_plan_cls_col'] = loss_plan_cls_col
        loss_plan_dict['loss_plan_cls_bd'] = loss_plan_cls_bd
        loss_plan_dict['loss_plan_cls_cl'] = loss_plan_cls_cl
        loss_plan_dict['loss_plan_cls_expert'] = loss_plan_cls_expert

        loss_plan_dict['loss_plan_reg'] = loss_plan_reg
        loss_plan_dict['loss_plan_bound'] = loss_plan_bound
        loss_plan_dict['loss_plan_agent_dis'] = loss_plan_agent_dis
        loss_plan_dict['loss_plan_map_theta'] = loss_plan_map_theta

        return loss_plan_dict
    
    def loss_single(self,
                    cls_scores,
                    bbox_preds,
                    mot_preds,
                    mot_cls_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_attr_labels_list,
                    gt_bboxes_ignore_list=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           gt_bboxes_list, gt_labels_list,
                                           gt_attr_labels_list, gt_bboxes_ignore_list)

        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         mot_targets_list, mot_weights_list, gt_fut_masks_list,
         num_total_pos, num_total_neg) = cls_reg_targets

        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)
        mot_targets = torch.cat(mot_targets_list, 0)
        mot_weights = torch.cat(mot_weights_list, 0)
        gt_fut_masks = torch.cat(gt_fut_masks_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights
        loss_bbox = self.loss_bbox(
            bbox_preds[isnotnan, :10],
            normalized_bbox_targets[isnotnan, :10],
            bbox_weights[isnotnan, :10],
            avg_factor=num_total_pos
        )

        # mot regression loss
        best_mot_preds = self.get_best_fut_preds(
            mot_preds.reshape(-1, self.mot_fut_mode, self.fut_ts, 2),
            mot_targets.reshape(-1, self.fut_ts, 2),
            gt_fut_masks
        )

        neg_inds = (bbox_weights[:, 0] == 0)
        mot_labels = self.get_mot_cls_target(
            mot_preds.reshape(-1, self.mot_fut_mode, self.fut_ts, 2),
            mot_targets.reshape(-1, self.fut_ts, 2),
            gt_fut_masks,
            neg_inds
        )

        loss_mot_reg = self.loss_mot_reg(
            best_mot_preds[isnotnan],
            mot_targets[isnotnan],
            mot_weights[isnotnan],
            avg_factor=num_total_pos
        )

        # mot classification loss
        mot_cls_scores = mot_cls_preds.reshape(-1, self.mot_fut_mode)
        # construct weighted avg_factor to match with the official DETR repo
        mot_cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.mot_bg_cls_weight
        if self.sync_cls_avg_factor:
            mot_cls_avg_factor = reduce_mean(
                mot_cls_scores.new_tensor([mot_cls_avg_factor]))

        mot_cls_avg_factor = max(mot_cls_avg_factor, 1)
        loss_mot_cls = self.loss_mot_cls(
            mot_cls_scores, mot_labels, label_weights, avg_factor=mot_cls_avg_factor
        )

        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            loss_cls = torch.nan_to_num(loss_cls)
            loss_bbox = torch.nan_to_num(loss_bbox)
            loss_mot_reg = torch.nan_to_num(loss_mot_reg)
            loss_mot_cls = torch.nan_to_num(loss_mot_cls)

        return loss_cls, loss_bbox, loss_mot_reg, loss_mot_cls

    def get_best_fut_preds(self,
             traj_preds,
             traj_targets,
             gt_fut_masks):
        """"Choose best preds among all modes.
        Args:
            traj_preds (Tensor): MultiModal traj preds with shape (num_box_preds, mot_fut_mode, fut_ts, 2).
            traj_targets (Tensor): Ground truth traj for each pred box with shape (num_box_preds, fut_ts, 2).
            gt_fut_masks (Tensor): Ground truth traj mask with shape (num_box_preds, fut_ts).
            pred_box_centers (Tensor): Pred box centers with shape (num_box_preds, 2).
            gt_box_centers (Tensor): Ground truth box centers with shape (num_box_preds, 2).

        Returns:
            best_traj_preds (Tensor): best traj preds (min displacement error with gt)
                with shape (num_box_preds, fut_ts*2).
        """

        cum_traj_preds = traj_preds.cumsum(dim=-2)
        cum_traj_targets = traj_targets.cumsum(dim=-2)

        # Get min pred mode indices.
        # (num_box_preds, mot_fut_mode, fut_ts)
        dist = torch.linalg.norm(cum_traj_targets[:, None, :, :] - cum_traj_preds, dim=-1)
        dist = dist * gt_fut_masks[:, None, :]
        dist = dist[..., -1]
        dist[torch.isnan(dist)] = dist[torch.isnan(dist)] * 0
        min_mode_idxs = torch.argmin(dist, dim=-1).tolist()
        box_idxs = torch.arange(traj_preds.shape[0]).tolist()
        best_traj_preds = traj_preds[box_idxs, min_mode_idxs, :, :].reshape(-1, self.fut_ts*2)

        return best_traj_preds

    def get_mot_cls_target(self,
             mot_preds,
             mot_targets,
             gt_fut_masks,
             neg_inds):
        """"Get motion trajectory mode classification target.
        Args:
            mot_preds (Tensor): MultiModal traj preds with shape (num_box_preds, mot_fut_mode, fut_ts, 2).
            mot_targets (Tensor): Ground truth traj for each pred box with shape (num_box_preds, fut_ts, 2).
            gt_fut_masks (Tensor): Ground truth traj mask with shape (num_box_preds, fut_ts).
            neg_inds (Tensor): Negtive indices with shape (num_box_preds,)

        Returns:
            mot_labels (Tensor): traj cls labels (num_box_preds,).
        """

        cum_mot_preds = mot_preds.cumsum(dim=-2)
        cum_mot_targets = mot_targets.cumsum(dim=-2)

        # Get min pred mode indices.
        # (num_box_preds, mot_fut_mode, fut_ts)
        dist = torch.linalg.norm(cum_mot_targets[:, None, :, :] - cum_mot_preds, dim=-1)
        dist = dist * gt_fut_masks[:, None, :]
        dist = dist[..., -1]
        dist[torch.isnan(dist)] = dist[torch.isnan(dist)] * 0
        mot_labels = torch.argmin(dist, dim=-1)
        mot_labels[neg_inds] = self.mot_fut_mode

        return mot_labels

    def map_loss_single(self,
                    cls_scores,
                    bbox_preds,
                    pts_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_shifts_pts_list,
                    gt_bboxes_ignore_list=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_pts_list (list[Tensor]): Ground truth pts for each image
                with shape (num_gts, fixed_num, 2) in [x,y] format.
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        pts_preds_list = [pts_preds[i] for i in range(num_imgs)]
        # import pdb;pdb.set_trace()
        cls_reg_targets = self.map_get_targets(cls_scores_list, bbox_preds_list,pts_preds_list,
                                           gt_bboxes_list, gt_labels_list,gt_shifts_pts_list,
                                           gt_bboxes_ignore_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         pts_targets_list, pts_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        # import pdb;pdb.set_trace()
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)
        pts_targets = torch.cat(pts_targets_list, 0)
        pts_weights = torch.cat(pts_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.map_cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.map_bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_map_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # import pdb;pdb.set_trace()
        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_2d_bbox(bbox_targets, self.pc_range)
        # normalized_bbox_targets = bbox_targets
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.map_code_weights

        loss_bbox = self.loss_map_bbox(
            bbox_preds[isnotnan, :4],
            normalized_bbox_targets[isnotnan,:4],
            bbox_weights[isnotnan, :4],
            avg_factor=num_total_pos)

        # regression pts CD loss
        # pts_preds = pts_preds
        # import pdb;pdb.set_trace()
        
        # num_samples, num_order, num_pts, num_coords
        normalized_pts_targets = normalize_2d_pts(pts_targets, self.pc_range)

        # num_samples, num_pts, num_coords
        pts_preds = pts_preds.reshape(-1, pts_preds.size(-2), pts_preds.size(-1))
        if self.map_num_pts_per_vec != self.map_num_pts_per_gt_vec:
            pts_preds = pts_preds.permute(0,2,1)
            pts_preds = F.interpolate(pts_preds, size=(self.map_num_pts_per_gt_vec), mode='linear',
                                    align_corners=True)
            pts_preds = pts_preds.permute(0,2,1).contiguous()

        # import pdb;pdb.set_trace()
        loss_pts = self.loss_map_pts(
            pts_preds[isnotnan,:,:],
            normalized_pts_targets[isnotnan,:,:], 
            pts_weights[isnotnan,:,:],
            avg_factor=num_total_pos)

        dir_weights = pts_weights[:, :-self.map_dir_interval,0]
        denormed_pts_preds = denormalize_2d_pts(pts_preds, self.pc_range)
        denormed_pts_preds_dir = denormed_pts_preds[:,self.map_dir_interval:,:] - \
            denormed_pts_preds[:,:-self.map_dir_interval,:]
        pts_targets_dir = pts_targets[:, self.map_dir_interval:,:] - pts_targets[:,:-self.map_dir_interval,:]
        # dir_weights = pts_weights[:, indice,:-1,0]
        # import pdb;pdb.set_trace()
        loss_dir = self.loss_map_dir(
            denormed_pts_preds_dir[isnotnan,:,:],
            pts_targets_dir[isnotnan,:,:],
            dir_weights[isnotnan,:],
            avg_factor=num_total_pos)

        bboxes = denormalize_2d_bbox(bbox_preds, self.pc_range)
        # regression IoU loss, defaultly GIoU loss
        loss_iou = self.loss_map_iou(
            bboxes[isnotnan, :4],
            bbox_targets[isnotnan, :4],
            bbox_weights[isnotnan, :4], 
            avg_factor=num_total_pos)

        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            loss_cls = torch.nan_to_num(loss_cls)
            loss_bbox = torch.nan_to_num(loss_bbox)
            loss_iou = torch.nan_to_num(loss_iou)
            loss_pts = torch.nan_to_num(loss_pts)
            loss_dir = torch.nan_to_num(loss_dir)
        return loss_cls, loss_bbox, loss_iou, loss_pts, loss_dir

    # NOTE: already support map
    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             gt_bboxes_list,
             gt_labels_list,
             map_gt_bboxes_list,
             map_gt_labels_list,
             preds_dicts,
             ego_fut_gt,
             ego_fut_masks,
             ego_fut_cmd,
             gt_attr_labels,
             traffic_signal,
             stop_sign_signal,
             gt_bboxes_ignore=None,
             map_gt_bboxes_ignore=None,
             img_metas=None):
        """"Loss function.
        Args:

            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            preds_dicts:
                all_cls_scores (Tensor): Classification score of all
                    decoder layers, has shape
                    [nb_dec, bs, num_query, cls_out_channels].
                all_bbox_preds (Tensor): Sigmoid regression
                    outputs of all decode layers. Each is a 4D-tensor with
                    normalized coordinate format (cx, cy, w, h) and shape
                    [nb_dec, bs, num_query, 4].
                enc_cls_scores (Tensor): Classification scores of
                    points on encode feature map , has shape
                    (N, h*w, num_classes). Only be passed when as_two_stage is
                    True, otherwise is None.
                enc_bbox_preds (Tensor): Regression results of each points
                    on the encode feature map, has shape (N, h*w, 4). Only be
                    passed when as_two_stage is True, otherwise is None.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'

        map_gt_vecs_list = copy.deepcopy(map_gt_bboxes_list)

        all_cls_scores = preds_dicts['all_cls_scores']
        all_bbox_preds = preds_dicts['all_bbox_preds']
        all_mot_preds = preds_dicts['all_mot_preds']
        all_mot_cls_scores = preds_dicts['all_mot_cls_scores']
        enc_cls_scores = preds_dicts['enc_cls_scores']
        enc_bbox_preds = preds_dicts['enc_bbox_preds']
        map_all_cls_scores = preds_dicts['map_all_cls_scores']
        map_all_bbox_preds = preds_dicts['map_all_bbox_preds']
        map_all_pts_preds = preds_dicts['map_all_pts_preds']
        map_enc_cls_scores = preds_dicts['map_enc_cls_scores']
        map_enc_bbox_preds = preds_dicts['map_enc_bbox_preds']
        map_enc_pts_preds = preds_dicts['map_enc_pts_preds']
        ego_fut_preds = preds_dicts['ego_fut_preds']
        tl_status_cls_scores = preds_dicts['tl_status_cls_scores']
        tl_trigger_cls_scores = preds_dicts['tl_trigger_cls_scores']
        stopsign_trigger_cls_scores = preds_dicts['stopsign_trigger_cls_scores']
        ego_cls_col_preds = preds_dicts['ego_cls_col_preds']
        ego_cls_bd_preds = preds_dicts['ego_cls_bd_preds']
        ego_cls_cl_preds = preds_dicts['ego_cls_cl_preds']
        ego_cls_expert_preds = preds_dicts['ego_cls_expert_preds']


        num_dec_layers = len(all_cls_scores)
        device = gt_labels_list[0].device

        gt_bboxes_list = [torch.cat(
            (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),
            dim=1).to(device) for gt_bboxes in gt_bboxes_list]

        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_attr_labels_list = [gt_attr_labels for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]

        losses_cls, losses_bbox, loss_mot_reg, loss_mot_cls = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds, all_mot_preds,
            all_mot_cls_scores, all_gt_bboxes_list,
            all_gt_labels_list, all_gt_attr_labels_list, all_gt_bboxes_ignore_list)
        

        num_dec_layers = len(map_all_cls_scores)
        device = map_gt_labels_list[0].device
        # gt_bboxes_list = [torch.cat(
        #     (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),
        #     dim=1).to(device) for gt_bboxes in gt_bboxes_list]
        # import pdb;pdb.set_trace()
        # gt_bboxes_list = [
        #     gt_bboxes.to(device) for gt_bboxes in gt_bboxes_list]
        map_gt_bboxes_list = [
            map_gt_bboxes.bbox.to(device) for map_gt_bboxes in map_gt_vecs_list]
        map_gt_pts_list = [
            map_gt_bboxes.fixed_num_sampled_points.to(device) for map_gt_bboxes in map_gt_vecs_list]
        if self.map_gt_shift_pts_pattern == 'v0':
            map_gt_shifts_pts_list = [
                gt_bboxes.shift_fixed_num_sampled_points.to(device) for gt_bboxes in map_gt_vecs_list]
        elif self.map_gt_shift_pts_pattern == 'v1':
            map_gt_shifts_pts_list = [
                gt_bboxes.shift_fixed_num_sampled_points_v1.to(device) for gt_bboxes in map_gt_vecs_list]
        elif self.map_gt_shift_pts_pattern == 'v2':
            map_gt_shifts_pts_list = [
                gt_bboxes.shift_fixed_num_sampled_points_v2.to(device) for gt_bboxes in map_gt_vecs_list]
        elif self.map_gt_shift_pts_pattern == 'v3':
            map_gt_shifts_pts_list = [
                gt_bboxes.shift_fixed_num_sampled_points_v3.to(device) for gt_bboxes in map_gt_vecs_list]
        elif self.map_gt_shift_pts_pattern == 'v4':
            map_gt_shifts_pts_list = [
                gt_bboxes.shift_fixed_num_sampled_points_v4.to(device) for gt_bboxes in map_gt_vecs_list]
        else:
            raise NotImplementedError
        map_all_gt_bboxes_list = [map_gt_bboxes_list for _ in range(num_dec_layers)]
        map_all_gt_labels_list = [map_gt_labels_list for _ in range(num_dec_layers)]
        map_all_gt_pts_list = [map_gt_pts_list for _ in range(num_dec_layers)]
        map_all_gt_shifts_pts_list = [map_gt_shifts_pts_list for _ in range(num_dec_layers)]
        map_all_gt_bboxes_ignore_list = [
            map_gt_bboxes_ignore for _ in range(num_dec_layers)
        ]
        # import pdb;pdb.set_trace()
        map_losses_cls, map_losses_bbox, map_losses_iou, \
            map_losses_pts, map_losses_dir = multi_apply(
            self.map_loss_single, map_all_cls_scores, map_all_bbox_preds,
            map_all_pts_preds, map_all_gt_bboxes_list, map_all_gt_labels_list,
            map_all_gt_shifts_pts_list, map_all_gt_bboxes_ignore_list)

        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]
        loss_dict['loss_mot_reg'] = loss_mot_reg[-1]
        loss_dict['loss_mot_cls'] = loss_mot_cls[-1]
        # loss from the last decoder layer
        loss_dict['loss_map_cls'] = map_losses_cls[-1]
        loss_dict['loss_map_bbox'] = map_losses_bbox[-1]
        loss_dict['loss_map_iou'] = map_losses_iou[-1]
        loss_dict['loss_map_pts'] = map_losses_pts[-1]
        loss_dict['loss_map_dir'] = map_losses_dir[-1]

        # Planning Loss
        ego_fut_gt = ego_fut_gt.squeeze(1)
        ego_fut_masks = ego_fut_masks.squeeze(1).squeeze(1)
        ego_fut_cmd = ego_fut_cmd.squeeze(1).squeeze(1)

        batch, num_agent = all_mot_preds[-1].shape[:2]
        agent_fut_preds = all_mot_preds[-1].view(batch, num_agent, self.mot_fut_mode, self.fut_ts, 2)
        agent_fut_cls_preds = all_mot_cls_scores[-1].view(batch, num_agent, self.mot_fut_mode)
        loss_plan_input = [ego_fut_preds, ego_fut_gt, ego_fut_masks, ego_fut_cmd,
                           map_all_pts_preds[-1], map_all_cls_scores[-1].sigmoid(),
                           all_bbox_preds[-1][..., 0:2], agent_fut_preds,
                           all_cls_scores[-1].sigmoid(), agent_fut_cls_preds.sigmoid(),
                           ego_cls_col_preds, ego_cls_bd_preds, ego_cls_cl_preds, ego_cls_expert_preds,
                           gt_bboxes_list, gt_attr_labels, 
                           map_gt_pts_list, map_gt_labels_list]

        loss_planning_dict = self.loss_planning(*loss_plan_input)
        loss_dict['loss_plan_cls_col'] = loss_planning_dict['loss_plan_cls_col']
        loss_dict['loss_plan_cls_bd'] = loss_planning_dict['loss_plan_cls_bd']
        loss_dict['loss_plan_cls_cl'] = loss_planning_dict['loss_plan_cls_cl']
        loss_dict['loss_plan_cls_expert'] = loss_planning_dict['loss_plan_cls_expert']
        loss_dict['loss_plan_reg'] = loss_planning_dict['loss_plan_reg']
        loss_dict['loss_plan_bound'] = loss_planning_dict['loss_plan_bound']
        loss_dict['loss_plan_agent_dis'] = loss_planning_dict['loss_plan_agent_dis']
        loss_dict['loss_plan_map_theta'] = loss_planning_dict['loss_plan_map_theta']


        # traffic light trigger classification
        tl_trigger_cls_scores = tl_trigger_cls_scores.reshape(-1, self.tl_trigger_num_cls)
        tl_trigger_labels = traffic_signal[..., 1].reshape(-1)
        tl_trigger_cls_avg_factor = tl_trigger_cls_scores.shape[0] * 1.0
        if self.sync_cls_avg_factor:
            tl_trigger_cls_avg_factor = reduce_mean(
                tl_trigger_cls_scores.new_tensor([tl_trigger_cls_avg_factor]))
        tl_trigger_cls_avg_factor = max(tl_trigger_cls_avg_factor, 1)
        loss_tl_trigger_cls = self.loss_tl_trigger_cls(
            tl_trigger_cls_scores, tl_trigger_labels,
            tl_trigger_cls_scores.new_ones(tl_trigger_labels.shape[0]),
            avg_factor=tl_trigger_cls_avg_factor)
        
        # stop sign trigger classification
        stopsign_trigger_cls_scores = stopsign_trigger_cls_scores.reshape(-1, self.stopsign_trigger_num_cls)
        stopsign_trigger_labels = stop_sign_signal.reshape(-1)
        stopsign_trigger_cls_avg_factor = stopsign_trigger_cls_scores.shape[0] * 1.0
        if self.sync_cls_avg_factor:
            stopsign_trigger_cls_avg_factor = reduce_mean(
                stopsign_trigger_cls_scores.new_tensor([stopsign_trigger_cls_avg_factor]))
        stopsign_trigger_cls_avg_factor = max(stopsign_trigger_cls_avg_factor, 1)
        loss_stopsign_trigger_cls = self.loss_stopsign_trigger_cls(
            stopsign_trigger_cls_scores, stopsign_trigger_labels,
            stopsign_trigger_cls_scores.new_ones(stopsign_trigger_labels.shape[0]),
            avg_factor=stopsign_trigger_cls_avg_factor)
        
        # traffic light status classification
        tl_status_weights = 1 - tl_trigger_labels
        tl_status_cls_scores = tl_status_cls_scores.reshape(-1, self.tl_status_num_cls)
        tl_status_labels = traffic_signal[..., 0].reshape(-1)
        tl_status_cls_avg_factor = tl_status_cls_scores.shape[0] * 1.0
        if self.sync_cls_avg_factor:
            tl_status_cls_avg_factor = reduce_mean(
                tl_status_cls_scores.new_tensor([tl_status_cls_avg_factor]))
        tl_status_cls_avg_factor = max(tl_status_cls_avg_factor, 1)
        loss_tl_status_cls = self.loss_tl_status_cls(
            tl_status_cls_scores, tl_status_labels,
            tl_status_weights,
            avg_factor=tl_status_cls_avg_factor)

        loss_dict['loss_tl_status_cls'] = loss_tl_status_cls
        loss_dict['loss_tl_trigger_cls'] = loss_tl_trigger_cls
        loss_dict['loss_stopsign_trigger_cls'] = loss_stopsign_trigger_cls

        # det loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i in zip(losses_cls[:-1], losses_bbox[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            num_dec_layer += 1

        # map loss from other decoder layers
        num_dec_layer = 0
        for map_loss_cls_i, map_loss_bbox_i, map_loss_iou_i, map_loss_pts_i, map_loss_dir_i in zip(
            map_losses_cls[:-1],
            map_losses_bbox[:-1],
            map_losses_iou[:-1],
            map_losses_pts[:-1],
            map_losses_dir[:-1]
        ):
            loss_dict[f'd{num_dec_layer}.loss_map_cls'] = map_loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_map_bbox'] = map_loss_bbox_i
            loss_dict[f'd{num_dec_layer}.loss_map_iou'] = map_loss_iou_i
            loss_dict[f'd{num_dec_layer}.loss_map_pts'] = map_loss_pts_i
            loss_dict[f'd{num_dec_layer}.loss_map_dir'] = map_loss_dir_i
            num_dec_layer += 1

        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            binary_labels_list = [
                torch.zeros_like(gt_labels_list[i])
                for i in range(len(all_gt_labels_list))
            ]
            enc_loss_cls, enc_losses_bbox = \
                self.loss_single(enc_cls_scores, enc_bbox_preds,
                                 gt_bboxes_list, binary_labels_list, gt_bboxes_ignore)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_losses_bbox

        if map_enc_cls_scores is not None:
            map_binary_labels_list = [
                torch.zeros_like(map_gt_labels_list[i])
                for i in range(len(map_all_gt_labels_list))
            ]
            # TODO bug here, but we dont care enc_loss now
            map_enc_loss_cls, map_enc_loss_bbox, map_enc_loss_iou, \
                 map_enc_loss_pts, map_enc_loss_dir = \
                self.map_loss_single(
                    map_enc_cls_scores, map_enc_bbox_preds,
                    map_enc_pts_preds, map_gt_bboxes_list,
                    map_binary_labels_list, map_gt_pts_list,
                    map_gt_bboxes_ignore
                )
            loss_dict['enc_loss_map_cls'] = map_enc_loss_cls
            loss_dict['enc_loss_map_bbox'] = map_enc_loss_bbox
            loss_dict['enc_loss_map_iou'] = map_enc_loss_iou
            loss_dict['enc_loss_map_pts'] = map_enc_loss_pts
            loss_dict['enc_loss_map_dir'] = map_enc_loss_dir

        return loss_dict

    # NOTE: already support map
    @force_fp32(apply_to=('preds_dicts'))
    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """

        det_preds_dicts = self.bbox_coder.decode(preds_dicts)
        # map_bboxes: xmin, ymin, xmax, ymax
        map_preds_dicts = self.map_bbox_coder.decode(preds_dicts)

        num_samples = len(det_preds_dicts)
        assert len(det_preds_dicts) == len(map_preds_dicts), \
             'len(preds_dict) should be equal to len(map_preds_dicts)'
        ret_list = []
        for i in range(num_samples):
            preds = det_preds_dicts[i]
            bboxes = preds['bboxes']
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
            code_size = bboxes.shape[-1]
            bboxes = img_metas[i]['box_type_3d'](bboxes, code_size)
            scores = preds['scores']
            labels = preds['labels']
            trajs = preds['trajs']
            trajs_cls = preds['trajs_cls']

            map_preds = map_preds_dicts[i]
            map_bboxes = map_preds['map_bboxes']
            # bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
            # code_size = bboxes.shape[-1]
            # bboxes = img_metas[i]['box_type_3d'](bboxes, code_size)
            map_scores = map_preds['map_scores']
            map_labels = map_preds['map_labels']
            map_pts = map_preds['map_pts']

            ret_list.append([bboxes, scores, labels, trajs, trajs_cls, map_bboxes,
                             map_scores, map_labels, map_pts])

        return ret_list

    def select_and_pad_pred_map(
        self,
        mot_pos,
        map_query,
        map_score,
        map_pos,
        map_thresh=0.5,
        dis_thresh=None,
        pe_normalization=True,
        use_fix_pad=False
    ):
        """select_and_pad_pred_map.
        Args:
            mot_pos: [B, A, 2]
            map_query: [B, P, D].
            map_score: [B, P, 3].
            map_pos: [B, P, pts, 2].
            map_thresh: map confidence threshold for filtering low-confidence preds
            dis_thresh: distance threshold for masking far maps for each agent in cross-attn
            use_fix_pad: always pad one lane instance for each batch
        Returns:
            selected_map_query: [B*A, P1(+1), D], P1 is the max inst num after filter and pad.
            selected_map_pos: [B*A, P1(+1), 2]
            selected_padding_mask: [B*A, P1(+1)]
        """
        
        if dis_thresh is None:
            raise NotImplementedError('Not implement yet')

        # use the most close pts pos in each map inst as the inst's pos
        batch, num_map = map_pos.shape[:2]
        map_dis = torch.sqrt(map_pos[..., 0]**2 + map_pos[..., 1]**2)
        min_map_pos_idx = map_dis.argmin(dim=-1).flatten()  # [B*P]
        min_map_pos = map_pos.flatten(0, 1)  # [B*P, pts, 2]
        min_map_pos = min_map_pos[range(min_map_pos.shape[0]), min_map_pos_idx]  # [B*P, 2]
        min_map_pos = min_map_pos.view(batch, num_map, 2)  # [B, P, 2]

        # select & pad map vectors for different batch using map_thresh
        map_score = map_score.sigmoid()
        map_max_score = map_score.max(dim=-1)[0]
        map_idx = map_max_score > map_thresh
        batch_max_pnum = 0
        for i in range(map_score.shape[0]):
            pnum = map_idx[i].sum()
            if pnum > batch_max_pnum:
                batch_max_pnum = pnum

        selected_map_query, selected_map_pos, selected_padding_mask = [], [], []
        for i in range(map_score.shape[0]):
            dim = map_query.shape[-1]
            valid_pnum = map_idx[i].sum()
            valid_map_query = map_query[i, map_idx[i]]
            valid_map_pos = min_map_pos[i, map_idx[i]]
            pad_pnum = batch_max_pnum - valid_pnum
            padding_mask = torch.tensor([False], device=map_score.device).repeat(batch_max_pnum)
            if pad_pnum != 0:
                valid_map_query = torch.cat([valid_map_query, torch.zeros((pad_pnum, dim), device=map_score.device)], dim=0)
                valid_map_pos = torch.cat([valid_map_pos, torch.zeros((pad_pnum, 2), device=map_score.device)], dim=0)
                padding_mask[valid_pnum:] = True
            selected_map_query.append(valid_map_query)
            selected_map_pos.append(valid_map_pos)
            selected_padding_mask.append(padding_mask)

        selected_map_query = torch.stack(selected_map_query, dim=0)
        selected_map_pos = torch.stack(selected_map_pos, dim=0)
        selected_padding_mask = torch.stack(selected_padding_mask, dim=0)

        # generate different pe for map vectors for each agent
        num_agent = mot_pos.shape[1]
        selected_map_query = selected_map_query.unsqueeze(1).repeat(1, num_agent, 1, 1)  # [B, A, max_P, D]
        selected_map_pos = selected_map_pos.unsqueeze(1).repeat(1, num_agent, 1, 1)  # [B, A, max_P, 2]
        selected_padding_mask = selected_padding_mask.unsqueeze(1).repeat(1, num_agent, 1)  # [B, A, max_P]
        # move lane to per-car coords system
        selected_map_dist = selected_map_pos - mot_pos[:, :, None, :]  # [B, A, max_P, 2]
        if pe_normalization:
            selected_map_pos = selected_map_pos - mot_pos[:, :, None, :]  # [B, A, max_P, 2]

        # filter far map inst for each agent
        map_dis = torch.sqrt(selected_map_dist[..., 0]**2 + selected_map_dist[..., 1]**2)
        valid_map_inst = (map_dis <= dis_thresh)  # [B, A, max_P]
        invalid_map_inst = (valid_map_inst == False)
        selected_padding_mask = selected_padding_mask + invalid_map_inst

        selected_map_query = selected_map_query.flatten(0, 1)
        selected_map_pos = selected_map_pos.flatten(0, 1)
        selected_padding_mask = selected_padding_mask.flatten(0, 1)

        num_batch = selected_padding_mask.shape[0]
        feat_dim = selected_map_query.shape[-1]
        if use_fix_pad:
            pad_map_query = torch.zeros((num_batch, 1, feat_dim), device=selected_map_query.device)
            pad_map_pos = torch.ones((num_batch, 1, 2), device=selected_map_pos.device)
            pad_lane_mask = torch.tensor([False], device=selected_padding_mask.device).unsqueeze(0).repeat(num_batch, 1)
            selected_map_query = torch.cat([selected_map_query, pad_map_query], dim=1)
            selected_map_pos = torch.cat([selected_map_pos, pad_map_pos], dim=1)
            selected_padding_mask = torch.cat([selected_padding_mask, pad_lane_mask], dim=1)

        return selected_map_query, selected_map_pos, selected_padding_mask


    def select_and_pad_query(
        self,
        query,
        query_pos,
        query_score,
        score_thresh=0.5,
        use_fix_pad=True
    ):
        """select_and_pad_query.
        Args:
            query: [B, Q, D].
            query_pos: [B, Q, 2]
            query_score: [B, Q, C].
            score_thresh: confidence threshold for filtering low-confidence query
            use_fix_pad: always pad one query instance for each batch
        Returns:
            selected_query: [B, Q', D]
            selected_query_pos: [B, Q', 2]
            selected_padding_mask: [B, Q']
        """

        # select & pad query for different batch using score_thresh
        query_score = query_score.sigmoid()
        query_score = query_score.max(dim=-1)[0]
        query_idx = query_score > score_thresh
        batch_max_qnum = 0
        for i in range(query_score.shape[0]):
            qnum = query_idx[i].sum()
            if qnum > batch_max_qnum:
                batch_max_qnum = qnum

        selected_query, selected_query_pos, selected_padding_mask = [], [], []
        for i in range(query_score.shape[0]):
            dim = query.shape[-1]
            valid_qnum = query_idx[i].sum()
            valid_query = query[i, query_idx[i]]
            valid_query_pos = query_pos[i, query_idx[i]]
            pad_qnum = batch_max_qnum - valid_qnum
            padding_mask = torch.tensor([False], device=query_score.device).repeat(batch_max_qnum)
            if pad_qnum != 0:
                valid_query = torch.cat([valid_query, torch.zeros((pad_qnum, dim), device=query_score.device)], dim=0)
                valid_query_pos = torch.cat([valid_query_pos, torch.zeros((pad_qnum, 2), device=query_score.device)], dim=0)
                padding_mask[valid_qnum:] = True
            selected_query.append(valid_query)
            selected_query_pos.append(valid_query_pos)
            selected_padding_mask.append(padding_mask)

        selected_query = torch.stack(selected_query, dim=0)
        selected_query_pos = torch.stack(selected_query_pos, dim=0)
        selected_padding_mask = torch.stack(selected_padding_mask, dim=0)

        num_batch = selected_padding_mask.shape[0]
        feat_dim = selected_query.shape[-1]
        if use_fix_pad:
            pad_query = torch.zeros((num_batch, 1, feat_dim), device=selected_query.device)
            pad_query_pos = torch.ones((num_batch, 1, 2), device=selected_query_pos.device)
            pad_mask = torch.tensor([False], device=selected_padding_mask.device).unsqueeze(0).repeat(num_batch, 1)
            selected_query = torch.cat([selected_query, pad_query], dim=1)
            selected_query_pos = torch.cat([selected_query_pos, pad_query_pos], dim=1)
            selected_padding_mask = torch.cat([selected_padding_mask, pad_mask], dim=1)

        return selected_query, selected_query_pos, selected_padding_mask


    def get_plan_col_target(self,
             ego_traj_preds,
             ego_traj_gts,
             agents_boxes_gts,
             agents_feats_gts):
        """"Get Trajectory mode classification target.
        Args:
            ego_traj_preds (Tensor): MultiModal traj preds with shape (B, plan_fut_mode, fut_ts, 2).
            ego_traj_gts (Tensor): traj gts with shape (B, 1, fut_ts, 2).
            agents_boxes_gts (List(Tensor)): Ground truth traj for each agent with shape (N_a, 9).
            agents_feats_gts (List(Tensor)): Ground truth feats for each agent with shape (N_a, 34).
        Returns:
            traj_labels (Tensor): traj cls labels (1, plan_fut_mode).
        """

        planning_metric = PlanningMetric(fut_ts=self.fut_ts)
        segmentation, pedestrian = planning_metric.get_label(agents_boxes_gts, agents_feats_gts)
        occupancy = torch.logical_or(segmentation, pedestrian)

        label_list = []
        for i in range(self.plan_fut_mode):
            label = planning_metric.evaluate_coll(
                ego_traj_preds[None, i].detach(),
                ego_traj_gts[None, i],
                occupancy)
            label_list.append(label)

        return torch.cat(label_list, dim=-1).to(agents_feats_gts.device)

    def get_plan_bd_target(self,
             ego_traj_preds,
             lane_preds,
             lane_score_preds,
             lane_bound_cls_idx=1,
             map_thresh=0.5):
        """"Get Trajectory mode classification target.
        Args:
            ego_traj_preds (Tensor): MultiModal traj preds with shape (mot_fut_mode, fut_ts, 2).
            lane_preds (Tensor): map preds/GT with shape (num_vec, num_pts, 2).
            lane_score_preds (Tensor): map scores with shape (num_vec, 3).
        Returns:
            traj_labels (Tensor): traj cls labels (1, mot_fut_mode).
        """
        # filter lane element according to confidence score and class
        # not_lane_bound_mask = lane_score_preds[..., lane_bound_cls_idx] < map_thresh
        not_lane_bound_mask = (lane_score_preds != lane_bound_cls_idx)

        # denormalize map pts
        lane_bound_preds = lane_preds.clone()
        # lane_bound_preds[..., 0:1] = (lane_bound_preds[..., 0:1] * (self.pc_range[3] -
        #                         self.pc_range[0]) + self.pc_range[0])
        # lane_bound_preds[..., 1:2] = (lane_bound_preds[..., 1:2] * (self.pc_range[4] -
        #                         self.pc_range[1]) + self.pc_range[1])
        # pad not-lane-boundary cls and low confidence preds
        lane_bound_preds[not_lane_bound_mask] = 1e6

        ego_traj_starts = ego_traj_preds[:, :-1, :]
        ego_traj_ends = ego_traj_preds
        padding_zeros = torch.zeros((self.plan_fut_mode, 1, 2), dtype=ego_traj_preds.dtype,
                                    device=ego_traj_preds.device)  # initial position
        ego_traj_starts = torch.cat((padding_zeros, ego_traj_starts), dim=1)
        V, P, _ = lane_bound_preds.size()
        ego_traj_expanded = ego_traj_ends.unsqueeze(2).unsqueeze(3)  # [num_plan_mode, T, 1, 1, 2]
        maps_expanded = lane_bound_preds.unsqueeze(0).unsqueeze(1)  # [1, 1, M, P, 2]

        dist = torch.linalg.norm(ego_traj_expanded - maps_expanded, dim=-1)  # [num_plan_mode, T, M, P]
        dist = dist.min(dim=-1, keepdim=False)[0]
        min_inst_idxs = torch.argmin(dist, dim=-1).tolist()
        mode_idxs = [[i] for i in range(dist.shape[0])]
        ts_idxs = [[i for i in range(dist.shape[1])] for j in range(dist.shape[0])]
        bd_target = lane_bound_preds.unsqueeze(0).unsqueeze(1).repeat(self.plan_fut_mode, self.fut_ts, 1, 1, 1)
        min_bd_insts = bd_target[mode_idxs, ts_idxs, min_inst_idxs]  # [B, T, P, 2]
        bd_inst_starts = min_bd_insts[:, :, :-1, :].flatten(0, 2)
        bd_inst_ends = min_bd_insts[:, :, 1:, :].flatten(0, 2)
        ego_traj_starts = ego_traj_starts.unsqueeze(2).repeat(1, 1, P-1, 1).flatten(0, 2)
        ego_traj_ends = ego_traj_ends.unsqueeze(2).repeat(1, 1, P-1, 1).flatten(0, 2)

        intersect_mask = segments_intersect(ego_traj_starts, ego_traj_ends,
                                            bd_inst_starts, bd_inst_ends)
        left_deviation = ego_traj_starts.new_tensor([-0.9, 2.4])
        right_deviation = ego_traj_starts.new_tensor([+0.9, 2.4])
        forward_deviation = ego_traj_starts.new_tensor([0., 2.4])
        intersect_mask_left = segments_intersect(ego_traj_starts + left_deviation, ego_traj_ends + left_deviation,
                                            bd_inst_starts, bd_inst_ends)
        intersect_mask_right = segments_intersect(ego_traj_starts + right_deviation, ego_traj_ends + right_deviation,
                                            bd_inst_starts, bd_inst_ends)
        intersect_mask_forward = segments_intersect(ego_traj_starts + forward_deviation, ego_traj_ends + forward_deviation,
                                            bd_inst_starts, bd_inst_ends)
        intersect_mask = intersect_mask | intersect_mask_left | intersect_mask_right | intersect_mask_forward
        # self.W = 1.85
        # self.H = 4.084
        intersect_mask = intersect_mask.reshape(self.plan_fut_mode, self.fut_ts, P-1)
        intersect_mask = intersect_mask.any(dim=-1).any(dim=-1)

        bd_overstep_labels = torch.zeros((self.plan_fut_mode), dtype=torch.long,
                                          device=ego_traj_preds.device)
        bd_overstep_labels[intersect_mask] = 1
        bd_overstep_labels[~intersect_mask] = 0

        return bd_overstep_labels


    def get_plan_cl_target(self,
             ego_traj_preds,
             lane_preds,
             lane_score_preds,
             lane_bound_cls_idx=3,
             map_thresh=0.5):


                     # filter lane element according to confidence score and class
        # not_lane_bound_mask = lane_score_preds[..., lane_bound_cls_idx] < map_thresh
        not_lane_bound_mask = (lane_score_preds != lane_bound_cls_idx)

        # denormalize map pts
        lane_centerline_preds = lane_preds.clone()
        # lane_centerline_preds[..., 0:1] = (lane_centerline_preds[..., 0:1] * (self.pc_range[3] -
        #                         self.pc_range[0]) + self.pc_range[0])
        # lane_centerline_preds[..., 1:2] = (lane_centerline_preds[..., 1:2] * (self.pc_range[4] -
        #                         self.pc_range[1]) + self.pc_range[1])
        # pad not-lane-boundary cls and low confidence preds
        lane_centerline_preds[not_lane_bound_mask] = 1e6



        ego_traj_expanded = ego_traj_preds.unsqueeze(2).unsqueeze(3)  # [num_plan_mode, T, 1, 1, 2]
       
        maps_interpolated = F.interpolate(lane_centerline_preds.permute(0, 2, 1), \
                        scale_factor=50, mode='linear', align_corners=True).permute(0, 2, 1)

        maps_expanded = maps_interpolated.unsqueeze(0).unsqueeze(1)  # [1, 1, M, P, 2]
        
        dist = torch.linalg.norm(ego_traj_expanded - maps_expanded, dim=-1)  # [num_plan_mode, T, M, P]

        dist = dist.min(dim=-1)[0] # map point dim
        dist = dist.sum(dim=1)   # dist = dist.max(dim=1)[0]  plan T dim   (max deviation)   or dist = dist.sum(dim=1)  (mean deviation)
        dist, nearest_map_ins_idx = dist.min(dim=-1)  # map ins dim
        maps_matched = maps_interpolated.index_select(dim=0, index=nearest_map_ins_idx)   # [num_plan_mode, P, 2]
        
        dist_2 = torch.linalg.norm(ego_traj_preds.unsqueeze(2) - maps_matched.unsqueeze(1), dim=-1)

        mode_idxs = [[i] for i in range(dist.shape[0])]
        point_idx = dist_2.min(dim=-1)[1]
        
        point_idx[point_idx==0] = 1
        map_segment_starts = maps_matched[mode_idxs, point_idx - 1]
        map_segment_ends = maps_matched[mode_idxs, point_idx]
        centerline_vector = map_segment_ends - map_segment_starts

        ego_traj_starts = ego_traj_preds[:, :-1, :]
        ego_traj_ends = ego_traj_preds
        padding_zeros = torch.zeros((self.plan_fut_mode, 1, 2), dtype=ego_traj_preds.dtype,
                                    device=ego_traj_preds.device)  # initial position
        ego_traj_starts = torch.cat((padding_zeros, ego_traj_starts), dim=1)
        ego_vector = ego_traj_ends - ego_traj_starts

        cos_sim = F.cosine_similarity(ego_vector, centerline_vector, dim=-1)

        
        cl_dir_labels = 1. - dist_2.min(dim=-1)[0].mean(dim=-1)
        # cl_dir_labels = cos_sim.mean(dim=-1) \
        #     - dist_2.min(dim=-1)[0].mean(dim=-1)
        cl_dir_labels = torch.clamp(cl_dir_labels, min=-1.)
    
        # cl_dir_labels = (cl_dir_labels < 0.5).long()

        # TODO   p2l  cost          

        return cl_dir_labels
        

    def get_plan_expert_target(self,
            ego_traj_preds,
            ego_fut_gt,
            ego_fut_masks,
            ego_cls_expert_preds, # (N, 1)
            plan_col_labels,
            plan_bd_labels,
            ):

        plan_expert_labels = torch.zeros((self.plan_fut_mode), dtype=torch.long,
                                          device=ego_traj_preds.device)

        plan_expert_labels_weight = torch.zeros((self.plan_fut_mode), dtype=ego_traj_preds.dtype,
                                          device=ego_traj_preds.device)
        
        if ego_fut_masks[0] == 1.:

            neg_idx = torch.ones((self.plan_fut_mode), dtype=torch.bool,
                                          device=ego_traj_preds.device)
            #### v1
            # traj_dis = torch.linalg.norm(ego_traj_preds[:,:,:] - ego_fut_gt.cumsum(dim=-2)[:,:,:], dim=-1).sum(dim=-1) \
            #     + torch.linalg.norm(ego_traj_preds[:,0,:] - ego_fut_gt.cumsum(dim=-2)[:,0,:], dim=-1) * 5.
            # plan_expert_labels[neg_idx] = 1
            # plan_expert_labels_weight[neg_idx] = torch.clip(traj_dis, min=0, max=2.) / 2.
            #### v2
            # traj_dis = torch.linalg.norm(ego_traj_preds[:,:,:] - ego_fut_gt.cumsum(dim=-2)[:,:,:], dim=-1).sum(dim=-1)
            # plan_expert_labels[neg_idx] = 1
            # plan_expert_labels_weight[neg_idx] = torch.clip(traj_dis, min=0, max=1.5) / 1.5
            #### v3
            # traj_dis = torch.linalg.norm(ego_traj_preds[:,:,:] - ego_fut_gt.cumsum(dim=-2)[:,:,:], dim=-1).sum(dim=-1)
            # plan_expert_labels[neg_idx] = 1
            # plan_expert_labels_weight[neg_idx] = torch.clip(traj_dis, min=0, max=10) / 1.5
            # #### v4
            # traj_dis = torch.linalg.norm(ego_traj_preds[:,:,:] - ego_fut_gt.cumsum(dim=-2)[:,:,:], dim=-1).sum(dim=-1)
            # plan_expert_labels[neg_idx] = 1
            # plan_expert_labels_weight[neg_idx] = torch.clip(traj_dis, min=0, max=15) / 1.5
            # #### v5
            # traj_dis = torch.linalg.norm(ego_traj_preds[:,:,:] - ego_fut_gt.cumsum(dim=-2)[:,:,:], dim=-1).sum(dim=-1)
            # plan_expert_labels[neg_idx] = 1
            # plan_expert_labels_weight[neg_idx] = torch.clip(traj_dis, min=0, max=30) / 1.5
            #### v6
            # traj_dis = torch.linalg.norm(ego_traj_preds[:,:,:] - ego_fut_gt.cumsum(dim=-2)[:,:,:], dim=-1).sum(dim=-1)
            # plan_expert_labels[neg_idx] = 1
            # plan_expert_labels_weight[neg_idx] = torch.clip(traj_dis, min=0, max=30) * 2.
            #### v7
            # traj_dis = torch.linalg.norm(ego_traj_preds[:,:,:] - ego_fut_gt.cumsum(dim=-2)[:,:,:], dim=-1).sum(dim=-1)
            # plan_expert_labels[neg_idx] = 1
            # plan_expert_labels_weight[neg_idx] = torch.clip(traj_dis, min=0, max=100.) * 2.
            #### v8
            # traj_dis = torch.linalg.norm(ego_traj_preds[:,:,:] - ego_fut_gt.cumsum(dim=-2)[:,:,:], dim=-1).sum(dim=-1)
            # plan_expert_labels[neg_idx] = 1
            # plan_expert_labels_weight[neg_idx] = torch.clip(traj_dis, min=0, max=100.) * 4.
            #### v9
            # traj_dis = torch.linalg.norm(ego_traj_preds[:,:,:] - ego_fut_gt.cumsum(dim=-2)[:,:,:], dim=-1).sum(dim=-1)
            # plan_expert_labels[neg_idx] = 1
            # plan_expert_labels_weight[neg_idx] = torch.clip(traj_dis, min=0, max=100.) * 10.
            # #### v10
            # traj_dis = torch.linalg.norm(ego_traj_preds[:,:,:] - ego_fut_gt.cumsum(dim=-2)[:,:,:], dim=-1).sum(dim=-1)
            # plan_expert_labels[neg_idx] = 1
            # plan_expert_labels_weight[neg_idx] = torch.clip(traj_dis, min=0, max=100.) * 20.
            #### v11
            traj_dis = torch.linalg.norm(ego_traj_preds[:,:,:] - ego_fut_gt.cumsum(dim=-2)[:,:,:], dim=-1).sum(dim=-1)
            plan_expert_labels[neg_idx] = 1
            plan_expert_labels_weight[neg_idx] = torch.clip(traj_dis, min=0, max=100.) * 100.

            plan_expert_labels[plan_col_labels == 1] = 1
            plan_expert_labels[plan_bd_labels == 1] = 1
            plan_expert_labels_weight[plan_col_labels == 1] = 100.
            plan_expert_labels_weight[plan_bd_labels == 1] = 100.

            # pos_idx = torch.linalg.norm(ego_traj_preds[:,:1,:] - ego_fut_gt.cumsum(dim=-2)[:,:1,:], dim=-1).mean(dim=-1).argmin()
            pos_idx = traj_dis.argmin()
            plan_expert_labels[pos_idx] = 0

            # add weights to balance trajs
            self.traj_selected_cnt[pos_idx] += 1.
            scaling_rate = self.traj_selected_cnt.sum() / self.traj_selected_cnt[pos_idx] / self.plan_fut_mode 
            scaling_rate = torch.clamp(scaling_rate, 0.5, 2.)
            plan_expert_labels_weight[pos_idx] = 100.  # * scaling_rate   

            # global pos_idx_cnt
            # pos_idx_cnt[pos_idx] += 1
            #-------
            # pos_idx = torch.linalg.norm(ego_traj_preds[:,:,:] - ego_fut_gt.cumsum(dim=-2)[:,:,:], dim=-1).mean(dim=-1).argmin()
            # rank = (ego_cls_expert_preds[pos_idx] < ego_cls_expert_preds).sum()
            # plan_expert_labels[pos_idx] = 0
            # plan_expert_labels_weight[pos_idx] = 500. * min(rank, 10)


            # neg_idx = torch.linalg.norm(ego_traj_preds[:,:1,:] - ego_fut_gt.cumsum(dim=-2)[:,:1,:], dim=-1).mean(dim=-1) >  -10e6 # all
            # plan_expert_labels[neg_idx] = 1
            # plan_expert_labels_weight[neg_idx] = min(rank, 10) / self.plan_fut_mode


            # plan_expert_labels[plan_col_labels == 1] = 1
            # plan_expert_labels[plan_bd_labels == 1] = 1
            # plan_expert_labels_weight[plan_col_labels == 1] = 1.
            # plan_expert_labels_weight[plan_bd_labels == 1] = 1.

        return plan_expert_labels, plan_expert_labels_weight
    
class PlanningMetric():
    def __init__(self, fut_ts=6):
        super().__init__()
        self.X_BOUND = [-50.0, 50.0, 0.5]  # Forward
        self.Y_BOUND = [-50.0, 50.0, 0.5]  # Sides
        self.Z_BOUND = [-10.0, 10.0, 20.0]  # Height
        self.fut_ts = fut_ts
        dx, bx, _ = self.gen_dx_bx(self.X_BOUND, self.Y_BOUND, self.Z_BOUND)
        self.dx, self.bx = dx[:2], bx[:2]

        bev_resolution, bev_start_position, bev_dimension = self.calculate_birds_eye_view_parameters(
            self.X_BOUND, self.Y_BOUND, self.Z_BOUND
        )
        self.bev_resolution = bev_resolution.numpy()
        self.bev_start_position = bev_start_position.numpy()
        self.bev_dimension = bev_dimension.numpy()

        self.W = 1.85
        self.H = 4.084

        self.category_index = {
            'human':[0,1,2,3],
            'vehicle':[0,1,2,3]
        }

    def gen_dx_bx(self, xbound, ybound, zbound):
        dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
        bx = torch.Tensor([row[0] + row[2]/2.0 for row in [xbound, ybound, zbound]])
        nx = torch.LongTensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]])

        return dx, bx, nx
    
    def calculate_birds_eye_view_parameters(self, x_bounds, y_bounds, z_bounds):
        """
        Parameters
        ----------
            x_bounds: Forward direction in the ego-car.
            y_bounds: Sides
            z_bounds: Height

        Returns
        -------
            bev_resolution: Bird's-eye view bev_resolution
            bev_start_position Bird's-eye view first element
            bev_dimension Bird's-eye view tensor spatial dimension
        """
        bev_resolution = torch.tensor([row[2] for row in [x_bounds, y_bounds, z_bounds]])
        bev_start_position = torch.tensor([row[0] + row[2] / 2.0 for row in [x_bounds, y_bounds, z_bounds]])
        bev_dimension = torch.tensor([(row[1] - row[0]) / row[2] for row in [x_bounds, y_bounds, z_bounds]],
                                    dtype=torch.long)

        return bev_resolution, bev_start_position, bev_dimension
    
    def get_label(
            self,
            gt_agent_boxes,
            gt_agent_feats
        ):
        segmentation_np, pedestrian_np = self.get_birds_eye_view_label(gt_agent_boxes,gt_agent_feats)
        segmentation = torch.from_numpy(segmentation_np).long().unsqueeze(0)
        pedestrian = torch.from_numpy(pedestrian_np).long().unsqueeze(0)

        return segmentation, pedestrian
    
    def get_birds_eye_view_label(
            self,
            gt_agent_boxes,
            gt_agent_feats
        ):
        '''
        gt_agent_boxes (LiDARInstance3DBoxes): list of GT Bboxs.
            dim 9 = (x,y,z)+(w,l,h)+yaw+(vx,vy)
        gt_agent_feats: (A, 4*T+10)
            dim 4*T+10 = fut_traj(T*2) + fut_mask(T) + goal(1) + lcf_feat(9) + fut_yaw(T)
            lcf_feat (x, y, yaw, vx, vy, width, length, height, type)
        '''

        segmentation = np.zeros((self.fut_ts, self.bev_dimension[0], self.bev_dimension[1]))
        pedestrian = np.zeros((self.fut_ts, self.bev_dimension[0], self.bev_dimension[1]))
        agent_num = gt_agent_feats.shape[0]

        gt_agent_boxes = gt_agent_boxes.cpu().numpy()  #(N, 9)
        gt_agent_feats = gt_agent_feats.cpu().numpy()

        gt_agent_fut_trajs = gt_agent_feats[..., :self.fut_ts*2].reshape(-1, self.fut_ts, 2)
        gt_agent_fut_mask = gt_agent_feats[..., self.fut_ts*2:self.fut_ts*3].reshape(-1, self.fut_ts)
        # gt_agent_lcf_feat = gt_agent_feats[..., T*3+1:T*3+10].reshape(-1, 9)
        gt_agent_fut_yaw = gt_agent_feats[..., self.fut_ts*3+10:self.fut_ts*4+10].reshape(-1, self.fut_ts, 1)
        gt_agent_fut_trajs = np.cumsum(gt_agent_fut_trajs, axis=1)
        gt_agent_fut_yaw = np.cumsum(gt_agent_fut_yaw, axis=1)

        gt_agent_boxes[:,6:7] = -1*(gt_agent_boxes[:, 6:7] + np.pi/2) # NOTE: convert yaw to lidar frame
        gt_agent_fut_trajs = gt_agent_fut_trajs + gt_agent_boxes[:, np.newaxis, 0:2]
        gt_agent_fut_yaw = gt_agent_fut_yaw + gt_agent_boxes[:, np.newaxis, 6:7]
        
        for t in range(self.fut_ts):
            for i in range(agent_num):
                if gt_agent_fut_mask[i][t] == 1:
                    # Filter out all non vehicle instances
                    category_index = int(gt_agent_feats[i, 3*self.fut_ts+9])
                    agent_length, agent_width = gt_agent_boxes[i][4], gt_agent_boxes[i][3]
                    x_a = gt_agent_fut_trajs[i, t, 0]
                    y_a = gt_agent_fut_trajs[i, t, 1]
                    yaw_a = gt_agent_fut_yaw[i, t, 0]
                    param = [x_a, y_a, yaw_a, agent_length, agent_width]
                    if (category_index in self.category_index['vehicle']):
                        poly_region = self._get_poly_region_in_image(param)
                        cv2.fillPoly(segmentation[t], [poly_region], 1.0)
                    if (category_index in self.category_index['human']):
                        poly_region = self._get_poly_region_in_image(param)
                        cv2.fillPoly(pedestrian[t], [poly_region], 1.0)

        return segmentation, pedestrian
    
    def _get_poly_region_in_image(self,param):
        lidar2cv_rot = np.array([[1,0], [0,-1]])
        x_a, y_a, yaw_a, agent_length, agent_width = param
        trans_a = np.array([[x_a, y_a]]).T
        rot_mat_a = np.array([[np.cos(yaw_a), -np.sin(yaw_a)],
                                [np.sin(yaw_a), np.cos(yaw_a)]])
        agent_corner = np.array([
            [agent_length/2, -agent_length/2, -agent_length/2, agent_length/2],
            [agent_width/2, agent_width/2, -agent_width/2, -agent_width/2]]) #(2,4)
        agent_corner_lidar = np.matmul(rot_mat_a, agent_corner) + trans_a #(2,4)
        # convert to cv frame
        agent_corner_cv2 = (np.matmul(lidar2cv_rot, agent_corner_lidar) \
            - self.bev_start_position[:2, None] + self.bev_resolution[:2, None] / 2.0).T / self.bev_resolution[:2] #(4,2)
        agent_corner_cv2 = np.round(agent_corner_cv2).astype(np.int32)

        return agent_corner_cv2

    def evaluate_single_coll(self, traj, segmentation, input_gt):
        '''
        traj: torch.Tensor (n_future, 2)
            自车lidar系为轨迹参考系
                ^ y
                |
                | 
                0------->
                        x
        segmentation: torch.Tensor (n_future, 200, 200)
        '''
        pts = np.array([
            [-self.H / 2. + 0.5, self.W / 2.],
            [self.H / 2. + 0.5, self.W / 2.],
            [self.H / 2. + 0.5, -self.W / 2.],
            [-self.H / 2. + 0.5, -self.W / 2.],
        ])
        pts = (pts - self.bx.cpu().numpy()) / (self.dx.cpu().numpy())
        pts[:, [0, 1]] = pts[:, [1, 0]]
        rr, cc = polygon(pts[:,1], pts[:,0])
        rc = np.concatenate([rr[:,None], cc[:,None]], axis=-1)

        n_future, _ = traj.shape
        trajs = traj.view(n_future, 1, 2)
        # 轨迹坐标系转换为:
        #  ^ x
        #  |
        #  | 
        #  0-------> y
        trajs_ = copy.deepcopy(trajs)
        trajs_[:,:,[0,1]] = trajs_[:,:,[1,0]] # can also change original tensor
        trajs_ = trajs_ / self.dx.to(trajs.device)
        trajs_ = trajs_.cpu().numpy() + rc # (n_future, 32, 2)

        r = (self.bev_dimension[0] - trajs_[:,:,0]).astype(np.int32)
        r = np.clip(r, 0, self.bev_dimension[0] - 1)

        c = trajs_[:,:,1].astype(np.int32)
        c = np.clip(c, 0, self.bev_dimension[1] - 1)

        collision = np.full(n_future, False)
        for t in range(n_future):
            rr = r[t]
            cc = c[t]
            I = np.logical_and(
                np.logical_and(rr >= 0, rr < self.bev_dimension[0]),
                np.logical_and(cc >= 0, cc < self.bev_dimension[1]),
            )
            collision[t] = np.any(segmentation[t, rr[I], cc[I]].cpu().numpy())

        return torch.from_numpy(collision).to(device=traj.device)


    def evaluate_coll(
            self, 
            trajs, 
            gt_trajs, 
            segmentation
        ):
        '''
        trajs: torch.Tensor (B, n_future, 2)
            自车lidar系为轨迹参考系
            ^ y
            |
            | 
            0------->
                    x
        gt_trajs: torch.Tensor (B, n_future, 2)
        segmentation: torch.Tensor (B, n_future, 200, 200)

        '''
        B, n_future, _ = trajs.shape
        # trajs = trajs * torch.tensor([-1, 1], device=trajs.device)
        # gt_trajs = gt_trajs * torch.tensor([-1, 1], device=gt_trajs.device)

        obj_box_coll_sum = torch.zeros(n_future, device=segmentation.device)

        for i in range(B):
            gt_box_coll = self.evaluate_single_coll(gt_trajs[i], segmentation[i], input_gt=True)

            xx, yy = trajs[i,:,0], trajs[i, :, 1]
            # lidar系下的轨迹转换到图片坐标系下
            xi = ((-self.bx[0]/2 - yy) / self.dx[0]).long()
            yi = ((-self.bx[1]/2 + xx) / self.dx[1]).long()

            m1 = torch.logical_and(
                torch.logical_and(xi >= 0, xi < self.bev_dimension[0]),
                torch.logical_and(yi >= 0, yi < self.bev_dimension[1]),
            ).to(gt_box_coll.device)
            m1 = torch.logical_and(m1, torch.logical_not(gt_box_coll))

            ti = torch.arange(n_future)
            # m2 = torch.logical_not(gt_box_coll)
            m2 = torch.ones_like(gt_box_coll)
            box_coll = self.evaluate_single_coll(trajs[i], segmentation[i], input_gt=False).to(ti.device)
            obj_box_coll_sum[ti[m2]] += (box_coll[ti[m2]]).long()

            if obj_box_coll_sum.max() > 0:
                return torch.ones((1), dtype=torch.long)
            else:
                return torch.zeros((1), dtype=torch.long)
