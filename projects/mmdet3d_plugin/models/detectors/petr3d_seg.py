# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
import torch
import mmcv
import numpy as np
from mmcv.parallel import DataContainer as DC
from os import path as osp
import copy
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmdet3d.core import (CameraInstance3DBoxes,LiDARInstance3DBoxes, bbox3d2result,
                          show_multi_modality_result)
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
import cv2
from einops import rearrange
def IOU (intputs,targets):
    numerator = 2 * (intputs * targets).sum(dim=1)
    denominator = intputs.sum(dim=1) + targets.sum(dim=1)
    loss = (numerator + 0.01) / (denominator + 0.01)
    return loss

@DETECTORS.register_module()
class Petr3D_seg(MVXTwoStageDetector):
    """Detr3D."""

    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(Petr3D_seg, self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)
        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask

    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        # print(img[0].size())
        if isinstance(img, list):
            img = torch.stack(img, dim=0)

        B = img.size(0)
        if img is not None:
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            for img_meta in img_metas:
                img_meta.update(input_shape=input_shape)
            if img.dim() == 5:
                if img.size(0) == 1 and img.size(1) != 1:
                    img.squeeze_()
                else:
                    B, N, C, H, W = img.size()
                    img = img.view(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)
            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    @auto_fp16(apply_to=('img'), out_fp32=True)
    def extract_feat(self, img, img_metas):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img, img_metas)
        return img_feats

    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          maps,
                          img_metas,
                          gt_bboxes_ignore=None):
        """Forward function for point cloud branch.
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
        Returns:
            dict: Losses of each branch.
        """
        outs = self.pts_bbox_head(pts_feats, img_metas)
        
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs,maps]
        losses = self.pts_bbox_head.loss(*loss_inputs)

        return losses

    @force_fp32(apply_to=('img', 'points'))
    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      maps=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      img_depth=None,
                      img_mask=None):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """

        img_feats = self.extract_feat(img=img, img_metas=img_metas)

        losses = dict()
        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                            gt_labels_3d, maps,img_metas,
                                            gt_bboxes_ignore)
        losses.update(losses_pts)
        return losses
    
    
    def img_show(self, imgs):
        import os
        import cv2
        import random
        import numpy as np
        mean= np.array([103.530, 116.280, 123.675])
        mean = mean.reshape(1,1,3)
        if not os.path.exists("./imgs"):
            os.makedirs("./imgs")
        name = str(random.randint(1,20))
        for i in range(imgs.size(1)):
            img = imgs[0][i]
            img = img.permute(1, 2, 0).detach().cpu().numpy()
            img = img + mean
            # print(img)

            cv2.imwrite("./imgs/"+name+"_"+str(i)+".png", img.astype(np.uint8))
            print(img.shape)

    
    def forward_test(self, img_metas,gt_map, maps,img=None, **kwargs):
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        img = [img] if img is None else img
        return self.simple_test(img_metas[0], gt_map,img[0],maps, **kwargs)

    def simple_test_pts(self, x, img_metas, gt_map,maps,rescale=False):
        """Test function of point cloud branch."""
        outs = self.pts_bbox_head(x, img_metas)
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]

        with torch.no_grad():
            
            lane_preds=outs['all_lane_preds'][5].squeeze(0)    #[B,N,H,W]
            # lane_pred_obj=outs['all_lane_cls'][5].squeeze(0)     #[B,N,2]
            n,w=lane_preds.size()
            # lane_preds=maps[0][0]
            
            pred_maps=lane_preds.view(256,3,16,16)


            f_lane=rearrange(pred_maps, '(h w) c h1 w2 -> c (h h1) (w w2)', h=16, w=16)
            f_lane=f_lane.sigmoid()
            f_lane[f_lane>=0.5]=1
            f_lane[f_lane<0.5]=0
            f_lane_show=copy.deepcopy(f_lane)
            gt_map_show=copy.deepcopy(gt_map[0])
            
            f_lane=f_lane.view(3,-1)
            gt_map=gt_map[0].view(3,-1) 
            
            ret_iou=IOU(f_lane,gt_map).cpu()
            show_res=False
            if show_res:
            # select good quality results
            # if ret_iou[0]>0.79 and ret_iou[1]>0.45 and ret_iou[2]>0.51:

                pres=f_lane_show
                pre=torch.zeros(256,256,3)
                pre+=255
                label=[[71,130,255],[255,255,0],[255,144,30]]
                # label=[[255,0,0],[0,255,0],[0,0,255]]
                pre[...,0][pres[0]==1]=label[0][0]
                pre[...,1][pres[0]==1]=label[0][1]
                pre[...,2][pres[0]==1]=label[0][2]
                pre[...,0][pres[2]==1]=label[2][0]
                pre[...,1][pres[2]==1]=label[2][1]
                pre[...,2][pres[2]==1]=label[2][2]
                pre[...,0][pres[1]==1]=label[1][0]
                pre[...,1][pres[1]==1]=label[1][1]
                pre[...,2][pres[1]==1]=label[1][2]
                cv2.imwrite('./res-pre/'+str(ret_iou[0])+'_'+str(ret_iou[1])+'_'+str(ret_iou[2])+'_'+img_metas[0]['sample_idx']+'.png',pre.cpu().numpy())
                pres=gt_map_show[0]
                pre=torch.zeros(256,256,3)
                pre+=255
                label=[[71,130,255],[255,255,0],[255,144,30]]
                # label=[[255,0,0],[0,255,0],[0,0,255]]
                pre[...,0][pres[0]==1]=label[0][0]
                pre[...,1][pres[0]==1]=label[0][1]
                pre[...,2][pres[0]==1]=label[0][2]
                pre[...,0][pres[2]==1]=label[2][0]
                pre[...,1][pres[2]==1]=label[2][1]
                pre[...,2][pres[2]==1]=label[2][2]
                pre[...,0][pres[1]==1]=label[1][0]
                pre[...,1][pres[1]==1]=label[1][1]
                pre[...,2][pres[1]==1]=label[1][2]
                cv2.imwrite('./res-gt/'+str(ret_iou[0])+'_'+str(ret_iou[1])+'_'+str(ret_iou[2])+'_'+img_metas[0]['sample_idx']+'.png',pre.cpu().numpy())
               
        return bbox_results, ret_iou


    
    def simple_test(self, img_metas,gt_map=None, img=None,maps=None, rescale=False):
        """Test function without augmentaiton."""
        img_feats = self.extract_feat(img=img, img_metas=img_metas)

        bbox_list = [dict() for i in range(len(img_metas))]
        bbox_pts,ret_iou = self.simple_test_pts(
            img_feats, img_metas, gt_map,maps,rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
            result_dict['ret_iou']=ret_iou
        return bbox_list

    def aug_test_pts(self, feats, img_metas, rescale=False):
        feats_list = []
        for j in range(len(feats[0])):
            feats_list_level = []
            for i in range(len(feats)):
                feats_list_level.append(feats[i][j])
            feats_list.append(torch.stack(feats_list_level, -1).mean(-1))
        outs = self.pts_bbox_head(feats_list, img_metas)
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def aug_test(self, img_metas, imgs=None, rescale=False):
        """Test function with augmentaiton."""
        img_feats = self.extract_feats(img_metas, imgs)
        img_metas = img_metas[0]
        bbox_list = [dict() for i in range(len(img_metas))]
        bbox_pts = self.aug_test_pts(img_feats, img_metas, rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list
    
    
    def show_results(self, data, result, out_dir, score_thr=0.1):
        """Results visualization.

        Args:
            data (list[dict]): Input images and the information of the sample.
            result (list[dict]): Prediction results.
            out_dir (str): Output directory of visualization result.
        """

        for batch_id in range(len(result)):
            if isinstance(data['img_metas'][0], DC):
                img_filename = data['img_metas'][0]._data[0][batch_id][
                    'filename']
                cam2img = data['img_metas'][0]._data[0][batch_id]['lidar2img']
            elif mmcv.is_list_of(data['img_metas'][0], dict):
                img_filename = data['img_metas'][0][batch_id]['filename']
                cam2img = data['img_metas'][0][batch_id]['lidar2img']
            else:
                ValueError(
                    f"Unsupported data type {type(data['img_metas'][0])} "
                    f'for visualization!')

            for i in range(len(img_filename)):
                if "once" in img_filename[i]:
                    img_path = img_filename[i].replace("/home/sunjianjian/workspace/temp/once_benchmark/data/","/data/Dataset/")
                    img = mmcv.imread(img_path)
                    
                    file_name =  img_path.split("/")[-2] + osp.split(img_path)[-1].split('.')[0]
                else:
                    img_path = img_filename[i]
                    img = mmcv.imread(img_path)
                    file_name =  osp.split(img_path)[-1].split('.')[0]
                print(file_name)
                assert out_dir is not None, 'Expect out_dir, got none.'

                pred_bboxes = result[batch_id]['pts_bbox']['boxes_3d']
                pred_scores = result[batch_id]['pts_bbox']['scores_3d']
                pred_labels = result[batch_id]['pts_bbox']['labels_3d']

                mask = pred_scores> score_thr

                pred_bboxes = pred_bboxes[mask]
                pred_scores = pred_scores[mask]
                pred_labels = pred_labels[mask]

                assert isinstance(pred_bboxes, LiDARInstance3DBoxes), \
                    f'unsupported predicted bbox type {type(pred_bboxes)}'
                

                show_multi_modality_result(
                    img,
                    None,
                    pred_bboxes,
                    cam2img[i],
                    out_dir,
                    file_name,
                    'lidar',
                    show=False)

