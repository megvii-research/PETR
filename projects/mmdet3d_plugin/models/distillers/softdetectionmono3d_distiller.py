import torch.nn as nn
import torch.nn.functional as F
import torch
from mmcv.runner import  load_checkpoint
from mmdet.models.builder import build_loss
from ..builder import DISTILLER, build_distill_loss
from mmdet3d.models import build_model
from mmdet.models.detectors import BaseDetector
from mmcv.runner import force_fp32, auto_fp16
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
import os
import numpy as np
import cv2
import math

@DISTILLER.register_module()
class SoftDetectionMonoDistiller(BaseDetector):
    """Base distiller for detectors.
    It typically consists of teacher_model and student_model.
    """
    def __init__(self,
                 teacher_cfg,
                 student_cfg,
                 distill_cfg=None,
                 student_pretrained=None,
                 teacher_pretrained=None,
                 use_grid_mask=False,):

        super(SoftDetectionMonoDistiller, self).__init__()
        self.use_grid_mask = use_grid_mask
        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.teacher = build_model(teacher_cfg.model,
                                        train_cfg=teacher_cfg.get('train_cfg'),
                                        test_cfg=teacher_cfg.get('test_cfg'))
        
        self.init_weights_teacher(teacher_pretrained)
        self.teacher.eval()
        self.student= build_model(student_cfg.model,
                                        train_cfg=student_cfg.get('train_cfg'),
                                        test_cfg=student_cfg.get('test_cfg'))
        if student_pretrained:
            checkpoint = load_checkpoint(self.student, student_pretrained, map_location='cpu')

        self.distill_losses = nn.ModuleDict()

        self.distill_cfg = distill_cfg
        student_modules = dict(self.student.named_modules())
        teacher_modules = dict(self.teacher.named_modules())

        def regitster_hooks(student_module,teacher_module):
            
            def hook_teacher_forward(module, input, output):
                    self.register_buffer(teacher_module,output)
                    
            def hook_student_forward(module, input, output):
                self.register_buffer(student_module,output)

            return hook_teacher_forward,hook_student_forward
        
        for item_loc in distill_cfg:
            
            student_module = 'student_' + item_loc.student_module.replace('.','_')
            teacher_module = 'teacher_' + item_loc.teacher_module.replace('.','_')

            self.register_buffer(student_module,None)
            self.register_buffer(teacher_module,None)

            hook_teacher_forward,hook_student_forward = regitster_hooks(student_module ,teacher_module )
            teacher_modules[item_loc.teacher_module].register_forward_hook(hook_teacher_forward)
            student_modules[item_loc.student_module].register_forward_hook(hook_student_forward)

            for item_loss in item_loc.methods:
                loss_name = item_loss.name
                self.distill_losses[loss_name] = build_distill_loss(item_loss)

    def base_parameters(self):
        return nn.ModuleList([self.student,self.distill_losses])
    def discriminator_parameters(self):
        return self.discriminator

    @property
    def with_neck(self):
        """bool: whether the detector has a neck"""
        return hasattr(self.student, 'neck') and self.student.neck is not None

    # TODO: these properties need to be carefully handled
    # for both single stage & two stage detectors
    @property
    def with_shared_head(self):
        """bool: whether the detector has a shared head in the RoI Head"""
        return hasattr(self.student, 'roi_head') and self.student.roi_head.with_shared_head

    @property
    def with_bbox(self):
        """bool: whether the detector has a bbox head"""
        return ((hasattr(self.student, 'roi_head') and self.student.roi_head.with_bbox)
                or (hasattr(self.student, 'bbox_head') and self.student.bbox_head is not None))

    @property
    def with_mask(self):
        """bool: whether the detector has a mask head"""
        return ((hasattr(self.student, 'roi_head') and self.student.roi_head.with_mask)
                or (hasattr(self.student, 'mask_head') and self.student.mask_head is not None))

    def init_weights_teacher(self, path=None):
        """Load the pretrained model in teacher detector.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if path:
            checkpoint = load_checkpoint(self.teacher, path, map_location='cpu')
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
        # print(self.teacher.img_neck.lateral_convs[0].conv.weight)

    @auto_fp16(apply_to=('img', ))
    def forward(self, img, img_metas, return_loss=True, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if torch.onnx.is_in_onnx_export():
            assert len(img_metas) == 1
            return self.onnx_export(img[0], img_metas[0])

        if return_loss:
            return self.forward_train(img, img_metas, **kwargs)
        else:
            return self.student.forward_test(img, img_metas, **kwargs)

            # if img_metas[0][0]['img_norm_cfg']['to_rgb']:
            #     bgr_img = [torch.flip(img[0].clone(), dims=[1])]
            # else:
            #     mean = torch.from_numpy(img_metas[0][0]['img_norm_cfg']['mean']).view(1, -1, 1, 1).to(img[0].device)
            #     std = torch.from_numpy(img_metas[0][0]['img_norm_cfg']['std']).view(1, -1, 1, 1).to(img[0].device)
            #     bgr_img = img[0].clone()*std + mean
            #     mean = torch.from_numpy(np.array([103.530, 116.280, 123.675])).view(1, -1, 1, 1).to(img[0].device)
            #     std = torch.from_numpy(np.array([57.375, 57.12, 58.395])).view(1, -1, 1, 1).to(img[0].device)
            #     bgr_img = (bgr_img - mean) / std 
            #     bgr_img = [bgr_img.type_as(img[0])]
            # return self.teacher.forward_test(bgr_img, img_metas, **kwargs)


    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      centers2d,
                      depths,
                      attr_labels=None,
                      gt_bboxes_ignore=None):

        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
        Returns:
            dict[str, Tensor]: A dictionary of loss components(student's losses and distiller's losses).
        """
        
        # # rgb_img, bgr_img = torch.chunk(img,2,dim=1)
        # rgb_img = img.clone()
        # if img_metas[0]['img_norm_cfg']['to_rgb']:
        #     bgr_img = torch.flip(img.clone(), dims=[1])
        # else:
        #     mean = torch.from_numpy(img_metas[0]['img_norm_cfg']['mean']).view(1, -1, 1, 1).to(img.device)
        #     std = torch.from_numpy(img_metas[0]['img_norm_cfg']['std']).view(1, -1, 1, 1).to(img.device)
        #     bgr_img = img.clone()*std + mean
        #     mean = torch.from_numpy(np.array([103.530, 116.280, 123.675])).view(1, -1, 1, 1).to(img.device)
        #     std = torch.from_numpy(np.array([57.375, 57.12, 58.395])).view(1, -1, 1, 1).to(img.device)
        #     bgr_img = (bgr_img - mean) / std 
        #     bgr_img = bgr_img.type_as(img)

        bgr_img = img.clone()
        if self.use_grid_mask:
            rgb_img = self.grid_mask(img.clone())
            # print(rgb_img.size())
        else:
            rgb_img = img.clone()
            
        mask = torch.ne(bgr_img, rgb_img)

        # print("rgb",img[0,:,0, 0])
        # print("bgr",bgr_img[0,:,0, 0])

        with torch.no_grad():
            self.teacher.eval()
            # loss = self.teacher.extract_feat(img=bgr_img)
            feats = self.teacher.extract_feat(img=bgr_img)
            # outs = self.teacher.bbox_head(feats)

            # depth = [bbox_pred[:, 3] for bbox_pred in outs[1]]
            # print(depth.size())
           
        student_loss = self.student.forward_train(rgb_img,
                                                img_metas,
                                                gt_bboxes,
                                                gt_labels,
                                                gt_bboxes_3d,
                                                gt_labels_3d,
                                                centers2d,
                                                depths,
                                                attr_labels,
                                                gt_bboxes_ignore)
        
        buffer_dict = dict(self.named_buffers())
        for item_loc in self.distill_cfg:
            
            student_module = 'student_' + item_loc.student_module.replace('.','_')
            teacher_module = 'teacher_' + item_loc.teacher_module.replace('.','_')
            
            student_feat = buffer_dict[student_module]
            teacher_feat = buffer_dict[teacher_module]

            attention_mask = F.interpolate(mask.float(), size=student_feat.shape[-2:], mode='nearest')
            attention_mask, _ = torch.max(attention_mask, 1, keepdim=True)
            # student_feat = student_feat * attention_mask
            # teacher_feat = teacher_feat * attention_mask

            # self.imshow(student_feat, teacher_feat)
            # self.imshow(attention_mask, attention_mask)

            for item_loss in item_loc.methods:
                loss_name = item_loss.name
                student_loss[loss_name] = self.distill_losses[loss_name](student_feat,teacher_feat)
                # student_loss[loss_name] = self.distill_losses[loss_name](student_feat,teacher_feat,attention_mask)
          
        return student_loss
    
    def simple_test(self, img, img_metas, **kwargs):
        return self.student.simple_test(img, img_metas, **kwargs)
    def aug_test(self, img, img_metas, **kwargs):
        return self.student.aug_test(img, img_metas, **kwargs)
    def extract_feat(self, img):
        """Extract features from images."""
        return self.student.extract_feat(img)

    def imshow(self, preds_S, preds_T):
        teacher_tau = 0.25
        student_tau = 0.25
        if preds_S.dim() == 3:
            preds_S, preds_T = preds_S.unsqueeze(1), preds_T.unsqueeze(1)

        N, C, H, W = preds_S.shape
        scale_factor = math.sqrt(W*H)
        shape = preds_S.shape
        softmax_pred_T = F.softmax(preds_T.clone().view(-1,W*H)/teacher_tau, dim=1) #*scale_factor
        softmax_pred_S = F.softmax(preds_S.clone().view(-1,W*H)/student_tau, dim=1) #*scale_factor

        S = softmax_pred_S.view(shape).detach().cpu().numpy()
        T = softmax_pred_T.view(shape).detach().cpu().numpy()

        # S = S[0, 0]
        # T = T[0, 0]

        S = np.max(S[0], axis=0)*255
        T = np.max(T[0], axis=0)*255
        print(S.max(), T.max())

        filename = np.random.randint(0,100)
        path = "./attention/"
        if not os.path.exists(path):
            os.makedirs(path)
        cv2.imwrite(path+str(filename)+"_s.png", S.astype(np.uint8))
        cv2.imwrite(path+str(filename)+"_t.png", T.astype(np.uint8))
        
        



