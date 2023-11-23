import numpy as np

import torch

from mmdet.core import bbox2result, bbox_mapping_back, multiclass_nms, merge_aug_masks, bbox_flip
from ..builder import DETECTORS
from .single_stage import SingleStageDetector
import torch.nn.functional as F
from mmdet.models.roi_heads.mask_heads.condconv_mask_head import aligned_bilinear
import copy
from PIL import Image 
@DETECTORS.register_module()
class RepPointsV2MaskDetector(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 mask_inbox=False):
        self.mask_inbox = mask_inbox
        super(RepPointsV2MaskDetector, self).__init__(backbone, neck, bbox_head, train_cfg,
                                                test_cfg, pretrained)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_sem_map=None,
                      gt_sem_weights=None,
                      gt_masks=None):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | list[BitmapMasks]) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        masks = []
        for mask in gt_masks:
            mask_tensor = img.new_tensor(mask.masks)
            mask_tensor = F.pad(mask_tensor, pad=(0, img.size(-1)-mask_tensor.size(-1), 0, img.size(-2)-mask_tensor.size(-2)))
            masks.append(mask_tensor)
        x = self.extract_feat(img)
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore, gt_sem_map, gt_sem_weights, masks)
        return losses
        
    def merge_aug_results(self, aug_bboxes, aug_scores, img_metas):
        """Merge augmented detection bboxes and scores.

        Args:
            aug_bboxes (list[Tensor]): shape (n, 4*#class)
            aug_scores (list[Tensor] or None): shape (n, #class)
            img_shapes (list[Tensor]): shape (3, ).

        Returns:
            tuple: (bboxes, scores)
        """
        recovered_bboxes = []
        for bboxes, img_info in zip(aug_bboxes, img_metas):
            img_shape = img_info[0]['img_shape']
            scale_factor = img_info[0]['scale_factor']
            flip = img_info[0]['flip']
            flip_direction = img_info[0]['flip_direction']
            bboxes = bbox_mapping_back(bboxes, img_shape, scale_factor, flip,
                                       flip_direction)
            recovered_bboxes.append(bboxes)
        bboxes = torch.cat(recovered_bboxes, dim=0)
            
        if aug_scores is None:
            return bboxes
        else:
            scores = torch.cat(aug_scores, dim=0)
            return bboxes, scores

    def mask2result(self, x, det_labels, inst_inds, img_meta, det_bboxes, pred_instances=None, rescale=True, return_score=False):
        resized_im_h, resized_im_w = img_meta['img_shape'][:2]
        ori_h, ori_w = img_meta['ori_shape'][:2]
        if pred_instances is not None:
            pred_instances = pred_instances[inst_inds]
        else:
            pred_instances = self.bbox_head.pred_instances[inst_inds]

        scale_factor = img_meta['scale_factor'] if rescale else [1, 1, 1, 1]
        pred_instances.boxsz = torch.stack((det_bboxes[:, 2] * scale_factor[2] - det_bboxes[:, 0] * scale_factor[0],
            det_bboxes[:, 3] * scale_factor[3] - det_bboxes[:, 1] * scale_factor[1]), axis=-1)
        mask_logits = self.bbox_head.mask_head(x, pred_instances)
        if len(pred_instances) > 0:
            mask_logits = aligned_bilinear(mask_logits, self.bbox_head.mask_head.head.mask_out_stride)
            mask_logits = mask_logits[:, :, :resized_im_h, :resized_im_w]
            mask_logits = F.interpolate(
                mask_logits,
                size=(ori_h, ori_w),
                mode="bilinear", align_corners=False
            ).squeeze(1)
            mask_pred = (mask_logits > 0.5)
            flip = img_meta['flip']
            flip_direction = img_meta['flip_direction']
            if flip:
                if flip_direction == 'horizontal':
                    mask_pred = mask_pred.cpu().numpy()[:, :, ::-1]
                elif flip_direction == 'vertical':
                    mask_pred = mask_pred.cpu().numpy()[:, ::-1, :]
                else:
                    raise ValueError
            else:
               mask_pred = mask_pred.cpu().numpy()
        else:
            mask_pred = torch.zeros((self.bbox_head.num_classes, *img_meta['ori_shape'][:2]), dtype=torch.int)
        cls_segms = [[] for _ in range(self.bbox_head.num_classes)]  # BG is not included in num_classes
        cls_scores =[[] for _ in range(self.bbox_head.num_classes)]

        for i, label in enumerate(det_labels):
            score = det_bboxes[i][-1]
            if self.mask_inbox:
                mask_pred_ = torch.zeros_like(mask_pred[i])
                det_bbox_ = det_bboxes[i, :-1].clone()
                det_bbox_[[0, 1]], det_bbox_[[2, 3]] = det_bbox_[[0, 1]].floor(), det_bbox_[[2, 3]].ceil()
                det_bbox_ = det_bbox_.int()
                mask_pred_[det_bbox_[1]:det_bbox_[3], det_bbox_[0]:det_bbox_[2]] = mask_pred[i][det_bbox_[1]:det_bbox_[3], det_bbox_[0]:det_bbox_[2]]
                cls_segms[label].append(mask_pred_.cpu().numpy())
            else:
                cls_segms[label].append(mask_pred[i].cpu().numpy() if isinstance(mask_pred[i], torch.Tensor) else mask_pred[i])
            cls_scores[label].append(score.cpu().numpy())
        
        if return_score:
            return cls_segms,cls_scores
        return cls_segms

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test time augmentation

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.
            [We all use True]
        Returns:
            np.ndarray: proposals
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        bbox_list = self.bbox_head.get_bboxes(
            *outs, img_metas, rescale=rescale)
        if not self.bbox_head.mask_head: # detection only
            bbox_results = [
                bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
                for det_bboxes, det_labels in bbox_list
            ]
            return bbox_results
        else:
            bbox_results = [
                bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
                for det_bboxes, det_labels, _ in bbox_list
            ]
            cls_segms = [
                self.mask2result([xl[[i]] for xl in x], det_labels, inst_inds, img_metas[i], det_bboxes) 
                for i, (det_bboxes, det_labels, inst_inds) in enumerate(bbox_list)
            ]
            
            return list(zip(bbox_results, cls_segms))

    def aug_test_simple(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[ndarray]: bbox results of each class
        """
        raise NotImplementedError

    def aug_test(self, imgs, img_metas, rescale=False):
        return self.aug_test_simple(imgs, img_metas, rescale)