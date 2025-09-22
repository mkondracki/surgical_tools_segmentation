import cv2
import numpy as np
from pathlib import Path
from scipy.ndimage import distance_transform_edt
from sklearn.metrics import jaccard_score, f1_score
import torch
import numpy as np
from monai.metrics import compute_surface_dice

# def per_class_metrics(pred_mask, gt_mask):
#     """
#     Compute per-class IoU, Dice using sklearn.
#     Returns dicts keyed by class ID.
#     """
#     pred_mask = pred_mask.astype(np.int32)
#     gt_mask = gt_mask.astype(np.int32)
    
#     classes = np.unique(gt_mask)
#     classes = classes[classes != 0]  # ignore background
    
#     iou_dict = {}
#     dice_dict = {}
    
#     for cls in classes:
#         gt_cls = (gt_mask == cls).astype(int)
#         pred_cls = (pred_mask == cls).astype(int)
        
#         if gt_cls.sum() == 0:
#             continue
        
#         # IoU
#         iou_dict[int(cls)] = jaccard_score(gt_cls.ravel(), pred_cls.ravel())
        
#         # Dice (F1 score)
#         dice_dict[int(cls)] = f1_score(gt_cls.ravel(), pred_cls.ravel())
    
#     return iou_dict, dice_dict


# def compute_nsd(pred_mask, gt_mask, distance_threshold=10):
#     """
#     Compute Normalized Surface Dice (NSD) using MONAI built-in function.
    
#     pred_mask: np.array (H,W), integer class IDs starting from 1
#     gt_mask: np.array (H,W), integer class IDs starting from 1
#     distance_threshold: tolerance in pixels
#     """
#     pred_mask = torch.as_tensor(pred_mask, dtype=torch.int64)
#     gt_mask = torch.as_tensor(gt_mask, dtype=torch.int64)
    
#     classes = gt_mask.unique().numpy()
    
#     gt_onehot = torch.zeros((1, int(len(classes)), *gt_mask.shape), dtype=torch.float32)
#     pred_onehot = torch.zeros_like(gt_onehot)
#     for i, c in enumerate(classes):
#         if c!=0:
#             if c in classes:
#                 gt_onehot[0][i] = (gt_mask == c).long()
#             if c in pred_mask.unique():
#                 pred_onehot[0][i] = (pred_mask == c).long()
    
#     nsd = compute_surface_dice(
#         y_pred=pred_onehot,  # add batch & channel dims
#         y=gt_onehot,
#         class_thresholds=[distance_threshold] * int(len(classes)-1) # ignore background class 0
#     )
    
#     return float(nsd.mean())




import torch
import monai

def _to_one_hot(mask_hw, n_classes_plus_bg: int) -> torch.Tensor:
    """
    Convert an integer mask (H,W) to one-hot tensor (1, C, H, W).
    """
    t = torch.as_tensor(mask_hw, dtype=torch.long).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    return monai.networks.utils.one_hot(t, n_classes_plus_bg, dim=1)  # (1,C,H,W)

def per_class_metrics(pred_mask, gt_mask, n_classes: int):
    """
    Compute per-class IoU and Dice using MONAI metrics.
    Returns dicts keyed by class ID (1..n_classes). Background (0) is ignored.
    Only classes present in the GT for the frame are returned (like before).
    """
    if pred_mask.ndim == 3:
        pred_mask = pred_mask[..., 0]
    if gt_mask.ndim == 3:
        gt_mask = gt_mask[..., 0]

    C_plus_bg = n_classes + 1
    y_pred = _to_one_hot(pred_mask, C_plus_bg)  # (1, C+1, H, W)
    y      = _to_one_hot(gt_mask,   C_plus_bg)

    gt_present = (y.sum(dim=(0, 2, 3)) > 0)            # (C+1,)
    gt_present_fg = gt_present[1:]                     # (C,) foreground only

    # IoU per class
    iou_metric = monai.metrics.MeanIoU(
        include_background=False,
        reduction="none",       # keep per-class scores
        get_not_nans=False,
        ignore_empty=False,
    )
    iou_scores = iou_metric(y_pred, y).squeeze(0)      # (C,)

    # Dice per class
    dice_metric = monai.metrics.DiceMetric(
        include_background=False,
        reduction="none",       # keep per-class scores
        ignore_empty=False,
    )
    dice_scores = dice_metric(y_pred, y).squeeze(0)    # (C,)

    # build dicts only for classes present in GT 
    iou_dict, dice_dict = {}, {}
    for idx in range(n_classes):            # idx: 0..C-1 (foreground channels)
        if bool(gt_present_fg[idx]):
            cls_id = idx + 1                # map channel index -> class id
            iou_val  = float(iou_scores[idx].item())
            dice_val = float(dice_scores[idx].item())
            iou_dict[cls_id]  = iou_val
            dice_dict[cls_id] = dice_val

    return iou_dict, dice_dict
