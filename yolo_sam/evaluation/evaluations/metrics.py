import cv2
import numpy as np
from pathlib import Path
from scipy.ndimage import distance_transform_edt
from sklearn.metrics import jaccard_score, f1_score
import torch
import numpy as np
from monai.metrics import compute_surface_dice

def per_class_metrics(pred_mask, gt_mask):
    """
    Compute per-class IoU, Dice using sklearn.
    Returns dicts keyed by class ID.
    """
    pred_mask = pred_mask.astype(np.int32)
    gt_mask = gt_mask.astype(np.int32)
    
    classes = np.unique(gt_mask)
    classes = classes[classes != 0]  # ignore background
    
    iou_dict = {}
    dice_dict = {}
    
    for cls in classes:
        gt_cls = (gt_mask == cls).astype(int)
        pred_cls = (pred_mask == cls).astype(int)
        
        if gt_cls.sum() == 0:
            continue
        
        # IoU
        iou_dict[int(cls)] = jaccard_score(gt_cls.ravel(), pred_cls.ravel())
        
        # Dice (F1 score)
        dice_dict[int(cls)] = f1_score(gt_cls.ravel(), pred_cls.ravel())
    
    return iou_dict, dice_dict


def compute_nsd(pred_mask, gt_mask, distance_threshold=10):
    """
    Compute Normalized Surface Dice (NSD) using MONAI built-in function.
    
    pred_mask: np.array (H,W), integer class IDs starting from 1
    gt_mask: np.array (H,W), integer class IDs starting from 1
    distance_threshold: tolerance in pixels
    """
    pred_mask = torch.as_tensor(pred_mask, dtype=torch.int64)
    gt_mask = torch.as_tensor(gt_mask, dtype=torch.int64)
    
    classes = gt_mask.unique().numpy()
    
    gt_onehot = torch.zeros((1, int(len(classes)), *gt_mask.shape), dtype=torch.float32)
    pred_onehot = torch.zeros_like(gt_onehot)
    for i, c in enumerate(classes):
        if c!=0:
            if c in classes:
                gt_onehot[0][i] = (gt_mask == c).long()
            if c in pred_mask.unique():
                pred_onehot[0][i] = (pred_mask == c).long()
    
    nsd = compute_surface_dice(
        y_pred=pred_onehot,  # add batch & channel dims
        y=gt_onehot,
        class_thresholds=[distance_threshold] * int(len(classes)-1) # ignore background class 0
    )
    
    return float(nsd.mean())
