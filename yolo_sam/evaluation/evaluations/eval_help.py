# metrics_miou_mnsd.py
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import jaccard_score, f1_score

import torch
import warnings
import monai
from monai.metrics import SurfaceDiceMetric
from monai.metrics import MeanIoU as MonaiMeanIoU


def per_class_metrics(pred_mask, gt_mask):
    """
    Compute per-class IoU, Dice using sklearn.
    Returns dicts keyed by class ID (background=0 ignored).
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

        iou_dict[int(cls)] = jaccard_score(gt_cls.ravel(), pred_cls.ravel())
        dice_dict[int(cls)] = f1_score(gt_cls.ravel(), pred_cls.ravel())

    return iou_dict, dice_dict


# --------------------------------------------
# Helpers from https://github.com/surgical-vision/SAR_RARP50-evaluation/blob/main/sarrarp50/metrics/segmentation.py#L80
# --------------------------------------------
def _to_one_hot(mask_hw: np.ndarray, n_classes_plus_bg: int) -> torch.Tensor:
    """
    Convert an integer mask (H,W) to one-hot tensor (1, C, H, W).
    """
    t = torch.from_numpy(mask_hw).long().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    return monai.networks.utils.one_hot(t, n_classes_plus_bg, dim=1)  # (1,C,H,W)


def _fix_nans_for_nsd(scores: torch.Tensor, pred_oh: torch.Tensor, ref_oh: torch.Tensor) -> torch.Tensor:
    """
    Replicates the 'fix_nans' logic:
    - If a channel is empty in both pred and ref -> set that score to 1
    - If a channel is empty in only one -> NaN -> set to 0
    """
    # mean over H,W to detect empty channels
    r_m = ref_oh.mean(dim=(2, 3))
    p_m = pred_oh.mean(dim=(2, 3))
    # ignore background channel at index 0 (we compute include_background=False later anyway)
    # but scores length equals number of foreground classes, so align mask to foreground only:
    # channels 1..C-1 -> foreground
    both_empty_fg = ((r_m == 0) & (p_m == 0))[:, 1:]  # (N, C-1)

    # Set both-empty to 1
    scores[both_empty_fg] = 1.0
    # Set remaining NaNs to 0
    scores.nan_to_num_(nan=0.0)
    return scores


def compute_frame_miou(pred_mask: np.ndarray, gt_mask: np.ndarray, n_classes: int) -> float:
    """
    mIoU using MONAI MeanIoU(include_background=False, reduction='mean'),
    consistent with the reference implementation.
    n_classes = number of FOREGROUND classes (exclude background).
    """
    if pred_mask.ndim == 3:
        pred_mask = pred_mask[..., 0]
    if gt_mask.ndim == 3:
        gt_mask = gt_mask[..., 0]

    C_plus_bg = n_classes + 1
    y_pred = _to_one_hot(pred_mask, C_plus_bg)  # (1, C+1, H, W)
    y = _to_one_hot(gt_mask, C_plus_bg)

    metric = MonaiMeanIoU(
        include_background=False,
        reduction="mean",
        get_not_nans=False,
        ignore_empty=False,
    )
    scores = metric(y_pred, y)  # shape: (C) foreground classes
    return float(scores.mean().item())


def compute_frame_mnsd(pred_mask: np.ndarray, gt_mask: np.ndarray, n_classes: int, distance_threshold: int = 10) -> float:
    """
    mNSD using MONAI SurfaceDiceMetric with NaN handling like the reference code.
    distance_threshold in pixels; applied per foreground channel.
    """
    if pred_mask.ndim == 3:
        pred_mask = pred_mask[..., 0]
    if gt_mask.ndim == 3:
        gt_mask = gt_mask[..., 0]

    C_plus_bg = n_classes + 1
    y_pred = _to_one_hot(pred_mask, C_plus_bg)  # (1, C+1, H, W)
    y = _to_one_hot(gt_mask, C_plus_bg)

    channel_tau = [distance_threshold] * n_classes  # one per foreground class

    metric = SurfaceDiceMetric(
        class_thresholds=channel_tau,
        include_background=False,
        reduction="mean",  # per-class, then we'll average
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores = metric(y_pred, y)  # (1, C) foreground classes

    # Apply the NaN fix behavior
    scores = _fix_nans_for_nsd(scores, y_pred, y)  # (1, C)
    return float(scores.mean().item())




def _read_mask_grayscale(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 3:
        img = img[..., 0]
    return img


def evaluate_video(pred_folder, gt_folder, n_classes=9, nsd_distance_threshold=10):
    """
    Computes:
      - per-class IoU/Dice (unchanged)
      - frame-level mIoU & mNSD implemented via MONAI like the reference code
    """
    pred_folder = Path(pred_folder)
    gt_folder = Path(gt_folder)

    gt_files = sorted(gt_folder.glob("*.png"))
    video_data = {}

    for gt_file in gt_files:
        gt_frame_idx = int(gt_file.stem)
        pred_file = pred_folder / f"frame_{gt_frame_idx}.png"

        if not pred_file.exists():
            continue

        gt_mask = _read_mask_grayscale(gt_file)
        pred_mask = _read_mask_grayscale(pred_file)

        # shape match
        if pred_mask.shape != gt_mask.shape:
            pred_mask = cv2.resize(
                pred_mask, (gt_mask.shape[1], gt_mask.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )

        # Keep your detailed per-class reporting
        iou_cls, dice_cls = per_class_metrics(pred_mask, gt_mask)

        # mIoU & mNSD aligned with the reference implementation
        miou = compute_frame_miou(pred_mask, gt_mask, n_classes=n_classes)
        mnsd = compute_frame_mnsd(
            pred_mask, gt_mask,
            n_classes=n_classes,
            distance_threshold=nsd_distance_threshold
        )

        video_data[gt_frame_idx] = {
            "per_class_iou": iou_cls,
            "per_class_dice": dice_cls,
            "mIoU": miou,
            "NSD": mnsd
        }

    return video_data


def evaluate_dataset(model_name, videos, pred_root, gt_root,
                     json_name="evaluation_results.json",
                     n_classes=9, nsd_distance_threshold=10):
    import json

    pred_root = Path(pred_root)
    gt_root = Path(gt_root)

    dataset_results = {
        "model": model_name,
        "videos": {}
    }

    for video in tqdm(videos):
        pred_folder = pred_root / f"{video}_frame_by_frame" / "masks"
        gt_folder = gt_root / video / "segmentation"

        video_data = evaluate_video(
            pred_folder, gt_folder,
            n_classes=n_classes,
            nsd_distance_threshold=nsd_distance_threshold
        )
        dataset_results["videos"][video] = video_data

    # dataset-level averages over frames
    all_miou, all_nsd = [], []
    for frames in dataset_results["videos"].values():
        for frame_data in frames.values():
            all_miou.append(frame_data["mIoU"])
            all_nsd.append(frame_data["NSD"])

    miou_dataset_avg = float(np.mean(all_miou)) if all_miou else 0.0
    nsd_dataset_avg = float(np.mean(all_nsd)) if all_nsd else 0.0
    final_score = float(np.sqrt(miou_dataset_avg * nsd_dataset_avg))

    dataset_results["dataset_metrics"] = {
        "mIoU_avg": miou_dataset_avg,
        "NSD_avg": nsd_dataset_avg,
        "final_score": final_score
    }

    # save to JSON
    output_json = pred_root / json_name
    with open(output_json, "w") as f:
        json.dump(dataset_results, f, indent=4)

    return dataset_results
