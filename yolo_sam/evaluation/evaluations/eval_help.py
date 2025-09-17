import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from metrics import *
import json

def evaluate_video(pred_folder, gt_folder):
    pred_folder = Path(pred_folder)
    gt_folder = Path(gt_folder)

    gt_files = sorted(gt_folder.glob("*.png"))
    video_data = {}  # store per-frame data

    for gt_file in gt_files:
        gt_frame_idx = int(gt_file.stem)
        pred_file = pred_folder / f"frame_{gt_frame_idx}.png"

        if not pred_file.exists():
            continue

        pred_mask = cv2.imread(str(pred_file), cv2.IMREAD_UNCHANGED)
        gt_mask = cv2.imread(str(gt_file), cv2.IMREAD_GRAYSCALE)

        if pred_mask.shape != gt_mask.shape:
            pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_NEAREST)

        iou_cls, dice_cls = per_class_metrics(pred_mask, gt_mask)
        miou = float(np.mean(list(iou_cls.values()))) if iou_cls else 0.0
        nsd = compute_nsd(pred_mask, gt_mask)

        video_data[gt_frame_idx] = {
            "per_class_iou": iou_cls,
            "per_class_dice": dice_cls,
            "mIoU": miou,
            "NSD": nsd
        }

    return video_data


def evaluate_dataset(model_name, videos, pred_root, gt_root, json_name="evaluation_results.json"):
    dataset_results = {
        "model": model_name,
        "videos": {}
    }

    for video in tqdm(videos):
        pred_folder = Path(pred_root) / f"{video}_frame_by_frame" / "masks"
        gt_folder = Path(gt_root) / video / "segmentation"

        video_data = evaluate_video(pred_folder, gt_folder)
        dataset_results["videos"][video] = video_data

    # compute dataset-level mIoU and NSD averages
    all_miou = []
    all_nsd = []
    for video, frames in dataset_results["videos"].items():
        for frame_data in frames.values():
            all_miou.append(frame_data["mIoU"])
            all_nsd.append(frame_data["NSD"])

    miou_dataset_avg = float(np.mean(all_miou)) if all_miou else 0.0
    nsd_dataset_avg = float(np.mean(all_nsd)) if all_nsd else 0.0
    final_score = np.sqrt(miou_dataset_avg * nsd_dataset_avg)

    dataset_results["dataset_metrics"] = {
        "mIoU_avg": miou_dataset_avg,
        "NSD_avg": nsd_dataset_avg,
        "final_score": final_score
    }

    # save to JSON
    output_json = Path(pred_root) / json_name
    with open(output_json, "w") as f:
        json.dump(dataset_results, f, indent=4)

    return dataset_results