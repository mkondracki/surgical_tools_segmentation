import cv2
import json
import numpy as np
from pathlib import Path
import torch
from tqdm import tqdm
import sys
import os
from ultralytics import YOLO
import time
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional

from utils import generate_colors, save_mask_as_png, create_visualization, create_output_video

sam_root = Path("/data/mkondrac/foundation_model_cardio/code/SAM/yolo_sam/sam2")
sys.path.insert(0, str(sam_root))  

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

def setup_sam2_model(sam2_checkpoint, model_cfg, device='cuda'):
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    image_predictor = SAM2ImagePredictor(sam2_model)
    return image_predictor

def _points_from_mask(mask: np.ndarray, max_points: int = 2) -> np.ndarray:
    """
    Derive 1–2 positive points from a YOLO-seg instance mask (float prob [0,1] or 0/1).
    Returns Nx2 array in absolute pixel coords: [x, y].
    """
    prob = mask.astype(np.float32)
    H, W = prob.shape[:2]
    pts = []

    # 1) center-of-mass on a binarized mask
    bin_m = (prob > 0.5).astype(np.uint8)
    if bin_m.sum() > 0:
        ys, xs = np.where(bin_m > 0)
        cx, cy = float(xs.mean()), float(ys.mean())
        pts.append([cx, cy])

    # 2) max-probability pixel (peak)
    flat_idx = int(prob.argmax())
    py, px = divmod(flat_idx, W)
    pts.append([float(px), float(py)])

    return np.array(pts[:max_points], dtype=np.float32)


def _points_from_box(bbox: np.ndarray) -> np.ndarray:
    """
    Single positive point at the box center. bbox is [x1,y1,x2,y2].
    """
    x1, y1, x2, y2 = bbox.astype(np.float32)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    return np.array([[cx, cy]], dtype=np.float32)

def extract_tracked_instances(res, conf_threshold: float = 0.25) -> Tuple[List[dict], np.ndarray, int]:
    frame_bgr = res.orig_img  # HxWx3 BGR
    frame_path = getattr(res, "path", None)
    try:
        frame_idx = int(Path(frame_path).stem.split('_')[-1]) if frame_path else int(getattr(res, "frame", 0))
    except Exception:
        frame_idx = int(getattr(res, "frame", 0))

    boxes = res.boxes
    ids   = None if boxes is None else boxes.id
    if boxes is None or ids is None:
        return [], frame_bgr, frame_idx

    ids_np   = ids.detach().cpu().numpy().astype(int).ravel()
    xyxy_np  = boxes.xyxy.detach().cpu().numpy().astype(np.float32)
    conf_np  = boxes.conf.detach().cpu().numpy().astype(np.float32).ravel()
    cls_np   = boxes.cls.detach().cpu().numpy().astype(int).ravel()

    masks_np = None
    if getattr(res, "masks", None) is not None and res.masks is not None:
        # (N, H, W) in original image space
        masks_np = res.masks.data.detach().cpu().numpy().astype(np.float32)

    instances = []
    for i in range(len(ids_np)):
        if conf_np[i] < conf_threshold:
            continue
        inst = {
            'track_id': int(ids_np[i]),
            'bbox': xyxy_np[i],
            'class': int(cls_np[i]),
            'conf': float(conf_np[i]),
            'mask': None if masks_np is None else masks_np[i]
        }
        instances.append(inst)

    return instances, frame_bgr, frame_idx


def process_video_frames_sam2(image_predictor, yolo_model, frames_folder, output_dir, device='cuda', 
                        save_visualizations=True, save_masks=True,
                        tracker_cfg: str = 'botsort.yaml'):
    
    frames_folder = Path(frames_folder)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if save_visualizations:
        viz_dir = output_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
    
    if save_masks:
        masks_dir = output_dir / "masks"
        masks_dir.mkdir(exist_ok=True)
    
    frame_files = sorted(frames_folder.glob("*.jpg"), 
                        key=lambda x: int(x.stem.split('_')[-1]))
    print(f"Found {len(frame_files)} frames")
    
    colors = generate_colors()
    
    processed_frames = 0
    failed_frames = []
    inference_times = {}
    
    stream = yolo_model.track(
        source=str(frames_folder),
        stream=True,
        tracker=tracker_cfg,
        conf=0.25,
        verbose=False
    )
    
    # for frame_file in tqdm(frame_files, desc="Processing frames"):
    for res in tqdm(stream, desc="Processing (tracked)"):        
        try:
            instances, image, frame_idx = extract_tracked_instances(res)
            if image is None or len(instances) == 0:
                print(f"Failed to load image at: {frame_idx}")
                continue
            
            
            sam_prompts = []
            for det in instances:
                bbox = det['bbox'].astype(np.float32)          # [4]
                box_for_sam = bbox.reshape(1, 4)               # SAM expects (1,4)
                
                Hi, Wi = image.shape[:2]

                mask = det.get('mask', None)                   # (Hm,Wm) or None
                if mask is not None:
                    mask = mask.astype(np.float32)
                    Hm, Wm = mask.shape[:2]
                    if (Hm, Wm) != (Hi, Wi):
                        mask = cv2.resize(mask, (Wi, Hi), interpolation=cv2.INTER_LINEAR)
                        if mask.min() < 0.0 or mask.max() > 1.0:
                            mask = np.clip(mask, 0.0, 1.0)

                    if (mask > 0.5).sum() > 0:
                        pts = _points_from_mask(mask, max_points=1)  # COM + peak in image coords
                    else:
                        pts = _points_from_box(bbox)
                else:
                    pts = _points_from_box(bbox)                 # fallback if no mask head

                if pts.size == 0:
                    pts = _points_from_box(bbox)

                lbls = np.ones((len(pts),), dtype=np.int32)    # all positive

                sam_prompts.append({
                    'track_id': int(det['track_id']),
                    'class_id': int(det['class']),
                    'box': box_for_sam,
                    'points': pts,
                    'point_labels': lbls
                })

            # SAM2 inference with built-in postprocess (per image once)
            image_predictor.set_image(image)
            start_time = time.time()
            frame_results = []


            with torch.inference_mode():
                for sp in sam_prompts:
                    try:
                        masks, scores, logits = image_predictor.predict(
                            box=sp['box'],
                            point_coords=sp['points'],
                            point_labels=sp['point_labels'],
                            multimask_output=False,
                            return_logits=False,
                        )

                        # Built-in SAM postprocess to original image size (if available)
                        post_mask = masks[0]

                        frame_results.append({
                            'track_id': int(sp['track_id']),
                            'class_id': int(sp['class_id']),
                            'bbox': sp['box'].reshape(-1),
                            'mask': post_mask.astype(bool),
                            'score': float(scores[0]),
                            'points': sp['points'],
                        })
                    except Exception as e:
                        print(f"SAM2 error (track {sp['track_id']}): {e}")
                        continue

            elapsed = time.time() - start_time
            inference_times[str(frame_idx)] = elapsed

            if not frame_results:
                continue
            processed_frames += 1

            # Save outputs
            if save_masks:
                mask_output_path = masks_dir / f"frame_{frame_idx}.png"
                save_mask_as_png(frame_results, mask_output_path, image.shape)

            if save_visualizations:
                overlay = create_visualization(image, frame_results, colors, draw_bbox=True, draw_points=True)
                mask_viz = create_visualization(image, frame_results, colors, draw_bbox=False, draw_points=False)
                combined = np.hstack([image, overlay, mask_viz])
                combined_path = viz_dir / f"frame_{frame_idx}_comparison.jpg"
                cv2.imwrite(str(combined_path), combined)

        except Exception as e:
            print(f"Error processing frame {frame_idx}: {e}")
            failed_frames.append(str(frame_idx))
            continue
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {processed_frames} frames")
    print(f"Failed frames: {len(failed_frames)}")
    print(f"Results saved to: {output_dir}")
    print(f"Mask encoding: PNG files with pixel values = class_id + 1")
    
    return inference_times

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    yolo_model_path = "/data/mkondrac/foundation_model_cardio/code/SAM/yolo_sam/yolo/surgical_tools_seg/yolo11n_surgery/weights/best.pt"
    yolo_model = YOLO(yolo_model_path)
    yolo_model.to(device)
    
    videos = [f"video_{i}" for i in range(41, 51)]
    
    model_configs = [
        ("finetuned", 
        "/data/mkondrac/foundation_model_cardio/code/SAM/yolo_sam/sam2/sam2_logs/configs/sam2.1_training/sam2.1_hiera_t_lora_sar_rarp50.yaml/checkpoints/checkpoint_best_val_loss.pt",
        "configs/sam2.1/sam2.1_hiera_t_lora.yaml"),
        
        # ("base",
        # "/data/mkondrac/foundation_model_cardio/code/SAM/yolo_sam/sam2/checkpoints/sam2.1_hiera_tiny.pt",
        # "configs/sam2.1/sam2.1_hiera_t.yaml"),
    ]
    
    for video in videos:
        frames_folder = f"/data/mkondrac/foundation_model_cardio/code/SAM/data/SAR-RARP50_test_set_downsamples_cut_SAM2_format/{video}"

        for model_name, checkpoint, config in model_configs:
            summary = {}
            
            print(f"Processing {video} with {model_name} SAM2 model...")
            
            image_predictor = setup_sam2_model(checkpoint, config, device)

            output_dir = f"/data/mkondrac/foundation_model_cardio/code/SAM/yolo_sam/results/{model_name}_tracked_sam2_out/{video}_frame_by_frame"

            inference_times = process_video_frames_sam2(
                image_predictor=image_predictor,
                yolo_model=yolo_model,
                frames_folder=frames_folder,
                output_dir=output_dir,
                device=device,
                save_visualizations=True,
                save_masks=True
            )

            output_video_path = Path(output_dir) / "segmentation_video.avi"
            create_output_video(
                viz_dir=Path(output_dir) / "visualizations",
                output_video_path=output_video_path,
                fps=30
            )
            
            summary.setdefault(video, {})[model_name] = inference_times
            with open(Path(output_dir) / "summary_inference.json", "w") as f:
                json.dump(summary, f, indent=4)

            print(f"SAM2 pipeline complete for {video} with {model_name} model!\n")

if __name__ == "__main__":
    main()