import cv2
import numpy as np
from pathlib import Path
import torch
from tqdm import tqdm
import sys
import os
from ultralytics import YOLO
import time
import json

from utils import generate_colors, save_mask_as_png, create_visualization, create_output_video

def process_single_frame_yolo(model, image):
    height, width = image.shape[:2]
    results = model(image)
    frame_results = []
    
    for result in results:
        if result.masks is not None:
            boxes = result.boxes
            masks = result.masks
            
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i])
                conf = float(boxes.conf[i])
                
                # rescale to the right image size both for bbox and mask
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = min(width-1, int(x2)), min(height-1, int(y2))
                bbox = np.array([x1, y1, x2, y2])
                
                raw_mask = masks.data[i].cpu().numpy().astype(np.uint8)  # [h_pred, w_pred]
                mask = cv2.resize(raw_mask, (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
                
                
                frame_results.append({
                    'class_id': cls_id,
                    'bbox': bbox,
                    'mask': mask,
                    'score': conf,
                })
    
    return frame_results

def process_video_frames_yolo(model, frames_folder, output_dir, save_visualizations=True, save_masks=True):
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
    
    for frame_file in tqdm(frame_files, desc="Processing frames"):
        frame_idx = str(int(frame_file.stem.split('_')[-1]))
        
        try:
            image = cv2.imread(str(frame_file))
            if image is None:
                print(f"Failed to load image: {frame_file}")
                continue
            
            # store the time for inference
            start_time = time.time()
            frame_results = process_single_frame_yolo(model, image)
            elapsed = time.time() - start_time
            inference_times[frame_idx] = elapsed
            
            if not frame_results:
                continue
            processed_frames += 1
            
            if save_masks:
                mask_output_path = masks_dir / f"frame_{frame_idx}.png"
                save_mask_as_png(frame_results, mask_output_path, image.shape)
            
            if save_visualizations:
                overlay = create_visualization(image, frame_results, colors, draw_bbox=True)
                mask_viz = create_visualization(image, frame_results, colors, draw_bbox=False)
                
                combined = np.hstack([image, overlay, mask_viz])
                combined_path = viz_dir / f"frame_{frame_idx}_comparison.jpg"
                cv2.imwrite(str(combined_path), combined)
                
        except Exception as e:
            print(f"Error processing frame {frame_idx}: {e}")
            failed_frames.append(frame_idx)
            continue
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {processed_frames} frames")
    print(f"Failed frames: {len(failed_frames)}")
    print(f"Results saved to: {output_dir}")
    print(f"Mask encoding: PNG files with pixel values = class_id + 1")
    
    return inference_times

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model_path = "/data/mkondrac/foundation_model_cardio/code/SAM/yolo_sam/yolo/surgical_tools_seg/yolo11n_surgery/weights/best.pt"
    model = YOLO(model_path)
    model.to(device)
    
    videos = [f"video_{i}" for i in range(41, 51)]
    
    summary = {}
    
    for video in videos:
        frames_folder = f"/data/mkondrac/foundation_model_cardio/code/SAM/data/SAR-RARP50_test_set_downsamples_cut_SAM2_format/{video}"
        output_dir = f"/data/mkondrac/foundation_model_cardio/code/SAM/yolo_sam/results/yolo_seg_out/{video}_frame_by_frame"
        
        print(f"Processing {video} with YOLO-seg model...")
        
        inference_times = process_video_frames_yolo(
            model=model,
            frames_folder=frames_folder,
            output_dir=output_dir,
            save_visualizations=True,
            save_masks=True
        )

        output_video_path = Path(output_dir) / "segmentation_video.avi"
        create_output_video(
            viz_dir=Path(output_dir) / "visualizations",
            output_video_path=output_video_path,
            fps=30
        )
        
        summary.setdefault(video, {})["yolo-seg"] = inference_times
        with open(Path(output_dir) / "summary_inference.json", "w") as f:
            json.dump(summary, f, indent=4)

        print(f"YOLO-seg pipeline complete for {video}!\n")

if __name__ == "__main__":
    main()