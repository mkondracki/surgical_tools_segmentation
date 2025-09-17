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

from utils import generate_colors, save_mask_as_png, create_visualization, create_output_video

sam_root = Path("/data/mkondrac/foundation_model_cardio/code/SAM/yolo_sam/sam2")
sys.path.insert(0, str(sam_root))  

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

def setup_sam2_model(sam2_checkpoint, model_cfg, device='cuda'):
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    image_predictor = SAM2ImagePredictor(sam2_model)
    return image_predictor

def get_yolo_boxes(yolo_model, image, conf_threshold=0.25):
    results = yolo_model(image)
    boxes = []
    
    for result in results:
        if result.boxes is not None:
            for i in range(len(result.boxes)):
                conf = float(result.boxes.conf[i])
                if conf >= conf_threshold:
                    bbox = result.boxes.xyxy[i].cpu().numpy()
                    cls_id = int(result.boxes.cls[i])
                    
                    boxes.append({
                        'bbox': bbox,
                        'class': cls_id,
                        'confidence': conf
                    })
    return boxes

def process_single_frame_sam2(image_predictor, image, boxes):
    image_predictor.set_image(image)
    frame_results = []
    
    with torch.inference_mode(): # don't track gradients
        for box_dict in boxes:
            bbox = np.array(box_dict["bbox"], dtype=np.float32)
            cls_id = box_dict['class']
            input_box = bbox.reshape(1, 4)
            
            try:
                masks, scores, logits = image_predictor.predict(
                    box=input_box,
                    multimask_output=False,
                )
                
                mask = masks[0] 
                score = scores[0]  
                mask = mask.astype(bool) if mask.dtype != bool else mask
                
                frame_results.append({
                    'class_id': cls_id,
                    'bbox': bbox,
                    'mask': mask,
                    'score': score,
                })
            except Exception as e:
                print(f"Error processing box {bbox} for class {cls_id}: {e}")
                continue
    
    return frame_results

def process_video_frames_sam2(image_predictor, yolo_model, frames_folder, output_dir, device='cuda', 
                        save_visualizations=True, save_masks=True):
    
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
            
            boxes = get_yolo_boxes(yolo_model, image)
            if not boxes:
                continue
            
            # store the time for inference
            start_time = time.time()
            frame_results = process_single_frame_sam2(image_predictor, image, boxes)
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
    
    yolo_model_path = "/data/mkondrac/foundation_model_cardio/code/SAM/yolo_sam/yolo/surgical_tools_det/yolo11n_surgery/weights/best.pt"
    yolo_model = YOLO(yolo_model_path)
    yolo_model.to(device)
    
    videos = [f"video_{i}" for i in range(41, 51)]
    
    model_configs = [
        ("finetuned", 
        "/data/mkondrac/foundation_model_cardio/code/SAM/yolo_sam/sam2/sam2_logs/configs/sam2.1_training/sam2.1_hiera_t_lora_sar_rarp50.yaml/checkpoints/checkpoint_best_val_loss.pt",
        "configs/sam2.1/sam2.1_hiera_t_lora.yaml"),
        
        ("base",
        "/data/mkondrac/foundation_model_cardio/code/SAM/yolo_sam/sam2/checkpoints/sam2.1_hiera_tiny.pt",
        "configs/sam2.1/sam2.1_hiera_t.yaml"),
    ]
    
    for video in videos:
        frames_folder = f"/data/mkondrac/foundation_model_cardio/code/SAM/data/SAR-RARP50_test_set_downsamples_cut_SAM2_format/{video}"

        for model_name, checkpoint, config in model_configs:
            summary = {}
            
            print(f"Processing {video} with {model_name} SAM2 model...")
            
            image_predictor = setup_sam2_model(checkpoint, config, device)

            output_dir = f"/data/mkondrac/foundation_model_cardio/code/SAM/yolo_sam/results/{model_name}_sam2_out/{video}_frame_by_frame"

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