import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# TODO: Import CLASSES from yolo_sam/dataset/save_datasets.py
CLASSES = {
    1: 0,   # Tool clasper
    2: 1,   # Tool wrist
    3: 2,   # Tool shaft
    4: 3,   # Suturing needle
    5: 4,   # Thread
    6: 5,   # Suction tool
    7: 6,   # Needle Holder
    8: 7,   # Clamps
    9: 8,   # Catheter
}

def generate_colors(num_classes=len(CLASSES)):
    colors = {}
    for i in range(num_classes):
        hue = int(i * 180 / max(1, num_classes))
        color = np.array(cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0])
        colors[i] = color.tolist()
    return colors

def save_mask_as_png(frame_results, output_path, image_shape):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mask_image = np.zeros(image_shape[:2], dtype=np.uint8)
    
    for result in frame_results:
        mask = result['mask']
        class_id = result['class_id']
        mask_image[mask] = class_id + 1
    
    cv2.imwrite(str(output_path), mask_image)

def create_visualization(image, frame_results, colors, alpha=0.5, draw_bbox=True):
    overlay = image.copy()
    
    for result in frame_results:
        mask = result['mask']
        cls_id = result['class_id']
        bbox = result['bbox']
        score = result['score']
        
        if mask.dtype != bool:
            mask = mask.astype(bool)
        
        if cls_id in colors:
            color = colors[cls_id]
            overlay[mask] = (alpha * overlay[mask] + (1-alpha) * np.array(color)).astype(np.uint8)
        
        if draw_bbox:
            x1, y1, x2, y2 = bbox.astype(int)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), colors.get(cls_id, [255, 255, 255]), 2)
            
            label = f"Class {cls_id}: {score:.3f}"
            cv2.putText(overlay, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                    colors.get(cls_id, [255, 255, 255]), 2)
        
        if "points" in result and result["points"] is not None:
            pts = result["points"]
            for (px, py) in pts.astype(int):
                cv2.circle(overlay, (px, py), radius=4, color=(0, 255, 255), thickness=-1)
    
    return overlay

def create_visualization(image, frame_results, colors, alpha=0.5, draw_bbox=True, draw_points=True):
    overlay = image.copy()
    
    for result in frame_results:
        mask = result['mask']
        cls_id = result['class_id']
        bbox = result['bbox']
        score = result['score']

        if mask.dtype != bool:
            mask = mask.astype(bool)
        color = tuple(colors.get(cls_id, [255, 255, 255]))

        overlay[mask] = (alpha * overlay[mask] + (1 - alpha) * np.array(color)).astype(np.uint8)

        # draw bbox
        if draw_bbox:
            x1, y1, x2, y2 = bbox.astype(int)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            label = f"Class {cls_id}: {score:.3f}"
            cv2.putText(
                overlay, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )

        # draw points (same color as box)
        if draw_points and "points" in result and result["points"] is not None:
            pts = result["points"]
            for (px, py) in np.round(pts).astype(int):
                cv2.circle(overlay, (px, py), radius=4, color=color, thickness=-1)

    return overlay

def create_output_video(viz_dir, output_video_path, fps=30):
    viz_dir = Path(viz_dir)
    comparison_files = sorted(viz_dir.glob("*_comparison.jpg"), 
                             key=lambda x: int(x.stem.split('_')[1]))
    
    if not comparison_files:
        print("No comparison images found for video creation")
        return
    
    first_frame = cv2.imread(str(comparison_files[0]))
    height, width = first_frame.shape[:2]
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_video = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
    
    print(f"Creating output video with {len(comparison_files)} frames...")
    for img_path in tqdm(comparison_files, desc="Writing video"):
        frame = cv2.imread(str(img_path))
        out_video.write(frame)
    
    out_video.release()
    print(f"Output video saved to: {output_video_path}")