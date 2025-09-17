import cv2
import os
import re
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import shutil
from scipy.spatial.distance import cdist

TRAIN_ROOT = "/data/mkondrac/foundation_model_cardio/code/SAM/data/SAR-RARP50_train_set"
TEST_ROOT = "/data/mkondrac/foundation_model_cardio/code/SAM/data/SAR-RARP50_test_set"
OBJ_DETECT_ROOT = "/data/mkondrac/foundation_model_cardio/code/SAM/yolo_sam/dataset/object_detection"
OBJ_SEG_ROOT_YOLO = "/data/mkondrac/foundation_model_cardio/code/SAM/yolo_sam/dataset/object_segmentation/yolo_format"
OBJ_SEG_ROOT_SAM = "/data/mkondrac/foundation_model_cardio/code/SAM/yolo_sam/dataset/object_segmentation/sam_format"

TARGET_WIDTH = 426
TARGET_HEIGHT = 240

# class mapping: pixel value -> YOLO class ID
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

def setup_folders():
    # YOLO detection and segmentation format (images/labels/split)
    for root in [OBJ_DETECT_ROOT, OBJ_SEG_ROOT_YOLO]:
        if os.path.exists(root):
            print(f"Removing existing dataset: {root}")
            shutil.rmtree(root)
        for split in ["train", "val", "test"]:
            os.makedirs(f"{root}/images/{split}")
            os.makedirs(f"{root}/labels/{split}")

    # SAM format (split/images and split/labels)
    if os.path.exists(OBJ_SEG_ROOT_SAM):
        print(f"Removing existing dataset: {OBJ_SEG_ROOT_SAM}")
        shutil.rmtree(OBJ_SEG_ROOT_SAM)
    for split in ["train", "val", "test"]:
        os.makedirs(f"{OBJ_SEG_ROOT_SAM}/{split}/images")
        os.makedirs(f"{OBJ_SEG_ROOT_SAM}/{split}/labels")

def get_surgery_id(folder_name):
    # extract surgery ID from folder name (e.g. video_17_1 -> 17)
    match = re.search(r"video_(\d+)", folder_name)
    return int(match.group(1)) if match else None

def get_components(mask, min_area=30, dilate_iter=5, kernel_size=5):
    # dilate to enforce stronger connectivity
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask_dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=dilate_iter)

    num_labels, labels = cv2.connectedComponents(mask_dilated)
    components = []

    for i in range(1, num_labels):  # skip background (0)
        region = (labels == i)
        comp_mask = np.logical_and(region, mask.astype(bool))
        if comp_mask.sum() < min_area:
            continue

        coords = np.where(comp_mask)
        centroid_y = np.mean(coords[0])
        centroid_x = np.mean(coords[1])

        components.append((comp_mask, centroid_x, centroid_y))

    return components

def create_sam_masks_with_components(mask_small, pixel_val):
    # create separate masks for connected components of tool classes
    class_mask = (mask_small == pixel_val)
    if not class_mask.any():
        return []
    
    components = get_components(class_mask)
    component_masks = []
    
    for i, (component_mask, _, _) in enumerate(components):
        # create a mask with only this component, keeping the original pixel value
        individual_mask = np.zeros_like(mask_small)
        individual_mask[component_mask] = 255.
        component_masks.append(individual_mask)
    
    return component_masks

def mask_to_bbox(mask, pixel_val):
    # convert binary mask to normalized bounding box coordinates
    components = get_components(mask)
    if not components:
        return []
    
    bboxes = []
    
    for component_mask, _, _ in components:
        coords = np.where(component_mask > 0)
        if len(coords[0]) == 0:
            continue
        
        h, w = mask.shape
        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()
        
        x_center = (x_min + x_max) / 2 / w
        y_center = (y_min + y_max) / 2 / h
        bbox_w = (x_max - x_min) / w
        bbox_h = (y_max - y_min) / h
        
        bboxes.append((CLASSES[pixel_val], x_center, y_center, bbox_w, bbox_h))
    
    return bboxes


def mask_to_polygons(mask, pixel_val):
    # convert binary mask to normalized polygon with left/right detection    
    polygons = []
    
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return polygons
    
    h, w = mask.shape
    
    for contour in contours:
        polygon = contour.squeeze()
        if len(polygon.shape) == 1:
            # This contour has only one point, skip it
            continue
        
        normalized_coords = []
        for x, y in polygon:
            normalized_coords.extend([x / w, y / h])
        
        class_id = CLASSES[pixel_val]
        polygons.append((class_id, normalized_coords))
    
    return polygons


def process_frame(frame, mask, split, filename):
    # process a single frame for all formats
    # resize
    frame_small = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)
    mask_small = cv2.resize(mask, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_NEAREST)
    
    # SAM format with connected components processing
    sam_img_folder = os.path.join(OBJ_SEG_ROOT_SAM, f"{split}/images/{filename}")
    sam_label_folder = os.path.join(OBJ_SEG_ROOT_SAM, f"{split}/labels/{filename}")
    os.makedirs(sam_img_folder, exist_ok=True)
    os.makedirs(sam_label_folder, exist_ok=True)
    
    # save the original image
    cv2.imwrite(os.path.join(sam_img_folder, "00000.jpg"), frame_small)
    
    # process each class and create individual masks for connected components
    mask_counter = 0
    for pixel_val in CLASSES.keys():
        component_masks = create_sam_masks_with_components(mask_small, pixel_val)
        
        for i, component_mask in enumerate(component_masks):
            mask_dir = f"{mask_counter:02d}_{i}" if len(component_masks) > 1 else f"{mask_counter:02d}"
            os.makedirs(os.path.join(sam_label_folder, mask_dir), exist_ok=True)
            cv2.imwrite(os.path.join(sam_label_folder, mask_dir, f"{0:05d}.png"), component_mask)
        
        mask_counter += 1
    
    # if no masks were created, save an empty mask to maintain consistency
    if mask_counter == 0:
        print(f"Empty mask for {filename}.")
    
    # YOLO format (unchanged)
    for root in [OBJ_DETECT_ROOT, OBJ_SEG_ROOT_YOLO]:
        img_path = f"{root}/images/{split}/{filename}.jpg"
        cv2.imwrite(img_path, frame_small)
        
    # process YOLO labels
    detection_labels = []
    segmentation_labels = []
    
    for pixel_val in CLASSES.keys():
        class_mask = (mask_small == pixel_val)
        if not class_mask.any():
            continue
        
        # YOLO Detection (bbox)
        bboxes = mask_to_bbox(class_mask, pixel_val)
        for bbox_data in bboxes:
            class_id = bbox_data[0]
            bbox_coords = bbox_data[1:]
            detection_labels.append(f"{class_id} {' '.join(f'{x:.6f}' for x in bbox_coords)}")
        
        # YOLO Segmentation (polygon)
        polygons = mask_to_polygons(class_mask, pixel_val)
        for polygon_data in polygons:
            class_id = polygon_data[0]
            coords = polygon_data[1]
            coords_str = ' '.join(f'{x:.6f}' for x in coords)
            segmentation_labels.append(f"{class_id} {coords_str}")
    
    with open(f"{OBJ_DETECT_ROOT}/labels/{split}/{filename}.txt", "w") as f:
        f.write('\n'.join(detection_labels))
    
    with open(f"{OBJ_SEG_ROOT_YOLO}/labels/{split}/{filename}.txt", "w") as f:
        f.write('\n'.join(segmentation_labels))
        

def process_video_folder(video_folder, split):
    # process a single video folder
    video_path = os.path.join(video_folder, "video_left.avi")
    masks_folder = os.path.join(video_folder, "segmentation")
    
    if not os.path.exists(video_path):
        return
    
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    folder_name = os.path.basename(video_folder)
    
    while cap.read()[0]:
        mask_file = os.path.join(masks_folder, f"{frame_idx:09d}.png")
        if os.path.exists(mask_file):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
            
            if ret and mask is not None:
                if not np.any(mask):
                    print(f"Skipping frame {frame_idx} in {folder_name} (mask all zeros)")
                else:
                    filename = f"{folder_name}_{frame_idx:09d}"
                    process_frame(frame, mask, split, filename)
        
        frame_idx += 1
    
    cap.release()


def process_train_val_split():
    # group videos by surgery ID
    surgeries = {}
    for folder in os.listdir(TRAIN_ROOT):
        surgery_id = get_surgery_id(folder)
        if surgery_id:
            surgeries.setdefault(surgery_id, []).append(os.path.join(TRAIN_ROOT, folder))
    
    # split cases 80/20
    surgery_ids = list(surgeries.keys())
    train_ids, val_ids = train_test_split(surgery_ids, test_size=0.2, random_state=0)
    
    print(f"Train surgeries: {sorted(train_ids)}")
    print(f"Val surgeries: {sorted(val_ids)}")
    
    for split, ids in [("train", train_ids), ("val", val_ids)]:
        for surgery_id in tqdm(ids, desc=f"Processing {split}"):
            for video_folder in surgeries[surgery_id]:
                process_video_folder(video_folder, split)

def process_test_split():
    for folder in tqdm(os.listdir(TEST_ROOT), desc="Processing test"):
        video_folder = os.path.join(TEST_ROOT, folder)
        if os.path.isdir(video_folder):
            process_video_folder(video_folder, "test")
            
def create_class_mapping():
    class_names = {
        0: "Tool_clasper", 1: "Tool_wrist", 2: "Tool_shaft", 3: "Suturing_needle",
        4: "Thread", 5: "Suction_tool", 6: "Needle_Holder", 7: "Clamps", 8: "Catheter"
    }
    
    # YOLO format class file
    with open(os.path.join(os.path.dirname(OBJ_DETECT_ROOT), "classes.txt"), "w") as f:
        for class_id in range(len(class_names)):
            f.write(f"{class_names.get(class_id, f'class_{class_id}')}\n")

def main():
    setup_folders()
    process_train_val_split()
    process_test_split()
    create_class_mapping()
    print(f"YOLO detection dataset created at: {OBJ_DETECT_ROOT}")
    print(f"YOLO segmentation dataset created at: {OBJ_SEG_ROOT_YOLO}")
    print(f"SAM segmentation dataset created at: {OBJ_SEG_ROOT_SAM}")

if __name__ == "__main__":
    main()