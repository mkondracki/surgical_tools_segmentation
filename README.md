# SAR-RARP50: A Detection-Segmentation Approach

This project explores methods for surgical device segmentation using the **SAR-RARP50** dataset ([Synapse link](https://www.synapse.org/Synapse:syn27618412/wiki/617968)). This dataset contains videos of **Robot-Assisted Radical Prostatectomy** (RARP), with segmentation masks of surgical instruments available every 60 video frames. The task combines **detection, segmentation, and video coherence**.
The approach is based on the **YOLO11n** model for live tracking ([Ultralytics YOLO](https://github.com/ultralytics/ultralytics)). The main focus of this work is the **combination of YOLO with SAM2**, two foundational models that show strong performance on natural images for tracking, detection, and segmentation. The idea is to use YOLO to obtain tracking bounding boxes and then prompt **SAM2** for segmentation.
All the checkpoints are available in this repository: [Google Drive link](https://drive.google.com/drive/u/0/folders/1tpXC1PhkBDdYjXW6sGeS4Ko2BhKJqITl)

---

## Installation
Step-by-step instructions to set up your project locally.

```bash
# Clone the repository
git clone https://github.com/username/repo-name.git

# Install dependencies
pip install -r requirements.txt
