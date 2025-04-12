import torch
from ultralytics import YOLO, SAM
from huggingface_hub import hf_hub_download
import os
import uuid
from PIL import Image

# Downloading model from Hugging Face Repo
yolo_model_path = hf_hub_download(
    repo_id="mibrahimm/brain_tumor_detection",
    filename="yolov11_brain_tumor.pt",
)

# Download SAM model from Hugging Face Repo
sam_model_path = hf_hub_download(
    repo_id="mibrahimm/brain_tumor_detection",
    filename="sam2_b.pt",
)

# Load models
yolo_model = YOLO(yolo_model_path)
sam_model = SAM(sam_model_path)

# PROCESS IMAGE FUNCTION
def process_image(image_path):
    # OBJECT DETECTION OF TEST IMAGE
    results = yolo_model(image_path, save=False)

    output_img_path = f"static/{uuid.uuid4().hex}.jpg"

    for result in results:
        class_ids = result.boxes.cls.tolist()
        if class_ids:
            boxes = result.boxes.xyxy

            # SEGMENTATION OF TEST IMAGE
            sam_results = sam_model(result.orig_img, bboxes=boxes, verbose=False, save=False)
            annotated = sam_results[0].plot()
            Image.fromarray(annotated).save(output_img_path)

    return output_img_path
