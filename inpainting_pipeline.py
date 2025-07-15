 
import torch
import cv2
import numpy as np
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor
from diffusers import StableDiffusionInpaintPipeline
from ultralytics import YOLO
import os

# --- Load Models ---
# Load YOLOv8 for object detection
detector = YOLO("yolov8s.pt")

# Load Segment Anything Model (SAM)
sam = sam_model_registry["vit_h"]("sam_vit_h_4b8939.pth")
sam_predictor = SamPredictor(sam)

# Load Stable Diffusion inpainting pipeline
inpaint_pipeline = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting")
inpaint_pipeline = inpaint_pipeline.to("cuda" if torch.cuda.is_available() else "cpu")

# --- Utility Functions ---
def load_image(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image_rgb

def detect_and_segment(image_rgb):
    # YOLO detection
    results = detector.predict(source=image_rgb, conf=0.3)
    bboxes = results[0].boxes.xyxy.cpu().numpy()

    # SAM segmentation
    sam_predictor.set_image(image_rgb)
    masks = []
    for box in bboxes:
        input_box = np.array(box, dtype=np.float32)
        mask, _, _ = sam_predictor.predict(box=input_box, multimask_output=False)
        masks.append(mask)
    return masks, bboxes

def create_mask_from_segment(masks, image_shape):
    final_mask = np.zeros(image_shape[:2], dtype=np.uint8)
    for mask in masks:
        final_mask = np.logical_or(final_mask, mask)
    return (final_mask * 255).astype(np.uint8)

def apply_inpainting(image_rgb, mask_np, prompt, output_path):
    image_pil = Image.fromarray(image_rgb).resize((512, 512))
    mask_pil = Image.fromarray(mask_np).resize((512, 512))
    result = inpaint_pipeline(prompt=prompt, image=image_pil, mask_image=mask_pil).images[0]
    result.save(output_path)
    print(f"Saved inpainted image to {output_path}")

# --- Main Execution ---
def run_pipeline(image_path, prompt, output_path="inpainted_result.png"):
    print("Loading image...")
    image_rgb = load_image(image_path)
    print("Detecting and segmenting objects...")
    masks, bboxes = detect_and_segment(image_rgb)
    print(f"Detected {len(bboxes)} objects")

    print("Creating combined mask...")
    mask_np = create_mask_from_segment(masks, image_rgb.shape)

    print("Running inpainting...")
    apply_inpainting(image_rgb, mask_np, prompt, output_path)
