"""
06 — Object detection

Classification asks "what's in the image?". Detection asks "WHAT and WHERE?":
predict a list of (class, bounding box, confidence) tuples per image.

KEY CONCEPTS
  IoU (Intersection over Union)
      Overlap metric: |A ∩ B| / |A ∪ B|. 1.0 = perfect, 0.0 = disjoint.
      Used to match predictions to ground truth and to deduplicate.
  NMS (Non-Max Suppression)
      Remove redundant overlapping boxes for the same object: keep the
      highest-confidence one, drop others with IoU > threshold.
  ANCHORS / ANCHOR-FREE
      Older detectors (Faster R-CNN, classic YOLO) propose boxes from a
      fixed grid of anchor templates. Modern ones (YOLOv8, DETR) skip
      anchors entirely.

This file shows IoU and NMS by hand, then runs YOLOv8 inference.

Run: python 06_object_detection.py
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image


# ───────────────────────────────────────────────────────────
# IoU (boxes as (x1, y1, x2, y2))
# ───────────────────────────────────────────────────────────
def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


print("IoU examples:")
print(f"  same box:        {iou((0, 0, 10, 10), (0, 0, 10, 10)):.2f}")
print(f"  half overlap:    {iou((0, 0, 10, 10), (5, 0, 15, 10)):.2f}")
print(f"  disjoint:        {iou((0, 0, 5, 5),   (10, 10, 15, 15)):.2f}")


# ───────────────────────────────────────────────────────────
# NMS by hand
# ───────────────────────────────────────────────────────────
def nms(boxes, scores, iou_threshold=0.5):
    """Greedy NMS. Returns the indices to KEEP."""
    order = np.argsort(scores)[::-1]
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        # Compute IoU between the highest-scoring box and the rest
        ious = np.array([iou(boxes[i], boxes[j]) for j in order[1:]])
        # Keep only those with low overlap
        order = order[1:][ious < iou_threshold]
    return keep


boxes  = [(0, 0, 10, 10), (1, 1, 11, 11), (50, 50, 60, 60)]
scores = [0.9, 0.8, 0.7]
print(f"\nNMS keeps: {nms(boxes, scores, iou_threshold=0.3)}  "
      "(box 0 wins over the overlapping box 1)")


# ───────────────────────────────────────────────────────────
# Run YOLOv8 on a sample image
# ───────────────────────────────────────────────────────────
# `ultralytics` ships pretrained weights and downloads one on first use.
# We'll run inference on a synthetic image (no real objects → no real
# detections) just to demonstrate the API. See exercise 03 for a real run.
from ultralytics import YOLO

model = YOLO("yolov8n.pt")        # downloads ~6 MB on first run, then cached
print("\nYOLO model classes (first 10):", list(model.names.values())[:10])
print(f"total classes: {len(model.names)}")

# A synthetic image — won't contain real objects but lets us check the pipeline.
fake = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)
results = model.predict(source=fake, verbose=False)

result = results[0]
print(f"\ndetections on synthetic image: {len(result.boxes)}  (expected ≈ 0)")

# How you'd parse REAL detections:
# for box in result.boxes:
#     cls_id = int(box.cls[0])
#     conf   = float(box.conf[0])
#     x1, y1, x2, y2 = box.xyxy[0].tolist()
#     print(f"  {model.names[cls_id]} (conf {conf:.2f}) at ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")


# ───────────────────────────────────────────────────────────
# YOLO model sizes (the n/s/m/l/x scale)
# ───────────────────────────────────────────────────────────
# yolov8n.pt — nano,   ~3 MB,   fastest, good for laptops/edge
# yolov8s.pt — small,  ~22 MB,  great default
# yolov8m.pt — medium, ~50 MB,  better accuracy
# yolov8l.pt — large
# yolov8x.pt — xlarge, the heaviest
# Each one trades latency for accuracy. Pick by your throughput target.


# ───────────────────────────────────────────────────────────
# When you'd train your OWN detector
# ───────────────────────────────────────────────────────────
# • Custom classes ultralytics doesn't already detect.
# • Domain-specific imagery (medical scans, factory parts, satellite).
#
# You'd label data in YOLO format (one .txt per image with
# class_id cx cy w h normalized 0..1) and run:
#
#   model = YOLO("yolov8n.pt")
#   model.train(data="my-dataset.yaml", epochs=50, imgsz=640)
#
# That single line handles loading, augmentation, AdamW, cosine schedule,
# evaluation, and saving.


# ───────────────────────────────────────────────────────────
# Other detection frameworks worth knowing
# ───────────────────────────────────────────────────────────
# • torchvision.models.detection — Faster R-CNN, RetinaNet, FCOS.
#   Pythonic, well-integrated, less polished than ultralytics for training.
# • Detectron2 (Meta) — high-quality, comes with strong baselines for
#   detection / segmentation / panoptic.
# • DETR / RT-DETR — transformer-based, anchor-free, less hand-tuning needed.
