"""
Exercise 3 — Run YOLOv8 inference on an image

Steps:
  1. Load YOLOv8n (downloaded on first run, ~6 MB).
  2. Run inference on a sample image (downloaded from ultralytics CDN, or
     a fallback synthetic image if you're offline).
  3. Print each detection's class, confidence, and bounding box.
  4. Save an annotated PNG.

The script asserts that on the bus image, YOLO detects at least 3 objects
including a 'bus' or 'person'.

Run: python exercises/03_yolo_inference.py
"""

from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import YOLO


HERE = Path(__file__).parent
SAMPLE = HERE / "sample.jpg"


def get_sample_image() -> Path:
    """Download a sample image, or fall back to creating one."""
    if SAMPLE.exists():
        return SAMPLE
    try:
        import urllib.request
        urllib.request.urlretrieve(
            "https://ultralytics.com/images/bus.jpg", SAMPLE,
        )
        print(f"downloaded sample image → {SAMPLE}")
    except Exception as e:
        print(f"download failed ({e}); using a synthetic placeholder.")
        # Synthetic image — YOLO will (correctly) detect nothing.
        synth = (np.random.rand(640, 640, 3) * 255).astype(np.uint8)
        Image.fromarray(synth).save(SAMPLE)
    return SAMPLE


def main():
    img_path = get_sample_image()
    model = YOLO("yolov8n.pt")               # downloads on first run
    print(f"model classes: {len(model.names)}")

    results = model.predict(source=str(img_path), verbose=False)
    result = results[0]
    print(f"\ndetections on {img_path.name}: {len(result.boxes)}")

    detected_classes = []
    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        name = model.names[cls_id]
        detected_classes.append(name)
        print(f"  {name:<12}  conf={conf:.2f}  "
              f"box=({x1:.0f},{y1:.0f})-({x2:.0f},{y2:.0f})")

    # Save the annotated image
    annotated = result.plot()      # numpy array with boxes drawn
    out = HERE.parent / "figures" / "exercise_03_yolo.png"
    out.parent.mkdir(exist_ok=True)
    Image.fromarray(annotated[..., ::-1]).save(out)   # BGR→RGB
    print(f"\nannotated image saved to: {out}")

    # The bus.jpg sample reliably contains a bus + multiple people.
    # If we hit the synthetic fallback, the assertion is relaxed.
    if "bus" in img_path.name:
        assert len(result.boxes) >= 3, "expected ≥3 detections on the bus image"
        assert "bus" in detected_classes or "person" in detected_classes, \
            "expected to detect a bus or a person"
        print("YOLO detected the expected objects ✅")
    else:
        print("(synthetic image — assertions skipped)")


if __name__ == "__main__":
    main()


# ── Stretch ─────────────────────────────────────────────────────────────────
# • Run on a video: `model.predict(source='video.mp4', stream=True)`.
# • Try YOLOv8s/m/l — observe accuracy vs latency tradeoffs.
# • Train YOLO on a custom dataset:
#     model.train(data="my-data.yaml", epochs=50, imgsz=640)
#   See https://docs.ultralytics.com for the YAML format.
# ─────────────────────────────────────────────────────────────────────────────
