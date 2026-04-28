# Setup — Module 3.2 (Computer Vision)

This module builds CNNs, fine-tunes pretrained models, runs YOLO, and uses image embeddings for retrieval.

## 1. Python ≥ 3.11 + PyTorch

If you completed Module 3.1, you already have this. Otherwise see [../module-3.1-deep-learning-foundations/INSTALL.md](../module-3.1-deep-learning-foundations/INSTALL.md).

## 2. Create / activate the venv

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1     # Windows PS
# or
source .venv/bin/activate       # macOS/Linux
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

| Package           | Purpose                                          |
|-------------------|--------------------------------------------------|
| `torch`, `torchvision` | The framework                              |
| `timm`            | Big catalog of pretrained image models           |
| `ultralytics`     | YOLOv8 — easy inference + training               |
| `albumentations`  | Fast, expressive image augmentation              |
| `Pillow`          | Image I/O                                        |
| `matplotlib`      | Visualization                                    |

### Optional: CLIP for text↔image search (stretch goal)

```bash
pip install open-clip-torch
```

## 4. Run the lessons

```bash
python 01_convolutions.py
python 02_pooling_and_cnn.py
python 03_classic_architectures.py
python 04_transfer_learning.py
python 05_augmentation.py
python 06_object_detection.py        # downloads YOLOv8n weights ~6MB on first run
python 07_segmentation.py
python 08_image_embeddings.py
```

Plotting scripts save PNGs to `figures/`.

## 5. Run the exercises

```bash
python exercises/01_cifar_cnn_from_scratch.py    # downloads CIFAR-10 ~170MB
python exercises/02_fine_tune_resnet.py
python exercises/03_yolo_inference.py
python exercises/04_image_search.py
python exercises/05_unet_segmentation.py
```

## Compute notes

- **CPU is fine** for everything in this module. Times in parentheses below are *rough* CPU estimates — much faster on GPU.
- CIFAR-10 small-CNN training: ~3-5 minutes for the assert-passing run.
- ResNet18 fine-tune on a small subset: ~2-3 minutes.
- YOLO inference on a single image: ~1-2 seconds (after first-run weight download).
- U-Net synthetic training: ~1-2 minutes.
- Image embedding search: ~10 seconds.

## Tip

The vision world has matured: for ANY new image task, your first move is "is there a pretrained model that already does most of this?". Module 3.2 is structured to build that instinct.
