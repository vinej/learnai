# Module 3.2 — Computer Vision

**Level:** 3 — Advanced
**Estimated time:** 2-3 weeks

## Goal
Build and fine-tune image models for classification, detection, and segmentation.

## Topics
### CNNs
- Convolutions: kernels, stride, padding, dilation
- Pooling: max, average, global
- Receptive field intuition
- Classic architectures: LeNet, AlexNet, VGG, **ResNet**, **EfficientNet**
- Modern: Vision Transformer (ViT), ConvNeXt

### Transfer learning
- Pretrained models from `torchvision.models` and `timm`
- Feature extraction vs fine-tuning
- Freezing layers, differential learning rates

### Data
- `torchvision.transforms`, `albumentations`
- Augmentation: flips, crops, color jitter, mixup, cutmix
- Building custom `Dataset` classes for images

### Beyond classification
- Object detection: anchors, NMS, IoU; **YOLO**, Faster R-CNN
- Semantic & instance segmentation: U-Net, Mask R-CNN
- Image embeddings (CLIP) for retrieval

## Exercises
1. Train a CNN from scratch on CIFAR-10; reach >85% test accuracy.
2. Fine-tune a pretrained ResNet on a custom dataset of your own photos.
3. Run YOLOv8 inference on a video file; draw bounding boxes.
4. Use CLIP embeddings to build a "find similar images" search.
5. Train a U-Net for binary segmentation on a small medical or satellite dataset.

## Resources
- `timm` library: https://github.com/huggingface/pytorch-image-models
- `albumentations`: https://albumentations.ai/
- Ultralytics YOLO docs: https://docs.ultralytics.com/
- Course: Stanford CS231n (lecture videos free)

## Checkpoint
You can fine-tune a pretrained vision model on a new dataset, and choose appropriate augmentation and architecture for the problem at hand.
