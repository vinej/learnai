# Setup — Module 5.4

```bash
pip install torch pytorch-lightning darts neuralforecast
```

GPU strongly recommended for PatchTST and TFT. CPU works for the small
LSTM and N-BEATS examples but is slow.

To verify CUDA:
```python
import torch
print(torch.cuda.is_available(), torch.cuda.device_count())
```

If you don't have a local GPU:
- Reduce `max_epochs`, sequence length, and model dim in each example.
- Use Colab / Modal / Lightning Studios.

The shared loaders from Module 5.1 are reused.
