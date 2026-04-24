# LW5 - Advanced Image Segmentation with U-Net Xception

Lab 5: Semantic segmentation using U-Net with Xception encoder on Oxford-IIIT Pet dataset.

## Task Description

Implement a U-Net-based semantic segmentation model with:
- **Three classes**: background, pet, border
- **Dataset**: Oxford-IIIT Pet (via TensorFlow Datasets)
- **Architecture**: U-Net with frozen Xception encoder (stage 1) → fine-tuned encoder (stage 2)
- **Metrics**: Track accuracy, loss, and training time
- **Output**: Visualization, model checkpoint, and metrics (JSON + CSV)

## Files

- `task.pdf` - assignment description
- `task.py` - Full implementation (850+ lines)
- `run_clean.sh` - Clean launcher (suppresses noisy startup logs)
- `IMPLEMENTATION_REVIEW.md` - Detailed technical review
- `output/` - Results directory (created by script)
  - `results.json` - Comprehensive metrics
  - `training_stage1.csv`, `training_stage2.csv` - Epoch logs
  - `best_model.keras` - Saved model (363MB)
  - `result_*.png` - 128 test predictions
  - `sample_predictions.png` - 3-sample visualization

## Requirements

```bash
pip install tensorflow[and-cuda] tensorflow-datasets matplotlib numpy
```

**Recommended**: TensorFlow 2.17.1+ with GPU support (CUDA 12.x)

## How to Run

### Option 1: With Clean Logs (Recommended)
```bash
cd /home/semanurln0/GPU_Workspace
Image-Recognition-Systems/LW5/run_clean.sh
```
This suppresses noisy TensorFlow GPU initialization messages.

### Option 2: Standard Execution
```bash
cd Image-Recognition-Systems/LW5
python task.py
```

Both produce identical results.

## What the Script Does

1. **Dataset**: Downloads Oxford-IIIT Pet (800 train, 200 val, 128 test)
2. **Preprocessing**: Resizes images to 160×160, normalizes, converts masks to 3-class format
3. **Model**: Builds U-Net with Xception encoder (44.8M parameters)
4. **Stage 1**: Trains decoder with frozen encoder (268s, 20 epochs, 91.13% val acc)
5. **Stage 2**: Fine-tunes top encoder layers (110s, 20 epochs, 91.2% val acc)
6. **Evaluation**: Tests on 128 samples (92.11% accuracy)
7. **Output**: Saves model, visualizations, and metrics (7.3 min total)

## Latest Results

```json
{
  "training": {
    "stage1_time_seconds": 268.58,
    "stage2_time_seconds": 110.53,
    "stopped_after_stage": 2
  },
  "validation": {
    "stage1_accuracy": 0.9113,
    "stage2_accuracy": 0.912
  },
  "test": {
    "accuracy": 0.9211,
    "loss": 0.2138
  },
  "total_runtime_seconds": 436.26
}
```

**GPU**: NVIDIA RTX 4070 (5520 MB VRAM)  
**Current accuracy**: 92.11%)

## Implementation Highlights

- ✅ Two-stage transfer learning (decoder → fine-tune)
- ✅ Proper skip connections with bilinear upsampling
- ✅ Sparse categorical crossentropy loss
- ✅ GPU-accelerated training (TensorFlow 2.17.1 with CUDA)
- ✅ Comprehensive metrics tracking (JSON + CSV)
- ✅ Deterministic training (fixed random seed)
- ✅ Memory-efficient data pipeline (caching, prefetching)
- ✅ Clean output management (no file accumulation)

## Known Limitations & Future Improvements

- Small training set (800 samples) limits performance
- Could benefit from data augmentation
- Longer training or larger model could improve accuracy

See `IMPLEMENTATION_REVIEW.md` for full technical details.
