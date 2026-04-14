# LW5

Lab 5: Advanced Image Segmentation with U-Net Xception.

This task uses the Oxford-IIIT Pet dataset and a Keras model to perform semantic segmentation (per-pixel classification) with three classes: background, pet, and border.

## Files

- `task.pdf` - assignment description
- `task.py` - Keras implementation using TFDS and U-Net Xception
- `output/` - generated visualizations and saved best model (created by the script)

## Requirements

Install the required Python packages in your environment (for example, in a virtualenv):

```bash
pip install tensorflow tensorflow-datasets matplotlib
```

## How to Run

From the `LW5/` folder:

```bash
python task.py
```

The script will:

1. Download the `oxford_iiit_pet` dataset via `tensorflow_datasets`.
2. Take 300 images from the training split for model training.
3. Prepare and normalize images and segmentation masks.
4. Build a U-Net-like decoder on top of a frozen Xception encoder.
5. Train the model, tracking validation loss and saving the best checkpoint to `output/best_model.keras`.
6. Generate and save visualizations (`output/sample_predictions.png`) showing input images, ground-truth masks, and predicted masks.
