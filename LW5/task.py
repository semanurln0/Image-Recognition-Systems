# pip install numpy tensorflow tensorflow-datasets matplotlib

import os
from pathlib import Path
import json
import time
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

# Set TensorFlow C++ log level before importing tensorflow.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["ABSL_LOG_SEVERITY_THRESHOLD"] = "3"
os.environ["GLOG_minloglevel"] = "3"

import tensorflow as tf
import tensorflow_datasets as tfds

# Suppress Python-side TensorFlow logging
tf.get_logger().setLevel("ERROR")

# Paths and hyperparameters
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

IMG_SIZE = 160
BATCH_SIZE = 16
TRAIN_SAMPLES = 800
VAL_SAMPLES = 200
TEST_SAMPLES = 128
BUFFER_SIZE = 1000
EPOCHS_STAGE1 = 20
EPOCHS_STAGE2 = 20
TARGET_ACCURACY = 0.97
FINETUNE_TOP_LAYERS = 36

SEED = 42
tf.keras.utils.set_random_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except (AttributeError, ValueError):
    pass


def load_oxford_iiit_pet():
    """Load Oxford-IIIT Pet with a train/val/test subset split."""
    (train_raw, val_raw, test_raw), info = tfds.load(
        "oxford_iiit_pet",
        split=[
            f"train[:{TRAIN_SAMPLES}]",
            f"train[{TRAIN_SAMPLES}:{TRAIN_SAMPLES + VAL_SAMPLES}]",
            f"test[:{TEST_SAMPLES}]",
        ],
        with_info=True,
        try_gcs=True,
    )
    return train_raw, val_raw, test_raw, info


def preprocess(example):
    image = example["image"]
    mask = example["segmentation_mask"]

    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.cast(image, tf.float32) / 255.0

    mask = tf.image.resize(mask, (IMG_SIZE, IMG_SIZE), method="nearest")
    mask = tf.cast(mask, tf.int32) - 1
    mask = tf.squeeze(mask, axis=-1)
    mask = tf.clip_by_value(mask, 0, 2)

    return image, mask


def prepare_dataset(ds, shuffle=False):
    ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.cache()
    if shuffle:
        ds = ds.shuffle(BUFFER_SIZE, reshuffle_each_iteration=True, seed=SEED)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def build_unet_xception(img_size=IMG_SIZE, num_classes=3):
    inputs = tf.keras.Input(shape=(img_size, img_size, 3))

    base_model = tf.keras.applications.Xception(
        weights="imagenet",
        include_top=False,
        input_tensor=inputs,
    )
    base_model.trainable = False

    skip_names = [
        "block1_conv2_act",
        "block3_sepconv2_bn",
        "block4_sepconv2_bn",
        "block13_sepconv2_bn",
    ]
    skips = [base_model.get_layer(name).output for name in skip_names]
    x = base_model.get_layer("block14_sepconv2_act").output

    decoder_filters = [512, 256, 128, 64]
    for filters, skip in zip(decoder_filters, reversed(skips)):
        x = tf.keras.layers.Conv2DTranspose(filters, 3, strides=2, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        skip_resized = tf.keras.layers.Lambda(
            lambda tensors: tf.image.resize(
                tensors[0],
                size=tf.shape(tensors[1])[1:3],
                method="bilinear",
            ),
        )([skip, x])

        x = tf.keras.layers.Concatenate()([x, skip_resized])
        x = tf.keras.layers.Conv2D(filters, 3, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv2D(filters, 3, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2DTranspose(32, 3, strides=2, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    outputs = tf.keras.layers.Conv2D(num_classes, 1, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="U_Net_Xception")
    return model


class TargetAccuracyCallback(tf.keras.callbacks.Callback):
    def __init__(self, target):
        super().__init__()
        self.target = target

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_acc = logs.get("val_accuracy")
        if val_acc is not None and val_acc >= self.target:
            print(f"\nReached target val_accuracy >= {self.target:.2f}. Stopping training.")
            self.model.stop_training = True


def create_callbacks(stage_name="train"):
    checkpoint_path = OUTPUT_DIR / "best_model.keras"
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=4,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy",
            mode="max",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
        TargetAccuracyCallback(TARGET_ACCURACY),
        tf.keras.callbacks.CSVLogger(str(OUTPUT_DIR / f"training_{stage_name}.csv"), append=False),
    ]


def decode_masks(mask_batch):
    return mask_batch


def visualize_batch(images, true_masks, pred_masks=None, max_samples=3):
    images = images[:max_samples]
    true_masks = true_masks[:max_samples]
    if pred_masks is not None:
        pred_masks = pred_masks[:max_samples]

    n_rows = 3 if pred_masks is not None else 2
    plt.figure(figsize=(6 * max_samples, 4 * n_rows))

    true_binary = decode_masks(true_masks)
    pred_binary = decode_masks(pred_masks) if pred_masks is not None else None

    for i in range(len(images)):
        plt.subplot(n_rows, max_samples, i + 1)
        plt.imshow(images[i])
        plt.axis("off")
        plt.title("Input")

        plt.subplot(n_rows, max_samples, max_samples + i + 1)
        plt.imshow(true_binary[i], cmap="gray", vmin=0, vmax=2)
        plt.axis("off")
        plt.title("True mask")

        if pred_binary is not None:
            plt.subplot(n_rows, max_samples, 2 * max_samples + i + 1)
            plt.imshow(pred_binary[i], cmap="gray", vmin=0, vmax=2)
            plt.axis("off")
            plt.title("Predicted mask")

    plt.tight_layout()
    plt_path = OUTPUT_DIR / "sample_predictions.png"
    plt.savefig(plt_path)
    plt.close()
    print(f"Saved visualization to {plt_path}")


def save_all_predictions(model, dataset):
    idx = 0
    for images, masks in dataset:
        probs = model.predict(images, verbose=0)
        preds = np.argmax(probs, axis=-1).astype(np.uint8)
        for i in range(images.shape[0]):
            result_path = OUTPUT_DIR / f"result_{idx:03d}.png"
            plt.figure(figsize=(9, 3))
            plt.subplot(1, 3, 1)
            plt.imshow(images[i])
            plt.axis("off")
            plt.title("Input")

            plt.subplot(1, 3, 2)
            plt.imshow(masks[i], cmap="gray", vmin=0, vmax=2)
            plt.axis("off")
            plt.title("True mask")

            plt.subplot(1, 3, 3)
            plt.imshow(preds[i], cmap="gray", vmin=0, vmax=2)
            plt.axis("off")
            plt.title("Predicted mask")

            plt.tight_layout()
            plt.savefig(result_path)
            plt.close()
            idx += 1


def clear_output_dir():
    """Remove all files and subdirectories from output directory (except results.json if it exists)."""
    if OUTPUT_DIR.exists():
        for item in OUTPUT_DIR.glob("*"):
            if item.is_file() and item.name != "results.json":
                item.unlink()
            elif item.is_dir():
                import shutil
                shutil.rmtree(item, ignore_errors=True)


def compile_model(model, learning_rate):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )


def train_until_target(model, train_ds, val_ds, results):
    print("Stage 1: training decoder with frozen Xception encoder...")
    compile_model(model, learning_rate=1e-3)
    
    stage1_start = time.time()
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_STAGE1,
        callbacks=create_callbacks("stage1"),
        verbose=1,
    )
    stage1_time = time.time() - stage1_start
    results["training"]["stage1_time_seconds"] = round(stage1_time, 2)

    val_start = time.time()
    val_metrics = model.evaluate(val_ds, verbose=0)
    val_time = time.time() - val_start
    results["validation"]["stage1_time_seconds"] = round(val_time, 2)
    
    metric_map = dict(zip(model.metrics_names, val_metrics))
    # Try 'accuracy' first, then 'compile_metrics', then use second value (index 1) as fallback
    current_val_acc = float(metric_map.get("accuracy") or metric_map.get("compile_metrics") or (val_metrics[1] if len(val_metrics) > 1 else 0.0))
    results["validation"]["stage1_accuracy"] = round(current_val_acc, 4)
    print(f"Validation accuracy after stage 1: {current_val_acc:.4f}")

    if current_val_acc >= TARGET_ACCURACY:
        results["training"]["stopped_after_stage"] = 1
        return

    print("Stage 2: fine-tuning top Xception layers...")
    # Unfreeze only the top Xception encoder blocks (layers with names starting with block11+)
    # Keep earlier blocks frozen to avoid catastrophic forgetting
    for layer in model.layers:
        layer_name = layer.name
        # Unfreeze only top blocks: block11, block12, block13, block14 and decoder layers
        if any(x in layer_name for x in ["block1", "block2", "block3", "block4", "block5", "block6", "block7", "block8", "block9", "block10"]):
            layer.trainable = False
        else:
            layer.trainable = True

    compile_model(model, learning_rate=1e-5)
    
    stage2_start = time.time()
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_STAGE2,
        callbacks=create_callbacks("stage2"),
        verbose=1,
    )
    stage2_time = time.time() - stage2_start
    results["training"]["stage2_time_seconds"] = round(stage2_time, 2)
    results["training"]["stopped_after_stage"] = 2
    
    val_start = time.time()
    val_metrics_stage2 = model.evaluate(val_ds, verbose=0)
    val_time = time.time() - val_start
    results["validation"]["stage2_time_seconds"] = round(val_time, 2)
    metric_map_stage2 = dict(zip(model.metrics_names, val_metrics_stage2))
    # Try 'accuracy' first, then 'compile_metrics', then use second value (index 1) as fallback
    stage2_val_acc = float(metric_map_stage2.get("accuracy") or metric_map_stage2.get("compile_metrics") or (val_metrics_stage2[1] if len(val_metrics_stage2) > 1 else 0.0))
    results["validation"]["stage2_accuracy"] = round(stage2_val_acc, 4)


def main():
    # Initialize timing and results tracking
    run_start_time = time.time()
    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "target_accuracy": TARGET_ACCURACY,
            "train_samples": TRAIN_SAMPLES,
            "val_samples": VAL_SAMPLES,
            "test_samples": TEST_SAMPLES,
            "epochs_stage1": EPOCHS_STAGE1,
            "epochs_stage2": EPOCHS_STAGE2,
            "img_size": IMG_SIZE,
            "batch_size": BATCH_SIZE,
        },
        "dataset": {"loading_time_seconds": 0, "preparation_time_seconds": 0},
        "model": {"build_time_seconds": 0},
        "training": {"stage1_time_seconds": 0, "stage2_time_seconds": 0, "stopped_after_stage": None},
        "validation": {"stage1_accuracy": 0, "stage2_accuracy": 0, "stage1_time_seconds": 0, "stage2_time_seconds": 0},
        "test": {"accuracy": 0, "loss": 0, "evaluation_time_seconds": 0, "prediction_save_time_seconds": 0},
        "total_runtime_seconds": 0,
    }

    clear_output_dir()

    print("Loading Oxford-IIIT Pet dataset via TFDS...")
    dataset_start = time.time()
    train_raw, val_raw, test_raw, info = load_oxford_iiit_pet()
    dataset_load_time = time.time() - dataset_start
    results["dataset"]["loading_time_seconds"] = round(dataset_load_time, 2)
    print("Dataset loaded.")
    print(f"Dataset train split size: {info.splits['train'].num_examples}")
    print(f"Using train/val/test samples: {TRAIN_SAMPLES}/{VAL_SAMPLES}/{TEST_SAMPLES}")

    print("Preparing datasets...")
    prep_start = time.time()
    train_ds = prepare_dataset(train_raw, shuffle=True)
    val_ds = prepare_dataset(val_raw, shuffle=False)
    test_ds = prepare_dataset(test_raw, shuffle=False)
    prep_time = time.time() - prep_start
    results["dataset"]["preparation_time_seconds"] = round(prep_time, 2)

    print("Visualizing a few input images and masks (before training)...")
    sample_images, sample_masks = next(iter(train_ds))
    visualize_batch(sample_images.numpy(), sample_masks.numpy(), pred_masks=None, max_samples=3)

    print("Building U-Net Xception model...")
    model_start = time.time()
    model = build_unet_xception()
    model_build_time = time.time() - model_start
    results["model"]["build_time_seconds"] = round(model_build_time, 2)
    model.summary()

    print(f"Training model (target val_accuracy >= {TARGET_ACCURACY:.2f})...")
    train_until_target(model, train_ds, val_ds, results)

    print("Evaluating on test data...")
    test_eval_start = time.time()
    test_metrics = model.evaluate(test_ds, verbose=2)
    test_eval_time = time.time() - test_eval_start
    results["test"]["evaluation_time_seconds"] = round(test_eval_time, 2)
    
    metrics_dict = dict(zip(model.metrics_names, test_metrics))
    print("Test metrics:", metrics_dict)
    
    # Try 'accuracy' first, then 'compile_metrics', then use second value (index 1) as fallback
    test_accuracy = float(metrics_dict.get("accuracy") or metrics_dict.get("compile_metrics") or (test_metrics[1] if len(test_metrics) > 1 else 0.0))
    test_loss = float(metrics_dict.get("loss", test_metrics[0] if test_metrics else 0.0))
    results["test"]["accuracy"] = round(test_accuracy, 4)
    results["test"]["loss"] = round(test_loss, 4)
    
    if test_accuracy < TARGET_ACCURACY:
        print(
            f"WARNING: test accuracy {test_accuracy:.4f} is below target {TARGET_ACCURACY:.2f}. "
            "Try increasing training samples/epochs if needed."
        )

    print("Saving predictions for all test images...")
    pred_save_start = time.time()
    save_all_predictions(model, test_ds)
    pred_save_time = time.time() - pred_save_start
    results["test"]["prediction_save_time_seconds"] = round(pred_save_time, 2)

    print("Generating visualization for a few samples...")
    test_images, test_masks = next(iter(test_ds))
    pred_probs = model.predict(test_images, verbose=0)
    pred_masks = np.argmax(pred_probs, axis=-1).astype(np.uint8)
    visualize_batch(
        test_images.numpy(),
        test_masks.numpy(),
        pred_masks=pred_masks,
        max_samples=3,
    )

    # Calculate total runtime and save results
    total_time = time.time() - run_start_time
    results["total_runtime_seconds"] = round(total_time, 2)
    
    results_path = OUTPUT_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n" + "="*60)
    print(f"Results saved to {results_path}")
    print(f"Total runtime: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print("="*60)
    print(json.dumps(results, indent=2))

    print("Done.")


if __name__ == "__main__":
    main()