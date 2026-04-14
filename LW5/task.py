# pip install numpy tensorflow tensorflow-datasets matplotlib

from pathlib import Path
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

# Reduce TensorFlow logging noise and disable oneDNN warning
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # hide INFO and WARNING
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # turn off oneDNN custom ops

# Paths and hyperparameters
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

IMG_SIZE = 160
BATCH_SIZE = 16
TRAIN_SAMPLES = 300
TEST_SAMPLES = 32
BUFFER_SIZE = 1000
EPOCHS = 25


# 1- loading data
def load_oxford_iiit_pet():
	(train_all, test_all), info = tfds.load(
		"oxford_iiit_pet",
		split=["train", "test"],
		with_info=True,
	)

	train_raw = train_all.take(TRAIN_SAMPLES)
	test_raw = test_all.take(TEST_SAMPLES)

	return train_raw, test_raw, info


# 2- preprocess and dataset pipeline
def preprocess(example):
	image = example["image"]
	mask = example["segmentation_mask"]

	image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
	image = tf.cast(image, tf.float32) / 255.0

	mask = tf.image.resize(mask, (IMG_SIZE, IMG_SIZE), method="nearest")
	mask = tf.cast(mask, tf.int32) - 1
	mask = tf.squeeze(mask, axis=-1)

	return image, mask


def prepare_dataset(ds, shuffle: bool = False):
	ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
	ds = ds.cache()
	if shuffle:
		ds = ds.shuffle(BUFFER_SIZE, reshuffle_each_iteration=True)
	ds = ds.batch(BATCH_SIZE)
	ds = ds.prefetch(tf.data.AUTOTUNE)
	return ds


# 3- model definition (Xception-based U-Net, like example.ipynb)
def build_unet_xception(img_size=IMG_SIZE, num_classes=3):
	inputs = tf.keras.Input(shape=(img_size, img_size, 3))

	base_model = tf.keras.applications.Xception(
		weights="imagenet",
		include_top=False,
		input_tensor=inputs,
	)

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

		# resize skip to match decoder spatial size
		skip_resized = tf.keras.layers.Lambda(
			lambda tensors: tf.image.resize(
					ensors[0],
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


def create_callbacks():
	checkpoint_path = OUTPUT_DIR / "best_model.keras"
	return [
		tf.keras.callbacks.ModelCheckpoint(
			filepath=str(checkpoint_path),
			monitor="val_loss",
			mode="min",
			save_best_only=True,
			verbose=1,
		),
		tf.keras.callbacks.EarlyStopping(
			monitor="val_loss",
			mode="min",
			patience=3,
			restore_best_weights=True,
			verbose=1,
		),
	]


def decode_masks(mask_batch: np.ndarray) -> np.ndarray:
	return mask_batch


def visualize_batch(images, true_masks, pred_masks=None, max_samples: int = 3):
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
		plt.imshow(true_binary[i], cmap="gray")
		plt.axis("off")
		plt.title("True mask")

		if pred_binary is not None:
			plt.subplot(n_rows, max_samples, 2 * max_samples + i + 1)
			plt.imshow(pred_binary[i], cmap="gray")
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
		probs = model.predict(images)
		preds = np.argmax(probs, axis=-1).astype(np.uint8)
		for i in range(images.shape[0]):
			result_path = OUTPUT_DIR / f"result_{idx:03d}.png"
			plt.figure(figsize=(9, 3))
			plt.subplot(1, 3, 1)
			plt.imshow(images[i])
			plt.axis("off")
			plt.title("Input")

			plt.subplot(1, 3, 2)
			plt.imshow(masks[i], cmap="gray")
			plt.axis("off")
			plt.title("True mask")

			plt.subplot(1, 3, 3)
			plt.imshow(preds[i], cmap="gray")
			plt.axis("off")
			plt.title("Predicted mask")

			plt.tight_layout()
			plt.savefig(result_path)
			plt.close()
			idx += 1


def clear_output_dir():
	for f in OUTPUT_DIR.glob("*"):
		if f.is_file():
			f.unlink()


def main():
	clear_output_dir()

	print("Loading Oxford-IIIT Pet dataset via TFDS...")
	train_raw, test_raw, info = load_oxford_iiit_pet()
	print("Dataset loaded.")
	print(f"Train subset size: {TRAIN_SAMPLES}")

	print("Preparing datasets...")
	train_ds = prepare_dataset(train_raw, shuffle=True)
	test_ds = prepare_dataset(test_raw, shuffle=False)

	print("Visualizing a few input images and masks (before training)...")
	sample_images, sample_masks = next(iter(train_ds))
	visualize_batch(sample_images.numpy(), sample_masks.numpy(), pred_masks=None, max_samples=3)

	print("Building U-Net Xception model...")
	model = build_unet_xception()
	model.summary()

	model.compile(
		optimizer="adam",
		loss="sparse_categorical_crossentropy",
		metrics=["accuracy"],
	)

	print("Training model...")
	model.fit(
		train_ds,
		validation_data=test_ds,
		epochs=EPOCHS,
		callbacks=create_callbacks(),
	)

	print("Evaluating on test data...")
	test_metrics = model.evaluate(test_ds, verbose=2)
	print("Test metrics:", dict(zip(model.metrics_names, test_metrics)))

	print("Saving predictions for all test images...")
	save_all_predictions(model, test_ds)

	print("Generating visualization for a few samples...")
	test_images, test_masks = next(iter(test_ds))
	pred_probs = model.predict(test_images)
	pred_masks = np.argmax(pred_probs, axis=-1).astype(np.uint8)
	visualize_batch(
		test_images.numpy(),
		test_masks.numpy(),
		pred_masks=pred_masks,
		max_samples=3,
	)

	print("Done.")


if __name__ == "__main__":
	main()
