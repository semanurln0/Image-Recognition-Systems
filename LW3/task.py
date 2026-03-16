import json
from pathlib import Path
import shutil

import cv2
import numpy as np


# Input / output paths (relative to this script file)
BASE_DIR = Path(__file__).resolve().parent
IMAGE_PATH = BASE_DIR / "image.png"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Clean start: remove old output files/folders
for item in OUTPUT_DIR.iterdir():
    if item.is_file() or item.is_symlink():
        item.unlink()
    elif item.is_dir():
        shutil.rmtree(item)


# 1) Load image
image = cv2.imread(str(IMAGE_PATH))
if image is None:
    raise FileNotFoundError(f"Could not read {IMAGE_PATH}")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# 2) Color spectrogram analysis (simple channel histograms + threshold)
hist_image = np.zeros((300, 256, 3), dtype=np.uint8)
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # B, G, R for drawing

for channel, color in enumerate(colors):
    hist = cv2.calcHist([image], [channel], None, [256], [0, 256]).flatten()
    hist = hist / hist.max() * 299
    for x in range(1, 256):
        y1 = 299 - int(hist[x - 1])
        y2 = 299 - int(hist[x])
        cv2.line(hist_image, (x - 1, y1), (x, y2), color, 1)

cv2.imwrite(str(OUTPUT_DIR / "01_color_spectrogram.png"), hist_image)

# Threshold for text/background separation
otsu_value, text_mask = cv2.threshold(
    gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)

# Small cleanup so text stays intact
kernel = np.ones((3, 3), np.uint8)
text_mask = cv2.morphologyEx(text_mask, cv2.MORPH_OPEN, kernel, iterations=1)
text_mask = cv2.morphologyEx(text_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
cv2.imwrite(str(OUTPUT_DIR / "02_text_mask.png"), text_mask)


# 3) Background removal
text_only_background_removed = cv2.bitwise_and(image, image, mask=text_mask)
cv2.imwrite(str(OUTPUT_DIR / "03_background_removed.png"), text_only_background_removed)


# 4) Edge detection
edges = cv2.Canny(gray, 50, 150)
cv2.imwrite(str(OUTPUT_DIR / "04_edges.png"), edges)


# 5) Segmentation mask from text + edges
edge_dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
segmentation_mask = cv2.bitwise_or(text_mask, edge_dilated)
segmentation_mask = cv2.morphologyEx(segmentation_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
cv2.imwrite(str(OUTPUT_DIR / "05_segmentation_mask.png"), segmentation_mask)


# 6) Apply segmentation mask -> two required image variations
# A) only text pixels
only_text_pixels = cv2.bitwise_and(image, image, mask=segmentation_mask)
cv2.imwrite(str(OUTPUT_DIR / "06A_only_text_pixels.png"), only_text_pixels)

# B) text + background with text highlighted
highlighted = image.copy()
highlighted[segmentation_mask > 0] = (255, 0, 0)  # red highlight in BGR
highlighted = cv2.addWeighted(highlighted, 0.35, image, 0.65, 0)
cv2.imwrite(str(OUTPUT_DIR / "06B_text_highlighted.png"), highlighted)


# 7) Combined results image with titles
tile_w, tile_h, title_h = 420, 260, 36
tiles = []
panels = [
    ("Original", image),
    ("Color Spectrogram", hist_image),
    ("Text Mask", text_mask),
    ("Background Removed", text_only_background_removed),
    ("Edges", edges),
    ("Segmentation Mask", segmentation_mask),
    ("Only Text Pixels", only_text_pixels),
    ("Text Highlighted", highlighted),
]

for title, panel in panels:
    if panel.ndim == 2:
        panel = cv2.cvtColor(panel, cv2.COLOR_GRAY2BGR)
    panel = cv2.resize(panel, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
    tile = np.full((tile_h + title_h, tile_w, 3), 245, dtype=np.uint8)
    tile[title_h:, :] = panel
    cv2.putText(tile, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 1, cv2.LINE_AA)
    tiles.append(tile)

top_row = np.hstack(tiles[:4])
bottom_row = np.hstack(tiles[4:8])
combined_results = np.vstack([top_row, bottom_row])
cv2.imwrite(str(OUTPUT_DIR / "07_combined_results_titled.png"), combined_results)


# Save simple threshold report
report = {
    "threshold_method": "Otsu on grayscale (inverse binary)",
    "otsu_threshold_value": float(otsu_value),
    "text_pixels": int(np.count_nonzero(segmentation_mask)),
    "background_pixels": int(segmentation_mask.size - np.count_nonzero(segmentation_mask)),
}

with (OUTPUT_DIR / "threshold_report.json").open("w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print("Done.")
print(f"Otsu threshold: {otsu_value:.2f}")
print(f"Results saved in: {OUTPUT_DIR.resolve()}")
