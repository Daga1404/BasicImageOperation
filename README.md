# Basic Image Operation

A two-part computer vision project built with OpenCV and Python. Part 1 performs offline image processing and analysis on a road-sign dataset. Part 2 connects to a live RTSP camera stream and detects geometric shapes in real time. This project is the foundation of a larger system aimed at recognising safety signs automatically.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Part 1 — Road Sign Image Processing (`part1.py`)](#part-1--road-sign-image-processing-part1py)
  - [What it does](#what-it-does)
  - [Configuration](#configuration-part-1)
  - [Running](#running-part-1)
  - [Output](#output)
- [Part 2 — Real-Time Shape Detection (`part2.py`)](#part-2--real-time-shape-detection-part2py)
  - [What it does](#what-it-does-1)
  - [Configuration](#configuration-part-2)
  - [Running](#running-part-2)
  - [View Modes](#view-modes)
  - [Shape Detection Logic](#shape-detection-logic)
- [Roadmap](#roadmap)

---

## Project Structure

```
BasicImageOperation/
├── part1.py          # Offline image processing pipeline
├── part2.py          # Real-time RTSP shape detector
└── README.md
```

> Image files are excluded from the repository. See the [dataset note](#part-1--road-sign-image-processing-part1py) below for how to source them.

---

## Requirements

```bash
pip install opencv-python numpy matplotlib
```

| Package | Purpose |
|---|---|
| `opencv-python` | Image processing and computer vision |
| `numpy` | Array operations and noise generation |
| `matplotlib` | Figure rendering and saving (Part 1 only) |

Python 3.8 or later is recommended.

---

## Part 1 — Road Sign Image Processing (`part1.py`)

### What it does

Runs a four-stage offline analysis pipeline over a set of road-sign images and saves a figure for each stage.

| Stage | Function | Description |
|---|---|---|
| 1 | `parte1_color_gris` | Side-by-side comparison of each image in colour and greyscale |
| 2 | `parte2_binarizacion_gris` | Greyscale thresholding — produces a binary (black/white) image for each input |
| 3 | `parte3_binarizacion_canal` | Single colour-channel extraction and thresholding (default: Red channel) |
| 4 | `parte4_ruido_y_filtros` | Gaussian noise injection at three severity levels followed by four smoothing filters |

Each stage also prints a written analysis of how the technique applies to road-sign detection.

**Dataset note:** The script expects 10 road-sign images (`road5.png`, `road461.png`, etc.) to be in the same directory as `part1.py`. These come from the [Road Sign Detection dataset](https://www.kaggle.com/datasets/andrewmvd/road-sign-detection) on Kaggle. Images are not included in this repository. Any missing image is skipped with a warning rather than crashing.

### Configuration (Part 1)

All tunable parameters are at the top of `part1.py`:

| Variable | Default | Description |
|---|---|---|
| `IMAGE_DIR` | `"."` | Directory containing the input images |
| `IMAGE_NAMES` | *(list)* | Filenames to load |
| `GRAY_THRESHOLD` | `127` | Pixel threshold for greyscale binarisation (0–255) |
| `COLOR_CHANNEL` | `2` | Channel for colour binarisation — `0` = Blue, `1` = Green, `2` = Red |
| `CHANNEL_THRESHOLD` | `100` | Pixel threshold applied to the selected colour channel |
| `NOISE_LEVELS` | `[10, 25, 50]` | Gaussian noise standard deviations used in Stage 4 |

### Running (Part 1)

```bash
python part1.py
```

### Output

Four PNG figures are saved to the working directory:

| File | Contents |
|---|---|
| `parte1_color_gris.png` | Colour vs greyscale grid |
| `parte2_binaria_gris.png` | Colour / greyscale / binary grid |
| `parte3_binaria_canal.png` | Colour / channel / binary grid |
| `parte4_ruido_sigma10.png` | Noise σ=10 with all four filters |
| `parte4_ruido_sigma25.png` | Noise σ=25 with all four filters |
| `parte4_ruido_sigma50.png` | Noise σ=50 with all four filters |

**Smoothing filter summary from Stage 4:**

| Filter | Behaviour | Best use |
|---|---|---|
| Mean 5×5 | Fast, blurs edges | Low-noise scenes |
| Gaussian 5×5 | Weighted blur, softer edge loss | General purpose |
| Median 5 | Removes impulse noise, preserves edges | Low-to-mid noise on signs |
| Bilateral | Preserves sharp edges while smoothing flat areas | High noise, must keep text/digits readable |

---

## Part 2 — Real-Time Shape Detection (`part2.py`)

### What it does

Connects to an RTSP camera stream, reads frames continuously on a background thread, and processes each frame to detect and label geometric shapes. Recognised shapes are annotated directly on the video feed.

The script supports three view modes selectable at startup:

- **Original** — raw colour feed
- **Edge detection** — greyscale frame with Canny edges highlighted in cyan
- **Shape detection** — greyscale frame with all detected shapes outlined and labelled

### Configuration (Part 2)

All tunable parameters are at the top of `part2.py`:

| Variable | Default | Description |
|---|---|---|
| `RTSP_URL` | `"rtsp://192.168.1.9:8554/stream"` | RTSP stream address |
| `CANNY_LOW` | `50` | Lower threshold for Canny edge detection |
| `CANNY_HIGH` | `150` | Upper threshold for Canny edge detection |
| `HOUGH_LINE_THRESHOLD` | `80` | Minimum votes for a line to be kept by HoughLinesP |
| `HOUGH_LINE_MIN_LEN` | `60` | Minimum line segment length in pixels |
| `HOUGH_LINE_MAX_GAP` | `10` | Maximum gap in pixels between collinear segments |
| `POLY_AREA_MIN` | `500` | Minimum contour area (px²) — filters out noise |
| `CIRCULARITY_THRESH` | `0.72` | Minimum circularity score (`4π·area/perimeter²`) for a round shape |
| `ELLIPSE_AXIS_RATIO` | `0.85` | Minimum minor/major axis ratio to label a round shape as "Circle" vs "Ellipse" |
| `POLYGON_MAX_CORNER_ANGLE` | `160` | Reserved for future use in corner-angle disambiguation |

**Annotation colours (BGR):**

| Constant | Colour | Used for |
|---|---|---|
| `COLOR_LINE` | Red | Detected straight lines |
| `COLOR_CIRCLE` | Blue | Circles and ellipses |
| `COLOR_POLY` | Green | Polygons |
| `COLOR_LABEL` | Yellow | All text labels |

### Running (Part 2)

```bash
python part2.py
```

You will be prompted to choose a view mode:

```
Select view mode:
  1 - Regular (original color)
  2 - Edge detection
  3 - Shape detection
Enter 1, 2 or 3:
```

Press **q** to quit.

### View Modes

**Mode 2 — Edge Detection**

Converts each frame to greyscale, applies a Gaussian blur, then runs Canny edge detection. Detected edges are drawn in neon cyan over the greyscale frame.

**Mode 3 — Shape Detection**

Runs a full detection pipeline on each frame:

1. **Lines** — detected with `HoughLinesP` and drawn in red.
2. **Shapes** — all contours are extracted using the full `RETR_TREE` hierarchy (so shapes inside other shapes are all detected).

### Shape Detection Logic

Contour classification uses a two-pass approach:

**Pass 1 — Classification**

Each contour is measured and classified:

1. Contours smaller than `POLY_AREA_MIN` are discarded.
2. **Circularity** is computed: `4π · area / perimeter²` (1.0 = perfect circle).
3. Two polygon approximations are computed:
   - *Coarse* (`epsilon = 0.02 × perimeter`) — used for polygon vertex counting.
   - *Fine* (`epsilon = 0.005 × perimeter`) — used to distinguish circles from high-sided polygons. Real polygon sides are already straight, so their vertex count does not change with finer epsilon. A circle's curved boundary produces significantly more vertices (≈13–16) at the finer tolerance.
4. A contour is classified as **round** if:
   - `circularity ≥ CIRCULARITY_THRESH`, **and**
   - the fine approximation has more than 12 vertices.
5. Round shapes are fitted with `cv2.fitEllipse`. If the minor/major axis ratio exceeds `ELLIPSE_AXIS_RATIO` the label is **Circle**, otherwise **Ellipse**.
6. Non-round contours with 3–12 coarse vertices are labelled by vertex count: Triangle, Quadrilateral, Pentagon, Hexagon, Heptagon, Octagon, Nonagon, Decagon, or N-gon.

**Pass 2 — Deduplication and Drawing**

Because `RETR_TREE` returns every edge in the image, a physical ring (such as a circular sign border) produces two concentric contours — one for each edge of the ring. Both would be labelled "Circle". To avoid duplicate labels:

> A contour is **suppressed** if its parent contour in the `RETR_TREE` hierarchy carries the **same label**.

This removes the duplicate inner-ring label while keeping differently-labelled inner shapes (e.g. a triangle drawn inside a circle is still shown).

**Supported shape labels:**

| Label | Vertices / Condition |
|---|---|
| Circle | Round, axis ratio ≥ 0.85 |
| Ellipse | Round, axis ratio < 0.85 |
| Triangle | 3 vertices |
| Quadrilateral | 4 vertices |
| Pentagon | 5 vertices |
| Hexagon | 6 vertices |
| Heptagon | 7 vertices |
| Octagon | 8 vertices |
| Nonagon | 9 vertices |
| Decagon | 10 vertices |
| N-gon | 11–12 vertices |

---

## Roadmap

- [ ] Combine shape hierarchy data (e.g. triangle-inside-circle) to classify composite safety signs
- [ ] Add sign-type lookup table mapping shape combinations to standard sign meanings
- [ ] Improve robustness under varying lighting and camera angles
- [ ] Replace RTSP stream with support for video files and webcam input
