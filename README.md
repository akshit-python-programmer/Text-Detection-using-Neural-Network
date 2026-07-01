# Handwritten Digit Recognition — Neural Network

A convolutional neural network (CNN) that recognises handwritten digits (0–9), with a live webcam demo that classifies digits in real time.

> **A note on when this was made.** I built this project in **2021, when I was in class 9** — it was one of my first proper machine-learning projects. I'm now sharing it publicly, years later, exactly as a record of where I started. The original scripts are preserved untouched in the repository; I've only *added* an updated, runnable version alongside them so anyone who clones this today can actually run it (the deep-learning libraries have changed a lot since 2021). You can see the original 2021 file timestamps in [`docs/date-proof-2021.png`](docs/date-proof-2021.png).

---

## What it does

The model takes an image of a single handwritten digit and predicts which digit it is. The pipeline:

1. **Loads** ~10,160 digit images (1,016 per class, digits 0–9) from `myData/`.
2. **Pre-processes** each image — resize to 32×32, convert to grayscale, apply histogram equalisation, and normalise pixel values to the 0–1 range.
3. **Augments** the training data on the fly (small shifts, zoom, shear and rotation) so the model generalises better.
4. **Trains** a CNN to classify the digit.
5. **Runs live inference** from a webcam (`test.py`) — point a digit at the camera and it shows the predicted number and confidence on screen.

## Technical details (in short)

The network is a compact VGG-style CNN built with Keras/TensorFlow:

| Stage | Layers |
|-------|--------|
| Feature extraction | `Conv2D(60, 5×5)` → `Conv2D(60, 5×5)` → `MaxPool` → `Conv2D(30, 3×3)` → `Conv2D(30, 3×3)` → `MaxPool` → `Dropout(0.5)` |
| Classifier | `Flatten` → `Dense(500, relu)` → `Dropout(0.5)` → `Dense(10, softmax)` |

- **Optimiser:** Adam (learning rate 0.001)
- **Loss:** categorical cross-entropy
- **Input:** 32×32 grayscale, single channel
- **Output:** probability distribution over 10 digit classes
- **Regularisation:** dropout + real-time image augmentation
- **Train / validation / test split:** 64% / 16% / 20%

A trained model is included (`model.h5`) so you can run the demo without training from scratch.

## Project structure

```
.
├── train.py                 # ORIGINAL training script (2021, kept as-is)
├── test.py                  # ORIGINAL webcam inference script (2021, kept as-is)
├── model.h5                 # Pre-trained model weights
├── 10.png                   # A sample test image
├── myData/                  # Dataset: folders 0–9, 1,016 images each
├── modern/
│   ├── train.py             # Updated training script (runs on current TensorFlow)
│   └── test.py              # Updated webcam inference (runs on current TensorFlow)
├── notebooks/
│   └── Digit recog.ipynb    # Original exploration notebook
├── docs/
│   └── date-proof-2021.png  # Screenshot showing original 2021 file dates
├── archive/                 # Older / duplicate files kept for history
│   ├── t.py                 # Earlier duplicate of the training script
│   ├── model_trained.p      # Pickled model (legacy format)
│   └── model_trained_10.p   # Pickled model (legacy format)
├── requirements.txt
└── README.md
```

## Installation & usage

**1. Clone the repository**

```bash
git clone https://github.com/akshit-python-programmer/Text-Detection-using-Neural-Network.git
cd Text-Detection-using-Neural-Network
```

**2. Install dependencies** (Python 3.10+ recommended)

```bash
pip install -r requirements.txt
```

**3. Run the live webcam demo** (uses the included pre-trained model)

```bash
python modern/test.py
```

Hold a handwritten digit up to your webcam. The predicted digit and its confidence appear on the video feed. Press **ESC** to quit.

**4. (Optional) Train the model yourself**

```bash
python modern/train.py
```

This retrains the CNN on the `myData/` dataset and saves the result to `model.keras`, along with accuracy/loss plots.

> **Tip:** Run the scripts from the project root so the `myData` path and model files resolve correctly.

> The scripts at the repo root (`train.py`, `test.py`) are the original 2021 versions and target older Keras/TensorFlow — they're preserved for authenticity. Use the `modern/` versions to actually run the project today.

## Future scope

A few directions I'd take this if I revisited it:

- **Extend beyond digits** to full handwritten characters (letters A–Z) so it becomes general handwritten-text recognition.
- **Multi-digit / word detection** — segment a line of handwriting and read each character, instead of one digit at a time.
- **A better demo** — a small web app (or a draw-on-canvas interface) so people can try it without a webcam.
- **Model improvements** — batch normalisation, a learning-rate schedule, and more training epochs to push accuracy higher.
- **Deployment** — export to TensorFlow Lite to run on a phone or a microcontroller.

## Dataset

The dataset (`myData/`) contains 10,160 grayscale images of handwritten digits, organised into ten folders (`0`–`9`) with 1,016 images each.

## Author

**Akshit Kumar Lall** — [@akshit-python-programmer](https://github.com/akshit-python-programmer)

Originally built in 2021 (class 9) and shared publicly as part of my project portfolio.
