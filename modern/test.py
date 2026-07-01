"""
Handwritten Digit Recognition — Live webcam inference (modernized).

Updated, runnable version of the original `test.py`. Fixes:
  - loads the modern `.keras` model (falls back to legacy model.h5)
  - `model.predict_classes()` was removed in newer Keras
    ->  `np.argmax(model.predict(...), axis=1)`
  - fixed the ESC-to-quit bug: `cv2.waitKey(1) & 0xFF == 27`
    (the original used `and`, so quitting never worked)

Run:  python modern/test.py
Press ESC to quit. Needs a webcam.
"""

import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# ----------------------------- parameters --------------------------------- #
WIDTH = 640
HEIGHT = 480
THRESHOLD = 0.55          # minimum probability to display a prediction
CAMERA_NO = 0
# -------------------------------------------------------------------------- #

# Prefer the modern model file, fall back to the original .h5
MODEL_PATH = "model.keras" if os.path.exists("model.keras") else "model.h5"
print(f"Loading model: {MODEL_PATH}")
model = load_model(MODEL_PATH)

cap = cv2.VideoCapture(CAMERA_NO)
cap.set(3, WIDTH)
cap.set(4, HEIGHT)


def pre_processing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255.0
    return img


while True:
    success, img_original = cap.read()
    if not success:
        print("Could not read from camera.")
        break

    img = np.asarray(img_original)
    img = cv2.resize(img, (32, 32))
    img = pre_processing(img)
    cv2.imshow("Processed Image", img)
    img = img.reshape(1, 32, 32, 1)

    predictions = model.predict(img, verbose=0)
    class_index = int(np.argmax(predictions, axis=1)[0])
    prob_val = float(np.amax(predictions))
    print(class_index, prob_val)

    if prob_val > THRESHOLD:
        cv2.putText(img_original, f"{class_index}   {prob_val:.2f}",
                    (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)

    cv2.imshow("Original Image", img_original)
    if cv2.waitKey(1) & 0xFF == 27:   # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
