---
title: "I Found the Neural Network I Built in Class 9 — Here's What Happened When I Tried to Run It Again"
published: false
description: "Four years ago I trained a CNN to read handwritten digits from my webcam. This week I tried to run it again — and none of my code worked anymore. Here's the project, the migration, and what I learned."
tags: machinelearning, python, deeplearning, beginners
cover_image: ""
---

# I Found the Neural Network I Built in Class 9 — Here's What Happened When I Tried to Run It Again

Four years ago, in class 9, I built a handwritten digit recognition system. It used a convolutional neural network to look at a digit through my webcam and tell me, in real time, which number it was.

At 14, that felt like actual magic. I didn't fully understand what a convolution was. I followed along, broke things, googled error messages, fixed them, and somehow ended up with a `model.h5` file that worked.

This week I found that project sitting in an old folder — and decided to clean it up and put it on GitHub. But when I tried to run my own code, almost nothing worked. The libraries had moved on without me. So this post is two things: a walkthrough of the project, and a small tour of everything that changed in the Python deep-learning world between 2021 and now.

## What the project does

The idea is simple:

1. Load ~10,000 images of handwritten digits (1,016 images each for digits 0–9).
2. Preprocess them — resize to 32×32, convert to grayscale, equalize the histogram, and normalize pixel values to 0–1.
3. Augment the training data on the fly (small shifts, zoom, shear, rotation) so the model doesn't just memorize.
4. Train a CNN to classify the digit.
5. Run live inference from a webcam.

## The model

It's a compact VGG-style CNN. Nothing fancy — two convolution blocks, dropout for regularization, and a dense classifier on top:

```python
model = Sequential([
    Conv2D(60, (5, 5), input_shape=(32, 32, 1), activation="relu"),
    Conv2D(60, (5, 5), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(30, (3, 3), activation="relu"),
    Conv2D(30, (3, 3), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.5),
    Flatten(),
    Dense(500, activation="relu"),
    Dropout(0.5),
    Dense(10, activation="softmax"),
])
model.compile(Adam(learning_rate=0.001),
              loss="categorical_crossentropy",
              metrics=["accuracy"])
```

Adam optimizer, categorical cross-entropy, softmax over 10 classes. The preprocessing step is the quiet hero — histogram equalization makes the model much more robust to lighting than raw pixels would be:

```python
def pre_processing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255.0
    return img
```

## The interesting part: my 2021 code wouldn't run

Here's what actually happened when I tried to run my old scripts. If you're maintaining or reviving any older Keras/TensorFlow code, this is the useful section to bookmark.

**1. The import paths moved.** In 2021 I wrote:

```python
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils.np_utils import to_categorical
```

Both of those paths are gone. Everything now lives under `tensorflow.keras`:

```python
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
```

**2. `Adam(lr=0.001)` broke.** The `lr` argument was renamed:

```python
# then
Adam(lr=0.001)
# now
Adam(learning_rate=0.001)
```

**3. `model.fit_generator()` was removed entirely.** Back then, training on an augmented data generator meant a separate method. Now `fit()` accepts generators directly:

```python
# then
model.fit_generator(dataGen.flow(X_train, y_train, batch_size=50), ...)
# now
model.fit(dataGen.flow(X_train, y_train, batch_size=50), ...)
```

**4. `model.predict_classes()` — the one that really got me.** My entire webcam demo relied on this one line, and it no longer exists:

```python
# then
classIndex = int(model.predict_classes(img))
# now
classIndex = int(np.argmax(model.predict(img), axis=1)[0])
```

**5. And a bug 14-year-old me never noticed.** My "press ESC to quit" line used a logical `and` instead of a bitwise `&`:

```python
# then (never actually worked)
if cv2.waitKey(1) and 0xFF == 27:
# now
if cv2.waitKey(1) & 0xFF == 27:
```

The original never let you quit with ESC. I'd been force-closing the window for weeks and never realized why.

## How I handled the old code

I didn't want to erase what I originally wrote — that felt like editing my own history. So in the repo I kept the original 2021 scripts exactly as they were, and added a `modern/` folder with updated versions that run on current TensorFlow. The README explains which is which. If you clone it today, `python modern/test.py` just works using the included pre-trained model.

## What I'd do differently now

Looking back, a few things stand out:

- The training loop used a fixed `steps_per_epoch` far larger than the dataset, so each "epoch" cycled the data many times over — it worked, but it was misleading.
- There was no fixed random seed, so results weren't reproducible.
- The whole model got pickled in one version, which is fragile across library versions. The `.h5` / `.keras` formats are the right call.

If I revisited it seriously, I'd add batch normalization, a learning-rate schedule, and extend it beyond digits to full handwritten characters — and probably wrap it in a small draw-on-canvas web demo so people can try it without a webcam.

## The takeaway

The technical lesson is that ML code rots fast — four years is enough to break almost every line. Pin your dependencies.

But the bigger lesson is personal. This wasn't good code. It was a 14-year-old following a tutorial and stumbling toward something that worked. And that's exactly why it mattered: I finished it. You learn more from one small project you actually complete than from ten you only watch.

If you're a student sitting on an unfinished idea — build the small version. Future you will be glad the file exists.

*The full project, including the original and modernized code, is on GitHub: [link].*
