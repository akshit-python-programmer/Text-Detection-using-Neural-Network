"""
Handwritten Digit Recognition — Training script (modernized).

This is an updated, runnable version of the original `train.py` I wrote in
class 9 (2021). The logic and CNN architecture are unchanged; only the parts
that broke on newer TensorFlow/Keras have been fixed:

  - imports moved to `tensorflow.keras.*`
  - `Adam(lr=...)`            ->  `Adam(learning_rate=...)`
  - `model.fit_generator()`  ->  `model.fit()` (accepts generators directly)
  - added a fixed random_state for reproducible splits
  - saves to the modern `.keras` format

Run:  python modern/train.py
(Run it from the project root so the "myData" path resolves.)
"""

import os
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam

# ----------------------------- parameters --------------------------------- #
PATH = "myData"
TEST_RATIO = 0.2
VALIDATION_RATIO = 0.2
IMAGE_DIMENSIONS = (32, 32, 3)
BATCH_SIZE = 50
EPOCHS = 10
SEED = 42
# -------------------------------------------------------------------------- #

print("Program started...")

class_folders = sorted(os.listdir(PATH))
no_of_classes = len(class_folders)
print(f"Total number of classes: {no_of_classes}")

images, class_no = [], []
for class_id in range(no_of_classes):
    pic_list = os.listdir(os.path.join(PATH, str(class_id)))
    for pic_name in tqdm(pic_list, desc=f"class {class_id}"):
        cur_img = cv2.imread(os.path.join(PATH, str(class_id), pic_name))
        cur_img = cv2.resize(cur_img, (IMAGE_DIMENSIONS[0], IMAGE_DIMENSIONS[1]))
        images.append(cur_img)
        class_no.append(class_id)

images = np.array(images)
class_no = np.array(class_no)
print("Dataset shape:", images.shape)

# ----------------------------- split -------------------------------------- #
X_train, X_test, y_train, y_test = train_test_split(
    images, class_no, test_size=TEST_RATIO, random_state=SEED
)
X_train, X_validation, y_train, y_validation = train_test_split(
    X_train, y_train, test_size=VALIDATION_RATIO, random_state=SEED
)
print("Train:", X_train.shape, "Test:", X_test.shape, "Val:", X_validation.shape)

# ------------------------ class distribution plot ------------------------- #
samples_per_class = [int(np.sum(y_train == i)) for i in range(no_of_classes)]
plt.figure(figsize=(10, 5))
plt.bar(range(no_of_classes), samples_per_class)
plt.title("Number of training images per class")
plt.xlabel("Class ID")
plt.ylabel("Number of images")
plt.tight_layout()
plt.savefig("class_distribution.png")
print("Saved class_distribution.png")

# ----------------------------- preprocessing ------------------------------ #
def pre_processing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255.0
    return img

X_train = np.array([pre_processing(i) for i in X_train])
X_test = np.array([pre_processing(i) for i in X_test])
X_validation = np.array([pre_processing(i) for i in X_validation])

X_train = X_train.reshape(*X_train.shape, 1)
X_test = X_test.reshape(*X_test.shape, 1)
X_validation = X_validation.reshape(*X_validation.shape, 1)

# ----------------------------- augmentation ------------------------------- #
data_gen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    shear_range=0.1,
    rotation_range=10,
)
data_gen.fit(X_train)

y_train = to_categorical(y_train, no_of_classes)
y_test = to_categorical(y_test, no_of_classes)
y_validation = to_categorical(y_validation, no_of_classes)

# ------------------------------- model ------------------------------------ #
def build_model():
    no_of_filters = 60
    size_of_filter1 = (5, 5)
    size_of_filter2 = (3, 3)
    size_of_pool = (2, 2)
    no_of_nodes = 500

    model = Sequential([
        Conv2D(no_of_filters, size_of_filter1,
               input_shape=(IMAGE_DIMENSIONS[0], IMAGE_DIMENSIONS[1], 1),
               activation="relu"),
        Conv2D(no_of_filters, size_of_filter1, activation="relu"),
        MaxPooling2D(pool_size=size_of_pool),
        Conv2D(no_of_filters // 2, size_of_filter2, activation="relu"),
        Conv2D(no_of_filters // 2, size_of_filter2, activation="relu"),
        MaxPooling2D(pool_size=size_of_pool),
        Dropout(0.5),
        Flatten(),
        Dense(no_of_nodes, activation="relu"),
        Dropout(0.5),
        Dense(no_of_classes, activation="softmax"),
    ])
    model.compile(Adam(learning_rate=0.001),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model

model = build_model()
model.summary()

# ------------------------------- training --------------------------------- #
history = model.fit(
    data_gen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    steps_per_epoch=len(X_train) // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_validation, y_validation),
    shuffle=True,
)

# ------------------------------- results ---------------------------------- #
plt.figure()
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.legend(["training", "validation"])
plt.title("Loss")
plt.xlabel("epoch")
plt.savefig("loss.png")

plt.figure()
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.legend(["training", "validation"])
plt.title("Accuracy")
plt.xlabel("epoch")
plt.savefig("accuracy.png")

score = model.evaluate(X_test, y_test, verbose=0)
print("Test score    =", score[0])
print("Test accuracy =", score[1])

model.save("model.keras")
print("Saved trained model to model.keras")
