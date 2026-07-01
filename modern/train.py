import cv2
import numpy as np
import os
from tqdm import tqdm
import pickle

from sklearn.model_selection import train_test_split
import matplotlib.pyplot  as plt

# updated these imports so it runs on the newer tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout,Flatten
from tensorflow.keras.layers import Conv2D,MaxPooling2D

print('Program Started...')

############################
path = "myData"
test_ratio = 0.2
validation_ratio = 0.2
imageDimensions = (32,32,3)
############################
myList = os.listdir(path)
print(f"Total No of Classes : {len(myList)}")
noOfClasses = len(myList)

images = []
classNo = []

for x in range(0,noOfClasses):
    myPicList = os.listdir(path + "/" + str(x) )
    for y in tqdm(myPicList):
        curImg = cv2.imread(path + "/" + str(x) + "/" + str(y))
        curImg = cv2.resize(curImg , (imageDimensions[0] , imageDimensions[1]))
        images.append(curImg)
        classNo.append(x)
    print(x)

print(len(images))
print(len(classNo))

images = np.array(images)
classNo = np.array(classNo)
print(images.shape)

#### Splitting the Data

X_train , X_test , y_train , y_test = train_test_split(images , classNo , test_size=test_ratio )
X_train , X_validation , y_train , y_validation = train_test_split(X_train , y_train , test_size=validation_ratio)


print(X_train.shape)
print(X_test.shape)
print(X_validation.shape)

numberOfSamples = []
for x in range(0,noOfClasses):
    print( len(np.where(y_train==x)[0]) )
    numberOfSamples.append( len(np.where(y_train==x)[0]) )

print(numberOfSamples)

plt.figure(figsize = (10,5))
plt.bar(range(0,noOfClasses) , numberOfSamples)
plt.title("No of images for each class")
plt.xlabel("class Id")
plt.ylabel("number.of images")
plt.show()

def preProcessing(img):
    img = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img

X_train = np.array(list(map(preProcessing , X_train)))
X_test = np.array(list(map(preProcessing , X_test)))
X_validation = np.array(list(map(preProcessing , X_validation)))

X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2],1)
X_validation = X_validation.reshape(X_validation.shape[0],X_validation.shape[1],X_validation.shape[2],1)

print(X_train.shape)

# img = X_train[30]
# print(img.shape)
# img = cv2.resize(img , (300,300))
# cv2.imshow("preProcessed image" , img)
# cv2.waitKey(0)

#### IMAGE AUGMENTATION
dataGen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2,
                             shear_range=0.1,
                             rotation_range=10)
dataGen.fit(X_train)

y_train = to_categorical(y_train,noOfClasses)
y_test = to_categorical(y_test,noOfClasses)
y_validation = to_categorical(y_validation,noOfClasses)

def myModel():
    noOfFilters = 60
    sizeOfFilter1 = (5,5)
    sizeOfFilter2 = (3, 3)
    sizeOfPool = (2,2)
    noOfNodes= 500

    model = Sequential()
    model.add((Conv2D(noOfFilters,sizeOfFilter1,input_shape=(imageDimensions[0],
                      imageDimensions[1],1),activation='relu')))
    model.add((Conv2D(noOfFilters, sizeOfFilter1, activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add((Conv2D(noOfFilters//2, sizeOfFilter2, activation='relu')))
    model.add((Conv2D(noOfFilters//2, sizeOfFilter2, activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(noOfNodes,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(noOfClasses, activation='softmax'))

    model.compile(Adam(learning_rate=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
    return model

model = myModel()
print(model.summary())

batchSizeVal = 50
epochsVal = 2
stepsPerEpochVal = 2000

history = model.fit(dataGen.flow(X_train,y_train,
                                 batch_size=batchSizeVal),
                                 steps_per_epoch=stepsPerEpochVal,
                                 epochs=epochsVal,
                                 validation_data=(X_validation,y_validation),
                                 shuffle=1)

#### PLOT THE RESULTS
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training','validation'])
plt.title('Loss')
plt.xlabel('epoch')
plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training','validation'])
plt.title('Accuracy')
plt.xlabel('epoch')
plt.show()

#### EVALUATE USING TEST IMAGES
score = model.evaluate(X_test,y_test,verbose=0)
print('Test Score = ',score[0])
print('Test Accuracy =', score[1])

#### SAVE THE TRAINED MODEL
model.save('model.h5')
# pickle_out= open("model_trained.p", "wb")
# pickle.dump(model,pickle_out)
# pickle_out.close()
