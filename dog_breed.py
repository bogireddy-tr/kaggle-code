import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical
from keras.models import  Sequential
from keras.layers.core import  Lambda , Dense, Flatten, Dropout
from keras.callbacks import EarlyStopping
from keras.layers import BatchNormalization, Conv2D , MaxPooling2D, Activation
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras import backend as K
from sklearn.preprocessing import LabelBinarizer

if K.image_data_format() == 'channels_first':
        input_shape = (3, 64, 64)
else:
        input_shape = (64, 64, 3)

train_path = pd.read_csv("/home/bogireddy/bogireddy/dog_breed/labels.csv")

train_paths = train_path.ix[:, 0].values
train_label = train_path.ix[:, 1].values

train_imgs = []
for i in range(len(train_paths)):
	string = "/home/bogireddy/bogireddy/dog_breed/train/" + train_paths[i] + ".jpg"
	imgs = cv2.imread(string, 3)
	imgs = cv2.resize(imgs, (64, 64))
	train_imgs.append(imgs)
train_imgs = np.asarray(train_imgs)
print "train"

test_paths = os.listdir('/home/bogireddy/bogireddy/dog_breed/test/')
names = []
for i in range(len(test_paths)):
	names.append(test_paths[i][:-4])
print names
test_imgs = []
for j in range(len(test_paths)):
	string = "/home/bogireddy/bogireddy/dog_breed/test/" + test_paths[j]
	imgs = cv2.imread(string, 3)
	imgs = cv2.resize(imgs, (64, 64))
	test_imgs.append(imgs)
test_imgs = np.asarray(test_imgs)
print "test"
mean_train = (train_imgs.mean()).astype('float32')
std_train = (train_imgs.std()).astype('float32')
mean_test = (test_imgs.mean()).astype('float32')
std_test = (test_imgs.std()).astype('float32')

train = (train_imgs - mean_train)/std_train
test = (test_imgs - mean_test)/std_test

encoder = LabelBinarizer()
labels = encoder.fit_transform(train_label)

model = Sequential()
model.add(Conv2D(32, 5, 5, border_mode = 'same', input_shape = input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (5, 5)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(512, kernel_initializer='normal', activation='relu'))
model.add(Dense(120, kernel_initializer='normal', activation = 'softmax'))
model.compile(loss='categorical_crossentropy', optimizer='RMSprop',metrics=['accuracy'])
filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
store = model.fit(train, labels, validation_data = (train, labels), batch_size = 20, epochs = 100, shuffle = True, callbacks = callbacks_list, verbose = 1)


#model.load_weights("/home/bogireddy/bogireddy/dog_breed/weights-improvement-43-1.00.hdf5")
out = model.predict(test)
submit = pd.DataFrame(out)

submit.insert(0, 'id', names)
submission = submit
submission.to_csv('new_submission.csv', index=False)
