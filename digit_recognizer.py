import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical
from keras.models import  Sequential
from keras.layers.core import  Lambda , Dense, Flatten, Dropout
from keras.callbacks import EarlyStopping
from keras.layers import BatchNormalization, Conv2D , MaxPooling2D, Activation
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras import backend as K

if K.image_data_format() == 'channels_first':
	input_shape = (1, 28, 28)
else:
	input_shape = (28, 28, 1)

train_path = pd.read_csv("/home/bogireddy/bogireddy/digit_recognizer/train.csv")
test_path = pd.read_csv("/home/bogireddy/bogireddy/digit_recognizer/test.csv")

train_load = (train_path.ix[:, 1:].values).astype('float32')
label_load = (train_path.ix[:, 0].values).astype('float32')
train_load = train_load.reshape(len(train_load), 28, 28, 1)

test_load = (test_path.values).astype('float')
test_load = test_load.reshape(len(test_load), 28, 28, 1)

mean_train = train_load.mean().astype('float32')
std_train = train_load.std().astype('float32')
mean_test = test_load.mean().astype('float32')
std_test = test_load.std().astype('float32')

train = (train_load - mean_train)/std_train
test = (test_load - mean_test)/std_test
label_load = to_categorical(label_load)


model = Sequential()
model.add(Conv2D(28, 3, 3, border_mode = 'same', input_shape = input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (3, 3)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128, kernel_initializer='normal', activation='relu'))
model.add(Dense(10, kernel_initializer='normal', activation = 'softmax'))
model.compile(loss='categorical_crossentropy', optimizer='RMSprop',metrics=['accuracy'])
filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
store = model.fit(train, label_load, validation_data = (train, label_load), batch_size = 45, epochs = 35, shuffle = True, callbacks = callbacks_list, verbose = 1)
history = store.history
loss_values = history['loss']
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, 'bo')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()
model.load_weights("/home/bogireddy/bogireddy/digit_recognizer/weights-improvement-34-1.00.hdf5")
out = model.predict(test)
predict = np.argmax(out, axis = 1)
submit = pd.DataFrame({"ImageId" : list(range(1, len(predict) + 1)), "Label" : predict})
submit.to_csv("submit.csv", index = False, header = True)
