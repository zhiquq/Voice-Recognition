import os
import sys
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import load_model
from keras.utils import np_utils
from speechemotionrecognition.utilities import get_feature_vector_from_mfcc
from examples.common import extract_data
from speechemotionrecognition.dnn import CNN
import numpy as np
import seaborn as sns
from keras import Sequential
from keras.layers import LSTM as KERAS_LSTM, Dense, Dropout, Conv2D, Flatten, \
    BatchNormalization, Activation, MaxPooling2D

import matplotlib.pyplot as plt

num_classes = 4
to_flatten = False
x_train, x_test, y_train, y_test, num_labels = extract_data(flatten=to_flatten)
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
in_shape = x_train[0].shape
x_train = x_train.reshape(x_train.shape[0], in_shape[0], in_shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], in_shape[0], in_shape[1], 1)

print('x_train',x_train.shape)
print('y_train',y_train.shape)
print('x_test',x_test.shape)
print('y_test',y_test.shape)

print(y_test)



# input_shape = x_train[0].shape
#
# model = Sequential()
#
# model.add(Conv2D(8, (13, 13),input_shape=(input_shape[0], input_shape[1], 1)))
# model.add(BatchNormalization(axis=-1))
# model.add(Activation('relu'))
# model.add(Conv2D(8, (13, 13)))
# model.add(BatchNormalization(axis=-1))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 1)))
# model.add(Conv2D(8, (13, 13)))
# model.add(BatchNormalization(axis=-1))
# model.add(Activation('relu'))
# model.add(Conv2D(8, (2, 2)))
# model.add(BatchNormalization(axis=-1))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 1)))
# model.add(Flatten())
# model.add(Dropout(0.4))
# model.add(Dense(64))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(Dropout(0.4))
# model.add(Dense(num_classes, activation='softmax'))
# model.summary()
#
# model.compile(loss='categorical_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])
#
# model.fit(x_train, y_train, batch_size=32, epochs=1)
# history = model.fit(x_train,
#                     y_train,
#                     batch_size=64,
#                     epochs=50,
#                     validation_data=(x_test, y_test))
# model.save('model_1.h5')
#
# print(history.history.keys())
# acc = history.history['acc']
# val_acc = history.history['val_acc']
# loss = history.history['loss']
# val_loss = history.history['val_loss']
#
# plt.subplot(1, 2, 1)
# plt.plot(acc, label='Training Accuracy')
# plt.plot(val_acc, label='Validation Accuracy')
# plt.title('Training and Validation Accuracy')
# plt.legend()
#
# plt.subplot(1, 2, 2)
# plt.plot(loss, label='Training Loss')
# plt.plot(val_loss, label='Validation Loss')
# plt.title('Training and Validation Loss')
# plt.legend()
# plt.show()
# model = load_model('./model_1.h5')
#
# y_pred = model.predict(x_test)
#
# y_p = []
# y_t = []
#
# for i in y_pred:
#     d = np.argmax(i)
#     y_p.append(d)
#
# for i in y_test:
#     d = np.argmax(i)
#     y_t.append(d)
#
#
# print(y_pred)
# print(y_test)
#
# data = confusion_matrix(y_t, y_p)
#
# print(data)
# sns.heatmap(data, annot=True,  fmt='d', cmap="coolwarm", center=12,
#                     annot_kws={'size':10,'weight':'bold','color':'b'})
#
# plt.show()
# print(classification_report(y_t, y_p))

