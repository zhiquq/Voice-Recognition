import os
import sys

from keras.models import load_model
from keras.utils import np_utils
from speechemotionrecognition.utilities import get_feature_vector_from_mfcc
from examples.common import extract_data
from speechemotionrecognition.dnn import CNN
import numpy as np
from keras import Sequential
from keras.layers import LSTM as KERAS_LSTM, Dense, Dropout, Conv2D, Flatten, \
    BatchNormalization, Activation, MaxPooling2D

num_classes = 6
to_flatten = False

model = load_model('./model_1.h5')
model.summary()

dir = r'D:\python_project\speech-emotion-recognitio\speech-emotion-recognition-master\dataset\Sad'
name_list = os.listdir(dir)



for name in name_list:
    if name.split('.')[-1]=='wav':
        path = os.path.join(dir,name)
        sample = get_feature_vector_from_mfcc(path, flatten=to_flatten)
        sample = np.expand_dims(sample, axis=2)
        result = model.predict(np.array([sample]))
        print(result)
        print(np.argmax(result))

