# import matplotlib.pyplot as plt
# import tensorflow as tf
# import glob
# import numpy as np
# import pandas as pd
# import os
# import librosa
# import librosa.display
# import librosa.feature
# import matplotlib.pyplot
# import matplotlib.pyplot as plt
# from Audiopy_ML import autoaudio

# from keras.models import load_model

# # 加载模型
# model = load_model('./test/machinelisten.h5',compile=False)

# pre_data_path = glob.glob(r"./data1/*.wav")

# from sklearn.preprocessing import StandardScaler, Normalizer
# print('0')
# yc = autoaudio.AutomatedExtractor_multiple(pre_data_path)
# print('00')
# yc = yc.applymap((lambda x: np.median(x)))
# print('000')
# y = yc.values
# print('1')
# y=Normalizer().fit_transform(y)
# print('2')
# y=StandardScaler().fit_transform(y)
# print('3')
# predicted_audio = model.predict(np.expand_dims(y[1], axis=0))
# print('4')
# print(predicted_audio)



import numpy as np
import librosa.feature
from keras.models import load_model
model = load_model('./test/machinelisten.h5',compile=False)
print('0')
from sklearn.preprocessing import StandardScaler, Normalizer
audio_file = "./data1/anomaly_id_00_00000000.wav"
print('00')
y, sr = librosa.load(audio_file, sr=None)
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=7)
print('000')
mfcc = np.median(mfccs, axis=1)
mfcc = mfcc.T
print('1')
y = Normalizer().fit_transform(mfcc)
print('2')
y = StandardScaler().fit_transform(y)
print('3')
predicted_audio = model.predict(np.expand_dims(y[0], axis=0))

print('4')