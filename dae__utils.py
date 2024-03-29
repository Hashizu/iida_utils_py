# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import re
import heapq
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from scipy.special import erfinv

import keras
import tensorflow as tf
from keras.layers import Dense, Input
from keras.layers import BatchNormalization
from keras.optimizers import Adam
from keras.models import Model, load_model
from keras import callbacks
from keras import backend as K
from keras.layers import Dropout
import gc
import random
from numba.decorators import jit


home_path ='./drive/My Drive/SIGNATE/'

def mape(y_true, y_pred):
  y_true, y_pred = np.array(y_true), np.array(y_pred)
  return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

@jit
def noise(array):
    height = len(array)
    width = len(array[0])
    rands = np.random.uniform(0, 1, (height, width) )
    copy  = np.copy(array)
    for h in range(height):
        for w in range(width):
            if rands[h, w] <= 0.05:
                swap_target_h = random.randint(0,h)
                copy[h, w] = array[swap_target_h, w]
    return copy

  
def DAE(df, d_df):
    """DenoisingAutoEncorder by keras
    
    Parameters
    ----------
    df : DataFrame
        Original input-DataFrame
    d_df : DataFrame 
        Added noise input-DataFrame
    
    """

    array = np.array(df)
    d_array = np.array(d_df)
    X_train, X_test, X_train_d, X_test_d = train_test_split(array, d_array, test_size=0.1, random_state=42)
    
    inp = Input(shape=(X_train_d.shape[1],))
    x = Dense(1000, activation="relu")(inp)
    x = BatchNormalization()(x)
    x = Dropout(rate = 0.3)(x)
    x = Dense(1000, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(rate = 0.3)(x)
    x = Dense(2500, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(rate = 0.2)(x)
    out = Dense(X_train.shape[1], activation="relu")(x)
    clf = Model(inputs=inp, outputs=out)
    clf.compile(loss='mean_squared_error', optimizer='adam')
    es = callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=1,
                                 verbose=1, mode='auto', baseline=None, restore_best_weights=True)
    rlr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                      patience=3, min_lr=1e-6, mode='max', verbose=0)
    clf.fit(X_train_d, X_train, validation_data=(X_test_d, X_test),
            callbacks=[es], epochs=500, batch_size=512,verbose=1)
    
    _input = clf.layers[0].input
    _output = clf.layers[1].output
    _output2 = clf.layers[2].output
    _output3 = clf.layers[3].output
    
    func = K.function([_input, K.learning_phase()], [_output,_output2, _output3])    
    d_array = [func(([d_array[x]],0)) for x in range(len(array))]
    
    return np.array(d_array)
  
  
def denoise(data):
  print('denoising process')
  # make noise data
  
  
  d_data = pd.DataFrame(noise(np.array(data)), columns=data.columns)

  
  none_list = list(set(data.columns)- set(d_data.columns))
  for x in none_list:
    d_data[x]  = np.zeros(len(d_data))

  d_data = d_data.fillna(0)*1
  data = data.fillna(0)*1

  d_data = normalize_rank(d_data, all_nr = True)
  data = normalize_rank(data, all_nr = True)
  
  
  dae_data = DAE(data, d_data)
  dae_data = dae_data.reshape(dae_data.shape[0],-1)
  return pd.DataFrame(dae_data)

def normalize_rank(df, all_nr= False):
  print('normalize rank')
  
  if all_nr:
    numerical = df.columns
  else:
    numerical =['...']
  
  for x in numerical:
    series = df[x].rank()
    M = series.max()
    m = series.min() 
    series = (series-m)/(M-m)
    series = series - series.mean()
    series = series.apply(erfinv) 
    df[x] = series
    
  return df

## Preprocess

data = pd.DataFrame(np.zeros((1000,200)))

d_data = denoise(data)

print('data loaded\n')
print('gc collection :  '+ str(gc.collect()))

train = data[:len(goto)]*1
test = data[len(goto):]*1

print('done')

# %%time
n_model = 2
skf = KFold(n_splits=5, random_state=1128, shuffle=True)

oof_val = np.zeros(len(train))
oof_train = np.zeros(len(train))
preds = np.zeros(len(test))
X = np.array(train)
X = X.reshape(X.shape[0],X.shape[-1])
y = np.array(data_y)
y = np.log(y)

i=0


def NN_model(X_train, y_train, X_val, y_val, X_test, y_test, test):
    inp = Input(shape=(X_train.shape[1],))
    x = Dense(500, activation="relu")(inp)
    x = BatchNormalization()(x)
    x = Dropout(rate = 0.5)(x)

    out = Dense(1, activation="relu")(x)
    clf = Model(inputs=inp, outputs=out)
    clf.compile(loss='mape', optimizer='adam')
    es = callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10,
                                 verbose=1, mode='auto', baseline=None, restore_best_weights=True)
    rlr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.945,
                                      patience=3, min_lr=1e-6, mode='max', verbose=0)
    clf.fit(X_train, y_train, validation_data=(X_val, y_val),
            callbacks=[es], epochs=500, batch_size=16,verbose=1)
    
    train_pred = clf.predict(X_train)
    test_pred = clf.predict(X_test)
    preds = clf.predict(test)
    
    return train_pred, test_pred, preds


for train_pj, test_pj in skf.split(range(data_pj.max()+1)):

  train_index = [x in train_pj for x in data_pj]
  test1_num, test2_num = train_test_split(test_pj, test_size=0.5, random_state=11)  
  test1_index = [x in test1_num for x in data_pj]
  test2_index = [x in test2_num for x in data_pj]
  
  X_train, X_test1, X_test2 = X[train_index], X[test1_index], X[test2_index]
  y_train, y_test1, y_test2 = y[train_index], y[test1_index], y[test2_index]
    
  i=i+1
  print('{} - a'.format(i))  
  
  temp_train, temp_test, preds_ = NN_model(X_train, y_train, X_test1, y_test1, X_test2, y_test2, test)    
  oof_val[test2_index] += temp_test.reshape(len(temp_test))
  oof_train[train_index] += temp_train.reshape(len(temp_train))/(skf.n_splits-1)/n_model
  preds += preds_.reshape(len(test))

   
  print('{} - b'.format(i))
  temp_train , temp_test, preds_ = NN_model(X_train, y_train, X_test2, y_test2, X_test1, y_test1, test)    
  oof_val[test1_index] += temp_test.reshape(len(temp_test))
  oof_train[train_index] += temp_train.reshape(len(temp_train))/(skf.n_splits-1)/n_model
  preds += preds_.reshape(len(test))
  
preds /= 10
print('train : ' + str(mape(data_y, pow(np.e,oof_train))))
print('valid : ' + str(mape(data_y, pow(np.e,oof_val))))
