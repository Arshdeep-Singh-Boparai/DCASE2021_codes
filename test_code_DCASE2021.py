import h5py    
import numpy as np    
from sklearn.metrics import classification_report,confusion_matrix
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D,AveragePooling2D
from keras import backend as K
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from scipy.stats import mode
import numpy as np
from keras import backend as K
from random import shuffle
from keras.callbacks import ModelCheckpoint
import os
from numpy import *
# SKLEARN
from sklearn.utils import shuffle
import tensorflow as tf
import scipy.io
import pickle
import os
from keras.models import Model
from keras.models import load_model


#%% load model

model=load_model('~/Singh_IITMandi_task1a_1_quant.h5')
model.summary()





#%%
x_test=np.load('~/X_test.npy')
labels_test=np.load('~/Y_test.npy')
#y_test = keras.utils.to_categorical(labels_test, 10)
pred_label=model.predict(x_test)


model.summary()

pred=np.argmax(pred_label,1)

asd=confusion_matrix(labels_test,pred);
accu=(np.trace(asd)/np.size(labels_test))*100;

print(accu,'accuracy quantized')
from sklearn.metrics import log_loss
logloss_overall = log_loss(y_true=labels_test, y_pred=pred_label.astype(np.float64))
print(logloss_overall,'baseline quantized')

#%%
'''
classes=['airport','bus','metro','metro_station','park','public_square','shopping_mall','street_pedestrian','street_traffic','tram']


a_log=[]
a_acc=[]
shift=297
for i in range(10):
    logloss_indi= log_loss(y_true=labels_test[i*shift: (i+1)*shift], y_pred=pred_label[i*shift: (i+1)*shift].astype(np.float64),labels=labels_test)
    print(asd[i,i]/2.97,'%', logloss_indi,'    ', classes[i])
    a_log.append(logloss_indi)
    a_acc.append(asd[i,i]/2.97)


print(np.average(a_log),'   a verage log-loss', '    avg acc',np.average(a_acc))


#%% evaluation label generation

path_eva='~/Evaluation_dataset/test/'
classes=['airport','bus','metro','metro_station','park','public_square','shopping_mall','street_pedestrian','street_traffic','tram']
Eva_label=[]
Eva_class=[]
Eva_prob=[]

for i in range(7920):
    file_name=path_eva+str(i)+'.npy'
    In=np.load(file_name)
    predd_out=model.predict(In)
    pred_label=np.argmax(predd_out)
    Eva_label.append(pred_label)
    Eva_class.append(classes[pred_label])
    Eva_prob.append(predd_out)
    print(i,'.....done..')


# np.save('D:/PhD Paper work/DCASE_2021_challenge_submission/System4/Eva_label',np.array(Eva_label))
# np.save('D:/PhD Paper work/DCASE_2021_challenge_submission/System4/Eva_class',np.array(Eva_class))
# np.save('D:/PhD Paper work/DCASE_2021_challenge_submission/System4/Eva_prob',np.array(Eva_prob))

print('saved',np.shape(Eva_prob))
print(accu,'accuracy')
print(logloss_overall,'baseline')
'''
