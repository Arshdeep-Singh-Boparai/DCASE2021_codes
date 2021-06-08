# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 21:06:57 2021

@author: Arshdeep Singh
"""
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


import matplotlib.pyplot as plt

from keras import backend as K
#K.set_image_dim_ordering('th')
from random import shuffle
from keras.callbacks import ModelCheckpoint
import os

# SKLEARN
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

import scipy.io
import pickle
from keras.models import Model
from keras.models import load_model
from scipy.stats.mstats import gmean

from sklearn.metrics import log_loss

#%% Unpruned model
input_shape=(40,500,1)
##model building
model = Sequential()
#convolutional layer with rectified linear unit activation
model.add(Conv2D(16, kernel_size=(7, 7),padding='same',input_shape=input_shape))

model.add(BatchNormalization(axis=-1)) #layer2
convout1= Activation('relu')
model.add(convout1) #laye



#''''''''''''''''''''''''''''''''''''''''''''''''

model.add(Conv2D(16, kernel_size=(7, 7),padding='same'))

model.add(BatchNormalization()) #layer2
convout2= Activation('relu')
model.add(convout2) #laye

model.add(MaxPooling2D(pool_size=(5, 5)))

model.add(Dropout(0.30))

model.add(Conv2D(32, kernel_size=(7, 7),padding='same'))

model.add(BatchNormalization()) #layer2
convout2= Activation('relu')
model.add(convout2) #laye

model.add(MaxPooling2D(pool_size=(4, 100)))

model.add(Dropout(0.30))


model.add(Flatten())

model.add(Dense(100,activation='relu'))
model.add(Dropout(0.30))

model.add(Dense(10, activation='softmax'))


model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

#%%

model.load_weights('~/model_quant_fold_1.h5')   # unpruned model pre-trained weights
W_dcas=model.get_weights()



#%%  DATA load for pre-trained model evaluation

x_test=np.load('/home/arshdeep/DCASE2021_task1a/DCASE2021/X_test.npy')
labels_test=np.load('/home/arshdeep/DCASE2021_task1a/DCASE2021/Y_test.npy')
#labels_train=np.load('Y_train')


pred_label=model.predict(x_test)

pred=np.argmax(pred_label,1)

asd=confusion_matrix(labels_test,pred);
accu=(np.trace(asd)/np.size(labels_test))*100;
print(accu,'accuracy')

logloss_overall = log_loss(y_true=labels_test, y_pred=pred_label, normalize=True)

print(logloss_overall)






#%% Pruned model l1-norm and GM pruning....

def CVPR_L1_Imp_index(W):  #L1 pruning importance  score calculation
	Score=[]
	for i in range(np.shape(W)[3]):
		Score.append(np.sum(np.abs(W[:,:,:,i])))
	return Score


def CVPR_GM_Imp_index(W): # #GM pruning importance  score calculation
	G_GM=gmean(np.abs(W.flatten()))
	Diff=[]
	for i in range(np.shape(W)[2]):
		F_GM=gmean(np.abs(W[:,:,i]).flatten())
		Diff.append((G_GM-F_GM)**2)
	return Diff	


#%%	 algorithm to save sorted indexex according to their imjportance scores	
os.chdir('~/L1_Pruning')

initia_weights=W_dcas
indexes=[0,6,12]#,18,24,30,36,42,48,54,60,66,72]
L=[0,6,12]#,4,5,6,7,8,9,10,11,12,13]

for j in range(len(L)):
	print(j)
	W_2D=initia_weights[indexes[j]]
	W=np.reshape(W_2D,(49,np.shape(W_2D)[2],np.shape(W_2D)[3]))
	print(np.shape(W),'layer  :','  ',L[j])
	print(np.shape(W),'shape of weights')
	Score_L1=	CVPR_L1_Imp_index(W_2D)
# 	Score_GM=CVPR_GM_Imp_index(W)
 	Sorted_arg=np.argsort(Score_L1)
# 	plt.figure()
# 	plt.plot(Score_GM/np.max(Score_GM))
	file_name='sim_index'+str(L[j])+'.npy'
	plt.title(file_name)
	np.save(file_name,Sorted_arg)
#	print(len(Imp_F),'important filters in layer ',L[j])
	 



#%% define pruning ratio for each layer

p1=0.25  #pruning ratio    
p2=0.25
p3=0.25


L1=np.arange(0,16*(1-p1))#
L2=np.arange(0,16*(1-p2))#np.load('sim_index2.npy')#np.arange(0,64)#np.arange(0,64)#
L3=np.arange(0,32*(1-p3))

print('..........L1 Pruning............')
os.chdir('~/L1_pruning')
L1=sort(np.load('sim_index0.npy')[16-len(L1):16])
L2=sort(np.load('sim_index6.npy')[16-len(L2):16])#np.arange(0,64)#
L3=sort(np.load('sim_index12.npy')[32-len(L3):32])

#np.save('Baseline_weights',W_dcas)
L3_n=L3*2
L3_n1=L3_n+1
w_f=[]
for i in range(len(L3)):
	w_f.append(L3_n[i])
	w_f.append(L3_n1[i])
	

Total_filter=len(L1)+len(L2)+len(L3)#+len(L4)+len(L5)+len(L6)+len(L7)+len(L8)+len(L9)+len(L10)+len(L11)+len(L12)+len(L13)
W=W_dcas



W_pruned=[W[0][:,:,:,L1],W[1][L1],W[2][L1],W[3][L1],W[4][L1],W[5][L1],W[6][:,:,L1,:][:,:,:,L2],W[7][L2],W[8][L2],W[9][L2],W[10][L2],W[11][L2],	W[12][:,:,L2,:][:,:,:,L3],W[13][L3],W[14][L3],W[15][L3],W[16][L3],W[17][L3],W[18][w_f,:],W[19],W[20],W[21]]

						
#%% new  pruned model

input_shape=(40,500,1)
##model building
model1 = Sequential()
#convolutional layer with rectified linear unit activation
model1.add(Conv2D(len(L1), kernel_size=(7, 7),padding='same',input_shape=input_shape))

model1.add(BatchNormalization(axis=-1)) #layer2
convout1= Activation('relu')
model1.add(convout1) #laye



#''''''''''''''''''''''''''''''''''''''''''''''''

model1.add(Conv2D(len(L2), kernel_size=(7, 7),padding='same'))

model1.add(BatchNormalization()) #layer2
convout2= Activation('relu')
model1.add(convout2) #laye

model1.add(MaxPooling2D(pool_size=(5, 5)))

model1.add(Dropout(0.30))



model1.add(Conv2D(len(L3), kernel_size=(7, 7),padding='same'))

model1.add(BatchNormalization()) #layer2
convout2= Activation('relu')
model1.add(convout2) #laye

model1.add(MaxPooling2D(pool_size=(4, 100)))

model1.add(Dropout(0.30))


model1.add(Flatten())

model1.add(Dense(100,activation='relu'))
model1.add(Dropout(0.30))

model1.add(Dense(10, activation='softmax'))


model1.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])


model1.set_weights(W_pruned)

pred_label=model1.predict(x_test)


pred=np.argmax(pred_label,1)

asd=confusion_matrix(labels_test,pred);
accu=(np.trace(asd)/np.size(labels_test))*100;
print(accu,'accuracy')
logloss_overall = log_loss(y_true=labels_test, y_pred=pred_label)
print(logloss_overall)
#%% finetuning of the pruned model.....


checkpointer = ModelCheckpoint(filepath='~/best_weights_DCASE2021_200_n1.h5py',monitor='val_loss',verbose=1, save_best_only=True,save_weights_only=True)
hist=model1.fit(x_train, y_train,batch_size=64,epochs=200,verbose=1,validation_data=(x_test, y_test),callbacks=[checkpointer],shuffle=True)

model1.load_weights('~/best_weights_DCASE2021_200_n1.h5py')
model1.save('~/pruned_model.h5')

pred_label=model1.predict(x_test)
pred=np.argmax(pred_label,1)

asd=confusion_matrix(labels_test,pred);
accu=(np.trace(asd)/np.size(labels_test))*100;
print(accu,'accuracy')



logloss_overall = log_loss(y_true=labels_test, y_pred=pred_label)
print(logloss_overall)

model1.summary()


#%% Quantization

tf.keras.backend.set_floatx('float16')
ws = model1.get_weights()
wsp = [w.astype(tf.keras.backend.floatx()) for w in ws]

#build new quantized model

model_quant.add(Conv2D(len(L2), kernel_size=(7, 7),padding='same'))

model_quant.add(BatchNormalization()) #layer2
convout2= Activation('relu')
model_quant.add(convout2) #laye

model_quant.add(MaxPooling2D(pool_size=(5, 5)))

model1.add(Dropout(0.30))



model_quant.add(Conv2D(len(L3), kernel_size=(7, 7),padding='same'))

model_quant.add(BatchNormalization()) #layer2
convout2= Activation('relu')
model_quant.add(convout2) #laye

model1.add(MaxPooling2D(pool_size=(4, 100)))

model_quant.add(Dropout(0.30))


model_quant.add(Flatten())

model_quant.add(Dense(100,activation='relu'))
model_quant.add(Dropout(0.30))

model_quant.add(Dense(10, activation='softmax'))


model_quant.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])


model_quant.set_weights(wsp)



