import os
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import theano
from numpy import *
import numpy as np
from PIL import Image
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split

from keras.utils import np_utils
from keras.models import Sequential
#from keras.layers.Convolutional import Convolution2D,MaxPooling2D
	
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.core import Dense,Dropout,Activation,Flatten
from keras.optimizers import SGD,RMSprop,adam




img_rows,img_cols=200,200
path1='/Users/priyankajaiswal/Desktop/Raw/'
path2='/Users/priyankajaiswal/Desktop/Output/'
#path1='/Users/priyankajaiswal/Desktop/detectioncode'
#path2='/Users/priyankajaiswal/Downloads/detectioncode/Output'

listing=os.listdir(path1)
num_samples=size(listing)
print (num_samples)

#for file in listing:
 #im=Image.open(path1 + file)
 #rgb=im.resize((img_rows,img_cols))
 #gray=rgb.convert('L')
 #gray.save(path2 + file,"JPEG")

imlist=os.listdir(path2)

im1=array(Image.open( path2 +  imlist[0]))
m,n=im1.shape[0:2]
imnbr=len(imlist)

imatrix=array([array(Image.open(path2 + im2)).flatten()
            for im2 in imlist],'f')
print(imatrix)


label=np.ones((num_samples,),dtype = int)
label[0:55]=0
label[55:111]=1
label[111:166]=2
label[166:221]=3
label[221:276]=4
label[276:331]=5
label[331:386]=6
label[386:441]=7
label[441:496]=8
label[496:551]=9
label[551:606]= 10
label[606:661]= 11
label[661:716]= 12
label[716:771]= 13
label[771:826]= 14
label[826:881]= 15
label[881:936]= 16
label[936:991]= 17
label[991:1046]= 18
label[1046:1101]= 19
label[1101:1156]= 20
label[1156:1211]= 21
label[1211:1266]= 22
label[1266:1321]= 23
label[1321:1376]= 24
label[1376:1431]= 25
label[1431:1486]= 26
label[1486:1541]= 27
label[1541:1596]= 28
label[1596:1651]= 29
label[1651:1706]= 30
label[1706:1761]= 31
label[1761:1816]= 32
label[1816:1871]= 33
label[1871:1926]= 34
label[1926:1981]= 35
label[1981:2036]= 36
label[2036:2091]= 37
label[2091:2146]= 38
label[2146:2201]= 39
label[2201:2256]= 40
label[2256:2311]= 41
label[2311:2366]= 42
label[2366:2421]= 43
label[2421:2476]= 44
label[2476:2531]= 45
label[2531:2586]= 46
label[2586:2641]= 47
label[2641:2696]= 48
label[2696:2751]= 49
label[2751:2806]= 50
label[2806:2861]= 51
label[2861:2916]= 52
label[2916:2971]= 53
label[2971:3026]= 54
label[3026:3081]= 55
label[3081:3136]= 56
label[3136:3191]= 57
label[3191:3246]= 58
label[3246:3301]= 59
label[3301:3356]= 60
label[3356:3411]= 61

data,label=shuffle(imatrix,label,random_state=4)
train_data=[data,label]

img=imatrix[20].reshape(img_rows,img_cols)
plt.imshow(img)

print(train_data[0].shape)
print(train_data[1].shape)

batch_size=32
nb_classes=62
nb_epoch=20

img_rows,img_cols=200,200
img_channels=1
nb_filters=32
nb_pool=2
nb_conv=3

(X,y)=(train_data[0],train_data[1])

X_train, X_test, y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=4)

X_train=X_train.reshape(X_train.shape[0],img_rows,img_cols,1)
X_test=X_test.reshape(X_test.shape[0],img_rows,img_cols,1)

X_train=X_train.astype('float32')
X_test=X_test.astype('float32')


X_train /=255
X_test /=255

print('X_train_samples',X_train.shape)
print(X_train.shape[0],'train_samples')
print(X_test.shape[0],'test samples')

Y_train=np_utils.to_categorical(y_train,nb_classes)
Y_test=np_utils.to_categorical(y_test,nb_classes)

model = Sequential()
model.add(Convolution2D(nb_filters,(nb_conv, nb_conv),
padding='valid',
input_shape=(img_rows, img_cols,1)))
convout1=Activation('relu')
model.add(Convolution2D(nb_filters, (nb_conv, nb_conv)))
convout2=Activation('relu')
model.add(MaxPooling2D(pool_size=(nb_pool,nb_pool)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
#X_train = X_train.transpose(0,3,1,2)
#Y	_train = Y_train.transpose(0,3,1,2)
model.fit(X_train,Y_train,batch_size=batch_size,epochs=nb_epoch,
	         verbose=1,validation_data=(X_test,Y_test))
#score = model.evaluate(X_test, Y_test, verbose=0)
print (X_train.shape)

