import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
from numpy import *
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from keras.utils import np_utils
import tensorflow
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

IMG_SIZE = 50 #50 by 50 pixels
LR = 0.001  #Learning rate =0.001

MODEL_NAME = 'alpha-{}-{}.model'.format(LR, '12connets-basic-final')

path1 = '/home/pmohata/All/'    #path of folder containing images

listing = os.listdir(path1)
num_samples=size(listing)
print (num_samples)

#Creating the data with the essential processing steps
def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(path1)):
        img_name = img.split('-')[0] #extracting the labels from the image name
        img_index = list(img_name[3:])
        img_num = int(img_index[1])*10 + int(img_index[2]) - 1 #storing the labels
        path = os.path.join(path1,img)
        img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (IMG_SIZE, IMG_SIZE)) #resizing the images and converting them into grayscale
        training_data.append([np.array(img), np.array(img_num)]) #appending the image data with the corresponding labels
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data

#loading the data
train_data = create_train_data()

shape(train_data)

train_data[-1]

#Spliting the data into train and test
train = train_data[:-600]
test = train_data[-600:]
shape(train)

#processing the train variables to be fed in tflearn
X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE, IMG_SIZE, 1) 

lbel_train = np.array([i[1] for i in train])
onehot_encoder = OneHotEncoder(sparse = False)
integer_encoded = lbel_train.reshape(len(lbel_train),1)
Y = onehot_encoder.fit_transform(integer_encoded)
print(Y)

#processing the test variables to be fed in tflearn
test_x = np.array([i[0] for i in test]).reshape(-1,IMG_SIZE, IMG_SIZE, 1) 

lbel_test = np.array([i[1] for i in test])
onehot_encoder = OneHotEncoder(sparse = False)
integer_encoded = lbel_test.reshape(len(lbel_test),1)
test_y = onehot_encoder.fit_transform(integer_encoded)
print(test_y)

shape(test_x)

shape(test_y)

#Defining the parameters for training the Convolutional Neural Network
convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 128, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 128, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 128, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.5)

convnet = fully_connected(convnet, 62, activation='softmax') #we have 62 classes
convnet = regression(convnet, optimizer='adam', learning_rate= LR , loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir = 'log' ) # saving the log file to visualize results using tensorboard

#Fitting the Convolutional Neural Network Model with 70 epochs
model.fit({'input': X}, {'targets': Y}, n_epoch=70, validation_set=({'input': test_x}, {'targets': test_y}), 
    snapshot_step=500, show_metric=True, run_id = MODEL_NAME)

#Saving the model
model.save(MODEL_NAME)

