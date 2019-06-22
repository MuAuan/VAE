from __future__ import print_function

import keras
from keras.datasets import mnist, cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.layers import Lambda, Input, Dense, Conv2DTranspose
from keras.models import Model
from keras.layers import *
import matplotlib.pyplot as plt
import cv2

batch_size = 128
num_classes = 10
epochs = 200

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
s=10
s0=1
x_train = x_train.reshape(50000, 32, 32,3 ) #28,28,1)
x_test = x_test.reshape(10000, 32, 32,3 ) #28,28,1)
x_train = x_train.astype('float32')/255.
x_test = x_test.astype('float32')/255.
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
# convert class vectors to binary class matrices
#y_train = keras.utils.to_categorical(y_train, num_classes)
#y_test = keras.utils.to_categorical(y_test, num_classes)
x_train1=[]
y_train1=[]
x_test1=[]
y_test1=[]
x_train1=x_train
y_train1=y_train
x_test1=x_test
y_test1=y_test
"""
for i in range(len(x_train)):
    if y_train[i]==s0:
        img = x_train[i][np.argmax(y_train[i])==s0] #7_8_4
        x_train1.append(img)
        img = y_train[i][np.argmax(y_train[i])==s0] #7_8_4
        y_train1.append(img)
for i in range(len(x_test)):
    if y_test[i]==s0:
        img = x_test[i][np.argmax(y_test[i])==s0] #7_8_4
        x_test1.append(img)
        img = y_test[i][np.argmax(y_test[i])==s0] #7_8_4
        y_test1.append(img)
"""        
x_train1=np.array(x_train1)
x_test1=np.array(x_test1)
print(x_train1.shape[0], 'train samples')
print(x_test1.shape[0], 'test samples')

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
x_train1_gray =  rgb2gray(x_train1).reshape(len(x_train1),32,32,1)
x_test1_gray =  rgb2gray(x_test1).reshape(len(x_test1),32,32,1)

# convert class vectors to binary class matrices
y_train1 = keras.utils.to_categorical(y_train1, num_classes)
y_test1 = keras.utils.to_categorical(y_test1, num_classes)


def model_cat(input_image=Input(shape=(32, 32,1 ))): #28,28,1))):
    # A common Conv2D model
    x = Conv2D(64, (3, 3), activation='relu',padding='same')(input_image)
    x = Conv2D(64, (3, 3), activation='relu',padding='same')(x)
    x = BatchNormalization(axis=3)(x)  
    x = Dropout(0.5)(x)                
    x = AveragePooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu',padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu',padding='same')(x)
    x = BatchNormalization(axis=3)(x)  
    x = Dropout(0.5)(x)                
    x = AveragePooling2D((2, 2))(x)    
    x = Conv2D(256, (3, 3), activation='relu',padding='same')(x)  
    x = Conv2D(256, (3, 3), activation='relu',padding='same')(x)  
    #x = BatchNormalization(axis=3)(x)  
    x = Dropout(0.5)(x)
    x = AveragePooling2D(pool_size=(2, 2), strides=None, border_mode='valid', dim_ordering='tf')(x)
    x = Flatten()(x)
    #z_mean = Dense(latent_dim, name='z_mean')(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    return Model(inputs=input_image, outputs=outputs)

model = model_cat(input_image=Input(shape=(32, 32,1 ))) #28,28,1)))
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

model.load_weights("../category/cifar10_color/cat_color_weights_"+str(s)+".h5")  #s

img=x_test1[0]
img_gray=x_test1_gray[0]
cat=model.predict(img_gray.reshape(1,32, 32,1 ))  #28,28,1))
plt.imshow(img.reshape(32,32,3))  #28,28))
plt.savefig("original_cifar10_"+str(s)+".jpg")

plt.pause(1)
plt.close()
s0=np.argmax(cat)
print(s0,np.argmax(y_test1[0]))


def encoder_decoder_model(input_image=Input(shape=(32, 32,1))):  #28,28,1))):
    x = Conv2D(32, (3, 3), activation='relu', strides=2, padding='same')(input_image)
    x = Conv2D(64, (3, 3), activation='relu', strides=2, padding='same')(x)
    #x = Conv2D(128, (3, 3), activation='relu', strides=2, padding='same')(x)
    x = Flatten()(x)
    #z_mean = Dense(latent_dim, name='z_mean')(x)
    # build decoder model
    #x = Dense(8 * 8 * 64, activation='relu')(x)  #(latent_inputs)
    x = Reshape((8, 8, 64))(x)
    #x = Conv2DTranspose(128, (3, 3), activation='relu', strides=2, padding='same')(x)
    x = Conv2DTranspose(64, (3, 3), activation='relu', strides=2, padding='same')(x)
    x = Conv2DTranspose(32, (3, 3), activation='relu', strides=2, padding='same')(x)
    outputs = Conv2DTranspose(3, (3, 3), activation='sigmoid', padding='same')(x)  #1
    return Model(input_image, outputs)

item='decoder'
#latent_dim=2
encoder_decoder = encoder_decoder_model(input_image=Input(shape=(32, 32,1)))  #28,28,1)))
encoder_decoder.summary()

encoder_decoder.compile(loss='binary_crossentropy',optimizer='adam')

encoder_decoder.load_weights("../category/cifar10_color/encoder_decoder_color_weights_"+str(s)+".h5")
fig=plt.figure(figsize=(30, 10))
size=(320,320)
s0=0
for i in range(1000):
    if np.argmax(y_test1[i])==s0:
        ax=fig.add_subplot(1,3,1)
        img=x_test1[i]
        img_gray=x_test1_gray[i]
        cat=encoder_decoder.predict(img_gray.reshape(1,32, 32,1 )) #/255   #28,28,1))/255
        cat=cat.reshape(32,32,3)
        plt.gray()
        img_gray=cv2.resize(img_gray, size,interpolation = cv2.INTER_CUBIC)
        ax.imshow(np.clip(img_gray.reshape(320, 320),0,1) ) #28,28))
        
        ax=fig.add_subplot(1,3,2)
        cat=cv2.resize(cat, size,interpolation = cv2.INTER_CUBIC)    
        ax.imshow(np.clip(cat.reshape(320, 320,3 ),0,1) ) #28,28))
        ax=fig.add_subplot(1,3,3)
        img=cv2.resize(img, size,interpolation = cv2.INTER_CUBIC)
        ax.imshow(np.clip(img.reshape(320, 320,3),0,1) ) #28,28))
        ax.set_title(str(np.argmax(y_test1[i])),size=100)
        
        plt.pause(0.1)
        print(np.argmax(y_test1[i]))
        plt.savefig("./"+str(s0)+"/original_cifar10_"+str(i)+".jpg")
        plt.clf()