from keras.layers import Lambda, Input, Dense, Conv2DTranspose
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import cv2

from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.layers import Dropout, Activation, Flatten
from keras.layers import Reshape, Embedding,InputLayer
import random

random.seed(1337)
L=10
s=7
epochs=100

# MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
image_size = x_train.shape[1]
original_dim = image_size * image_size
x_train = x_train[:60000].astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train[:60000], (len(x_train[:60000]), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1)) 
y_train=y_train[:60000]

#input_shape = (original_dim, )
input_shape = (image_size, image_size, 1)
intermediate_dim = 512
batch_size = 128
latent_dim = L

# VAE model = encoder + decoder
# build encoder model
inputs = Input(shape=input_shape, name='encoder_input')
x = Conv2D(32, (3, 3), activation='relu', strides=2, padding='same')(inputs)
x = Conv2D(64, (3, 3), activation='relu', strides=2, padding='same')(x)
shape = K.int_shape(x)
print("shape[1], shape[2], shape[3]",shape[1], shape[2], shape[3])
x = Flatten()(x)
z_mean = Dense(latent_dim, name='z_mean')(x)

# instantiate encoder model
encoder = Model(inputs, z_mean, name='encoder')
encoder.summary()

# build decoder model
# decoder
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(shape[1] * shape[2] * shape[3], activation='relu')(latent_inputs)
x = Reshape((shape[1], shape[2], shape[3]))(x)
x = Conv2DTranspose(64, (3, 3), activation='relu', strides=2, padding='same')(x)
x = Conv2DTranspose(32, (3, 3), activation='relu', strides=2, padding='same')(x)
outputs = Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='same')(x)

# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()

# instantiate VAE model
outputs = decoder(encoder(inputs))
vae = Model(inputs, outputs, name='vae_mlp')

vae.compile(loss='binary_crossentropy',optimizer='adam') #loss='binary_crossentropy' #loss="mse"

# 学習に使うデータを限定する
x_train1 = x_train[y_train==s] #7_8_4
x_test1 = x_test[y_test==s] #7_8_4

print(len(x_train1),len(x_test1))

encoder.load_weights("./mnist1000/L"+str(L)+"/"+str(7) +"/100epoch/encoder_mnist_weights_epoch100.h5")
decoder.load_weights("./mnist1000/L"+str(L)+"/"+str(7) +"/100epoch/decoder_mnist_weights_epoch100.h5")

# Get 1st layer Convolution2D weights.
# In this example, weights.shape is (1, 3, 3)
weights = encoder.layers[1].get_weights()[0].transpose(3,2,0,1)
print("weights.shape[0]",weights[0].shape)
fig = plt.figure()
for i, weight_3d in enumerate(weights):
    for j, weight_2d in enumerate(weight_3d):
        sub = fig.add_subplot(weights.shape[0], weight_3d.shape[0], i*weight_3d.shape[0]+j+1)
        sub.axis('off')
        sub.imshow(weight_2d, 'Greys')
plt.savefig("./mnist1000/L"+str(L)+"/"+str(7) +"/encoder_weights_layers[1]_L"+str(L)+"_7.png")
plt.pause(1)

# In this example, weights.shape is (32, 3, 3)
weights = encoder.layers[2].get_weights()[0].transpose(3,2,0,1)
print("weights.shape[0]",weights[0].shape)
fig = plt.figure()
for i, weight_3d in enumerate(weights):
    for j, weight_2d in enumerate(weight_3d):
        sub = fig.add_subplot(weights.shape[0], weight_3d.shape[0], i*weight_3d.shape[0]+j+1)
        sub.axis('off')
        sub.imshow(weight_2d, 'Greys')
plt.savefig("./mnist1000/L"+str(L)+"/"+str(7) +"/encoder_weights_layers[2]_L"+str(L)+"_7.png")
plt.pause(1)

# Get 1st layer Convolution2D weights.
# In this example, weights.shape is (1, 3, 3)
weights = decoder.layers[1].get_weights()[0].transpose(3,2,0,1)
print("weights.shape[0]",weights[0].shape)
fig = plt.figure()
for i, weight_3d in enumerate(weights):
    for j, weight_2d in enumerate(weight_3d):
        sub = fig.add_subplot(weights.shape[0], weight_3d.shape[0], i*weight_3d.shape[0]+j+1)
        sub.axis('off')
        sub.imshow(weight_2d, 'Greys')
plt.savefig("./mnist1000/L"+str(L)+"/"+str(7) +"/decoder_weights_layers[1]_L"+str(L)+"_7.png")
plt.show()

# In this example, weights.shape is (32, 3, 3)
weights = decoder.layers[2].get_weights()[0].transpose(3,2,0,1)
print("weights.shape[0]",weights[0].shape)

fig = plt.figure()
for i, weight_3d in enumerate(weights):
    for j, weight_2d in enumerate(weight_3d):
        sub = fig.add_subplot(weights.shape[0], weight_3d.shape[0], i*weight_3d.shape[0]+j+1)
        sub.axis('off')
        sub.imshow(weight_2d, 'Greys')
plt.savefig("./mnist1000/L"+str(L)+"/"+str(7) +"/decoder_weights_layers[2]_L"+str(L)+"_7.png")
plt.show()

# In this example, weights.shape is (32, 3, 3)
weights = decoder.layers[3].get_weights()[0].transpose(3,2,0,1)
print("weights.shape[0]",weights[0].shape)

fig = plt.figure()
for i, weight_3d in enumerate(weights):
    for j, weight_2d in enumerate(weight_3d):
        sub = fig.add_subplot(weights.shape[0], weight_3d.shape[0], i*weight_3d.shape[0]+j+1)
        sub.axis('off')
        sub.imshow(weight_2d, 'Greys')
plt.savefig("./mnist1000/L"+str(L)+"/"+str(7) +"/decoder_weights_layers[3]_L"+str(L)+"_7.png")
plt.show()

