#https://keras.io/examples/variational_autoencoder/

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Lambda, Input, Dense, Conv2DTranspose
from keras.models import Model
from keras.datasets import mnist, cifar10
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

def plot_fig_vae(x_test, vae_imgs,j):
    n = 100
    plt.figure(figsize=(10, 16))
    for i in range(1,n):
        # display original
        ax = plt.subplot(20, n*0.1, i)
        plt.imshow(x_test[i].reshape(32, 32,3))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(20, n*0.1, i + n)
        plt.imshow(vae_imgs[i].reshape(64, 64,3))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.savefig("mnist_color_vae_{}".format(j))    
    plt.pause(1)
    plt.close()

# Cifar10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
image_size = x_train.shape[1]
original_dim = image_size * image_size
x_train = x_train[:60000].astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train[:60000], (len(x_train[:60000]), 32, 32, 3))
x_test = np.reshape(x_test, (len(x_test), 32, 32, 3)) 
y_train=y_train[:60000]

img_rows, img_cols=64,64
X_train =[]
X_test = []
for i in range(50000):
    dst = cv2.resize(x_train[i], (img_rows, img_cols), interpolation=cv2.INTER_CUBIC) #cv2.INTER_LINEAR #cv2.INTER_CUBIC
    dst = dst[:,:,::-1]  
    X_train.append(dst)
for i in range(10000):
    dst = cv2.resize(x_test[i], (img_rows, img_cols), interpolation=cv2.INTER_CUBIC)
    dst = dst[:,:,::-1]  
    X_test.append(dst)
x_train_size = np.array(X_train)
x_test_size = np.array(X_test)


#input_shape = (original_dim, )
input_shape = (image_size, image_size, 3)
batch_size = 128
latent_dim = 256
epochs = 10
kernel_size = 3

# VAE model = encoder + decoder
# build encoder model
inputs = Input(shape=input_shape, name='encoder_input')
x = Conv2D(64, (3, 3), activation='relu', strides=2, padding='same')(inputs)
x = Conv2D(128, (3, 3), activation='relu', strides=2, padding='same')(x)
x = Conv2D(256, (3, 3), activation='relu', strides=2, padding='same')(x)
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
x = Conv2DTranspose(256, (3, 3), activation='relu', strides=2, padding='same')(x)
x = Conv2DTranspose(128, (3, 3), activation='relu', strides=2, padding='same')(x)
x = Conv2DTranspose(64, (3, 3), activation='relu', strides=2, padding='same')(x)
x = Conv2DTranspose(32, (3, 3), activation='relu', strides=2, padding='same')(x)
outputs = Conv2DTranspose(3, (3, 3), activation='sigmoid', padding='same')(x)

# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()

# instantiate VAE model
outputs = decoder(encoder(inputs))
vae = Model(inputs, outputs, name='vae_mlp')

vae.compile(loss='binary_crossentropy',optimizer='adam') #loss='binary_crossentropy' #loss="mse"

# 学習に使うデータを限定する
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

#x_train_gray =  rgb2gray(x_train_size).reshape(10000,64,64,1)
#x_test_gray =  rgb2gray(x_test_size).reshape(10000,64,64,1)

# autoencoderの実行
vae.fit(x_train,x_train_size,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(x_test, x_test_size))

vae_imgs = vae.predict(x_test)
        
plot_fig_vae(x_test,vae_imgs,0)

# 実行結果の表示
n = 10
decoded_imgs = vae.predict(x_test[:n])

plt.figure(figsize=(10, 4))
for i in range(n):
    # original_image
    orig_img0 = x_test_size[i].reshape(2*image_size, 2*image_size,3)
    orig_img = x_test[i].reshape(image_size, image_size,3)

    # reconstructed_image
    reconst_img = decoded_imgs[i].reshape(2*image_size, 2*image_size,3)

    # diff image
    diff_img = ((orig_img0 - reconst_img)+2)/4
    diff_img = (diff_img*255).astype(np.uint8)
    orig_img0 = (orig_img0*255).astype(np.uint8)
    orig_img = (orig_img*255).astype(np.uint8)
    reconst_img = (reconst_img*255).astype(np.uint8)
    diff_img_color = cv2.applyColorMap(diff_img, cv2.COLORMAP_JET)

    # display original
    ax = plt.subplot(4, n,  i + 1)
    plt.imshow(orig_img0, cmap=plt.cm.gray)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(4, n,  i+ n*1 + 1)
    plt.imshow(orig_img, cmap=plt.cm.gray)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(4, n, i + n*2 + 1)
    plt.imshow(reconst_img, cmap=plt.cm.gray)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display diff
    ax = plt.subplot(4, n, i + n*3 + 1)
    plt.imshow(diff_img, cmap=plt.cm.jet)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.savefig("autodetect_color.jpg")
plt.pause(1)
plt.close()

# 学習結果の保存
vae.save('./ae_mnist_nr.h5')

# json and weights
model_json = vae.to_json()
with open('ae_cifar10.json', 'w') as json_file:
    json_file.write(model_json)
vae.save_weights('ae_cifar10_weights_color.h5')
