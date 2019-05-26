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
import keras

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

def plot_fig_vae(x_test,x_test_gray, vae_imgs,k):
    n = 10
    plt.figure(figsize=(8, 16))
    for j in range(0,n):
        for i in range(1,n+1):
            # display original
            ax = plt.subplot(30, n*1, i+3*10*j)
            plt.imshow(x_test[i+10*j].reshape(32, 32,3))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            # display original_gray
            ax = plt.subplot(30, n*1, i+3*10*j+10)
            plt.imshow(x_test_gray[i+10*j].reshape(32, 32))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            # display reconstruction
            ax = plt.subplot(30, n*1, i + 3*10*j+20)
            plt.imshow(vae_imgs[i+10*j].reshape(256,256,3)) #128, 128,3))
            #plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    plt.savefig("./size/cifar10_size_vae_0_{}".format(k))    
    plt.pause(1)
    plt.close()
    
def plot_fig_vae2(x_test_gray,x_test_canny, vae_imgs,k):
    n = 10
    plt.figure(figsize=(16, 16))
    for j in range(0,5):
        for i in range(1,n+1):
            plt.gray()
            ax1 = plt.subplot(10, n*1, i + 2*10*j)
            ax1.imshow(x_test_gray[i+10*j].reshape(32,32))  #256,256,3))    #128, 128,3))
            ax2 = plt.subplot(10, n*1, i + 10*(2*j+1))
            ax2.imshow(vae_imgs[i+10*j].reshape(32,32))
            ax1.get_xaxis().set_visible(False)
            ax1.get_yaxis().set_visible(False)
            ax2.get_xaxis().set_visible(False)
            ax2.get_yaxis().set_visible(False)
    plt.savefig("./size/cifar10_4xsize_vae_0_{}".format(k))    
    plt.pause(1)
    plt.close()

# 実行結果の表示
def plot_fig4(x_test,x_test_gray,vae_imgs,k):
    n = 20
    decoded_imgs = vae.predict(x_test_gray[:n])

    plt.figure(figsize=(20, 4))
    for i in range(n):
        # original_image
        orig_img0 = x_test_size[i].reshape(8*image_size, 8*image_size,3)
        orig_img = x_test[i].reshape(image_size, image_size,3)
        orig_img_gray = x_test_gray[i].reshape(image_size, image_size)

        # reconstructed_image
        reconst_img = decoded_imgs[i].reshape(8*image_size, 8*image_size,3)

        # diff image
        diff_img = ((orig_img0 - reconst_img)+2)/4
        diff_img = (diff_img*255).astype(np.uint8)
        orig_img0 = (orig_img0*255).astype(np.uint8)
        orig_img = (orig_img*255).astype(np.uint8)
        orig_img_gray = (orig_img_gray*255).astype(np.uint8)
        reconst_img = (reconst_img*255).astype(np.uint8)
        diff_img_color = cv2.applyColorMap(diff_img, cv2.COLORMAP_JET)

        # display original
        ax = plt.subplot(5, n,  i + 1)
        plt.imshow(orig_img0, cmap=plt.cm.gray)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(5, n,  i+ n*1 + 1)
        plt.imshow(orig_img, cmap=plt.cm.gray)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(5, n,  i+ n*2 + 1)
        plt.imshow(orig_img_gray, cmap=plt.cm.gray)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    
        # display reconstruction
        ax = plt.subplot(5, n, i + n*3 + 1)
        plt.imshow(reconst_img, cmap=plt.cm.gray)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display diff
        ax = plt.subplot(5, n, i + n*4 + 1)
        plt.imshow(diff_img, cmap=plt.cm.jet)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.savefig("./size/autodetect_size_0_{}.jpg".format(k))
    plt.pause(1)
    plt.close()
    
    
# Cifar10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
image_size = x_train.shape[1]
original_dim = image_size * image_size
x_train = x_train[:60000].astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 32, 32, 3))
x_test = np.reshape(x_test, (len(x_test), 32, 32, 3)) 
y_train=y_train[:60000]

x_train1=[]
x_test1=[]
for i in range(50000):
    if y_train[i]==9:
        x_train1.append(x_train[i])
for i in range(10000):
    if y_test[i]==9:
        x_test1.append(x_test[i])
x_train1=np.array(x_train1)
x_test1=np.array(x_test1)

img_rows, img_cols=256,256 #128,128
X_train =[]
X_test = []
for i in range(len(x_train1)):
    dst = cv2.resize(x_train1[i], (img_rows, img_cols), interpolation=cv2.INTER_CUBIC) #cv2.INTER_LINEAR #cv2.INTER_CUBIC
    #dst = dst[:,:,::-1]  
    X_train.append(dst)
for i in range(len(x_test1)):
    dst = cv2.resize(x_test1[i], (img_rows, img_cols), interpolation=cv2.INTER_CUBIC)
    #dst = dst[:,:,::-1]  
    X_test.append(dst)
x_train_size = np.array(X_train)
x_test_size = np.array(X_test)


#input_shape = (original_dim, )
input_shape = (image_size, image_size, 1)
batch_size = 128
latent_dim = 2048 #4096 #1024
epochs = 101
kernel_size = 3

# VAE model = encoder + decoder
# build encoder model
inputs = Input(shape=input_shape, name='encoder_input')
x = Conv2D(32, (3, 3), activation='relu', strides=2, padding='same')(inputs)
x = Conv2D(64, (3, 3), activation='relu', strides=2, padding='same')(x)
x = Conv2D(128, (3, 3), activation='relu', strides=2, padding='same')(x)
#x=inputs
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
x = Conv2DTranspose(512, (3, 3), activation='relu', strides=2, padding='same')(x)
x = Conv2DTranspose(256, (3, 3), activation='relu', strides=2, padding='same')(x)
x = Conv2DTranspose(128, (3, 3), activation='relu', strides=2, padding='same')(x)
#x = Conv2DTranspose(64, (3, 3), activation='relu', strides=2, padding='same')(x)
#x = Conv2DTranspose(32, (3, 3), activation='relu', strides=2, padding='same')(x)
#x = Conv2DTranspose(16, (3, 3), activation='relu', strides=2, padding='same')(x)
#outputs = Conv2DTranspose(3, (3, 3), activation='sigmoid', padding='same')(x)
outputs = Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='same')(x)


# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()

# instantiate VAE model
outputs = decoder(encoder(inputs))
vae = Model(inputs, outputs, name='vae_mlp')

vae.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy']) #loss='binary_crossentropy' #loss="mse"

# 学習に使うデータを限定する
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

x_train_gray =  rgb2gray(x_train1).reshape(len(x_train1),32,32,1)
x_test_gray =  rgb2gray(x_test1).reshape(len(x_test1),32,32,1)
from keras.preprocessing.image import load_img, img_to_array

x_train_canny=[]
x_test_canny=[]
imgs=[]
for i in range(len(x_train1)):
    x_train_canny =  rgb2gray(x_train1[i]).reshape(32,32)
    x_train_canny = img_to_array(x_train_canny*255).astype(np.uint8)
    x_train_canny = cv2.Canny(x_train_canny, 128,128 )
    x_train_canny = cv2.bitwise_not(x_train_canny)
    imgs.append(x_train_canny)
x_train_canny=img_to_array(imgs).reshape(len(x_train1),32,32,1)
imgs2=[]
for i in range(len(x_test1)):
    x_test_canny =  rgb2gray(x_test1[i]).reshape(32,32)
    x_test_canny = img_to_array(x_test_canny*255).astype(np.uint8)
    x_test_canny = cv2.Canny(x_test_canny, 128,128 )
    x_test_canny = cv2.bitwise_not(x_test_canny)
    imgs2.append(x_test_canny)
x_test_canny=img_to_array(imgs2).reshape(len(x_test1),32,32,1)

#vae.load_weights('./size/ae_cifar10_weights_color_size_0_200.h5',by_name=True) 
#encoder.load_weights('./size/encoder_mnist_weights_200.h5', by_name=True)
#decoder.load_weights('./size/decoder_mnist_weights_200.h5', by_name=True)

class Check_layer(keras.callbacks.Callback): 
    def on_epoch_end(self, epoch, logs={}):
        s=epoch
        if s%5==0:
            vae_imgs = vae.predict(x_test_canny)
            #plot_fig_vae(x_test_gray,x_test_canny,vae_imgs,k=s)
            plot_fig_vae2(x_test_gray,x_test_canny,vae_imgs,k=s)
            #plot_fig4(x_test_gray,x_test_canny,vae_imgs,k=s)
            #vae.save_weights('./size/ae_cifar10_weights_color_size_0_{}.h5'.format(s))
            #encoder.save_weights('./size/encoder_mnist_weights_{}.h5'.format(s))
            #decoder.save_weights('./size/decoder_mnist_weights_{}.h5'.format(s))

ch_layer = Check_layer()
callbacks=[ch_layer]

# autoencoderの実行
vae.fit(x_train_canny,x_train_gray,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(x_test_canny, x_test_gray),
       callbacks=callbacks
       )
      
# 学習結果の保存
vae.save('./size/ae_cifar10_size.h5')
encoder.save_weights('./size/encoder_mnist_weights.h5')
decoder.save_weights('./size/decoder_mnist_weights.h5')

# json and weights
model_json = vae.to_json()
with open('ae_cifar10.json', 'w') as json_file:
    json_file.write(model_json)
