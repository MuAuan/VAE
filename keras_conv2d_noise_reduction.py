#https://keras.io/examples/variational_autoencoder/

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Lambda, Input, Dense, Conv2DTranspose
from keras.models import Model
from keras.datasets import mnist,cifar10
from keras.losses import mse, binary_crossentropy
from keras import backend as K
import keras

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import cv2

from keras.layers import Conv2D
from keras.models import Sequential
from keras.layers import Dropout, Activation, Flatten
from keras.layers import Reshape, Embedding,InputLayer

def plot_fig_vae(x_test,x_test_noisy, vae_imgs,k):
    n = 5
    plt.figure(figsize=(8,12 ))
    for j in range(0,n):
        for i in range(1,2*n+1):
            # display original
            ax = plt.subplot(15, 2*n*1, i+3*10*j)
            plt.imshow(x_test[i+10*j].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            # display original_gray
            ax = plt.subplot(15, 2*n*1, i+3*10*j+10)
            plt.imshow(x_test_noisy[i+10*j].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            # display reconstruction
            ax = plt.subplot(15, 2*n*1, i + 3*10*j+20)
            plt.imshow(vae_imgs[i+10*j].reshape(28,28)) #128, 128,3))
            #plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    plt.savefig("./mnist1000/noise/"+str(10)+"/cifar10_noisy_vae_0_{}.jpg".format(k))    
    plt.pause(1)
    plt.close()    
  
def plot_results(models,
                 data,
                 batch_size=32,
                 model_name="vae_mnist",
                 epochs=0):
    encoder, decoder = models
    x_test, y_test = data
    os.makedirs(model_name, exist_ok=True)

    filename1 = "./mnist1000/noise/"+str(10)+"/vae_mean_noisy"+str(epochs)
    # display a 2D plot of the digit classes in the latent space
    z_mean = encoder.predict(x_test, batch_size=batch_size)
    plt.figure(figsize=(12, 10))
    sc=plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test,s=50,cmap=plt.cm.jet)
    plt.colorbar(sc)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.savefig(filename1+"z0z1.jpg")
    plt.pause(1)
    plt.close()

def plot_irregular(x_test,x_test_noisy,decoded_imgs,epoch):
    # 実行結果の表示
    n = 10

    plt.figure(figsize=(10, 4))
    for i in range(n):
        # original_image
        orig_img0 = x_test[i].reshape(28, 28)
        orig_img = x_test_noisy[i].reshape(28, 28)

        # reconstructed_image
        reconst_img = decoded_imgs[i].reshape(28, 28)

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

        ax = plt.subplot(4, n,  i+ n*1 + 1)
        plt.imshow(orig_img, cmap=plt.cm.gray)

        # display reconstruction
        ax = plt.subplot(4, n, i + n*2 + 1)
        plt.imshow(reconst_img, cmap=plt.cm.gray)

        # display diff
        ax = plt.subplot(4, n, i + n*3 + 1)
        plt.imshow(diff_img, cmap=plt.cm.jet)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.savefig("./mnist1000/noise/10/autodetect_noise_reduction"+str(epoch)+".jpg")
    plt.pause(1)
    plt.close()
    
    
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
batch_size = 128
latent_dim = 10
epochs = 11
s=10

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

noise_train = np.random.normal(loc=0.5, scale=0.5, size=x_train.shape)
noise_test = np.random.normal(loc=0.5, scale=0.5, size=x_test.shape)
# 学習に使うデータを限定する
x_train_noisy = np.clip(x_train + noise_train, 0, 1)  #[y_train==7]
x_test_noisy = np.clip(x_test+ noise_test, 0, 1)  #[y_test==7]

class Check_layer(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if epoch%2==0:
            vae.save_weights("./mnist1000/noise/"+str(s)+"/vae_mnist_weightsL2_"+str(epoch)+"_noise.h5")
            encoder.save_weights("./mnist1000/noise/"+str(s)+"/encoder_mnist_weightsL2_"+str(epoch)+"_noise.h5")
            decoder.save_weights("./mnist1000/noise/"+str(s)+"/decoder_mnist_weightsL2_"+str(epoch)+"_noise.h5")
            vae_imgs = vae.predict(x_test_noisy)
            plot_fig_vae(x_test,x_test_noisy, vae_imgs,k=epoch)
            models=(encoder,decoder)
            data = (x_test_noisy, y_test)
            plot_results(models, data, batch_size=32, model_name="vae_mnist",epochs=epoch)
            n=10
            decoded_imgs = vae.predict(x_test_noisy[:n])
            plot_irregular(x_test,x_test_noisy,decoded_imgs,epoch=epoch)
            
        
ch_layer = Check_layer()
callbacks = [ch_layer]

# autoencoderの実行
vae.fit(x_train_noisy,x_train,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                validation_data=(x_test_noisy, x_test))

