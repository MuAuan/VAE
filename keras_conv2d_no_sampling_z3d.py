#https://keras.io/examples/variational_autoencoder/

from keras.layers import Input, Conv2DTranspose
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras import backend as K

import keras
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import cv2

from keras.layers import Conv2D, Dense
from keras.layers import Activation, Flatten
from keras.layers import Reshape, Embedding,InputLayer
#for 3D plotting
from mpl_toolkits.mplot3d import Axes3D

def plot_results(models,
                 data,
                 batch_size=32,
                 model_name="vae_mnist",epochs=0):

    encoder, decoder = models
    x_test, y_test = data
    os.makedirs(model_name, exist_ok=True)

    filename1 = "./mnist1000/z3d/vae_mean_all"
    # display a 2D plot of the digit classes in the latent space
    z_mean = encoder.predict(x_test, batch_size=batch_size)
    fig = plt.figure(figsize=(16, 12))
    ax1 = fig.add_subplot(111 , projection='3d')
    sc = ax1.scatter(z_mean[:, 0], z_mean[:, 1], z_mean[:, 2], c=y_test,s=50,cmap=plt.cm.jet)
    plt.colorbar(sc)
    ax1.set_xlabel("z[0]")
    ax1.set_ylabel("z[1]")
    ax1.set_zlabel("z[2]")
    plt.savefig(filename1+"z0z1z2"+str(epochs)+".png")
    plt.show()
    plt.close()

    filename2 = "./mnist1000/z3d/digits_over_latent_all"
    # display a 30x30 2D manifold of digits
    n = 10
    digit_size = 28
    #figure = np.zeros((digit_size * n, digit_size * n))
    
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x_min,grid_x_max=np.min(z_mean[:, 0]),np.max(z_mean[:, 0])
    grid_y_min,grid_y_max=np.min(z_mean[:, 1]),np.max(z_mean[:, 1])
    grid_x = np.linspace(grid_x_min,grid_x_max, n)   #(-4, 4, n)
    grid_y = np.linspace(grid_y_min,grid_y_max, n)[::-1]  #(-4, 4, n)[::-1]
    fig = plt.figure(figsize=(16,8))
    for z0 in range(-16,16,1):
        zi=z0/4
        p=z0+17
        ax=fig.add_subplot(4,8,p)
        figure = np.zeros((digit_size * n, digit_size * n))
        for i, yi in enumerate(grid_y):
            for j, xi in enumerate(grid_x):
                z_sample = np.array([[xi, yi,zi]])
                x_decoded = decoder.predict(z_sample)
                digit = x_decoded[0].reshape(digit_size, digit_size)
                figure[i * digit_size: (i + 1) * digit_size,
                       j * digit_size: (j + 1) * digit_size] = digit
        ax.axis("off")
        ax.set_title("z0={0:}".format(z0))
        ax.imshow(figure, cmap='Greys_r')
        plt.axis("off")
    plt.savefig(filename2+str(epochs)+".png")
    plt.pause(1)
    plt.close()
    

# MNIST dataset
#(x_train, _), (x_test, _) = mnist.load_data()
(x_train, y_train), (x_test, y_test) = mnist.load_data()
image_size = x_train.shape[1]
original_dim = image_size * image_size
x_train = x_train[:60000].astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train[:60000], (len(x_train[:60000]), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1)) 
y_train=y_train[:60000]


# network parameters
#input_shape = (original_dim, )
input_shape = (image_size, image_size, 1)
batch_size = 128
latent_dim = 3
epochs = 101

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
x_train1 = x_train  #[y_train==7]
x_test1 = x_test  #[y_test==7]

class Check_layer(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if epoch%10==0:
            vae.save_weights("./mnist1000/z3d/vae_mnist_weightsL2_"+str(epoch)+"_1000.h5")
            encoder.save_weights("./mnist1000/z3d/encoder_mnist_weightsL2_"+str(epoch)+"_1000.h5")
            decoder.save_weights("./mnist1000/z3d/decoder_mnist_weightsL2_"+str(epoch)+"_1000.h5")
            models=(encoder,decoder)
            data = (x_test, y_test)
            plot_results(models, data, batch_size=32, model_name="vae_mnist",epochs=epoch)
        
ch_layer = Check_layer()
callbacks = [ch_layer]

# vaeの学習
vae.fit(x_train1,x_train1,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                validation_data=(x_test1, x_test1))

encoder.load_weights("./mnist1000/z3d/encoder_mnist_weightsL2_"+str(100)+"_1000.h5")
decoder.load_weights("./mnist1000/z3d/decoder_mnist_weightsL2_"+str(100)+"_1000.h5")

models = (encoder, decoder)
data = (x_test, y_test)

plot_results(models,
             data,
             batch_size=batch_size,
             model_name="vae_mlp")

# 実行結果の表示
n = 10
decoded_imgs = vae.predict(x_test[:n])

plt.figure(figsize=(10, 4))
for i in range(n):
    # original_image
    orig_img = x_test[i].reshape(image_size, image_size)

    # reconstructed_image
    reconst_img = decoded_imgs[i].reshape(image_size, image_size)

    # diff image
    diff_img = ((orig_img - reconst_img)+2)/4
    diff_img = (diff_img*255).astype(np.uint8)
    orig_img = (orig_img*255).astype(np.uint8)
    reconst_img = (reconst_img*255).astype(np.uint8)
    diff_img_color = cv2.applyColorMap(diff_img, cv2.COLORMAP_JET)

    # display original
    ax = plt.subplot(3, n,  i + 1)
    plt.imshow(orig_img, cmap=plt.cm.gray)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(3, n, i + n + 1)
    plt.imshow(reconst_img, cmap=plt.cm.gray)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display diff
    ax = plt.subplot(3, n, i + n*2 + 1)
    plt.imshow(diff_img, cmap=plt.cm.jet)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.savefig("./mnist1000/z3d/autodetect_all.jpg")
plt.pause(1)
plt.close()
