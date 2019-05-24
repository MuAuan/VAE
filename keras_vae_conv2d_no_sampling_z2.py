#https://keras.io/examples/variational_autoencoder/

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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

def plot_results(models,
                 data,
                 batch_size=32,
                 model_name="vae_mnist"):
    """Plots labels and MNIST digits as function of 2-dim latent vector
    # Arguments
        models (tuple): encoder and decoder models
        data (tuple): test data and label
        batch_size (int): prediction batch size
        model_name (string): which model is using this function
    """
    encoder, decoder = models
    x_test, y_test = data
    os.makedirs(model_name, exist_ok=True)

    filename1 = "./mnist1000/vae_mean100L2_3"
    # display a 2D plot of the digit classes in the latent space
    z_mean = encoder.predict(x_test, batch_size=batch_size)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.savefig(filename1+"z0z1.png")
    plt.pause(1)
    plt.close()

    filename2 = "./mnist1000/digits_over_latent100L2_3"
    # display a 30x30 2D manifold of digits
    n = 30
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-4, 4, n)
    grid_y = np.linspace(-4, 4, n)[::-1]
    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit
    plt.figure(figsize=(10, 10))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range + 1
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap='Greys_r')
    plt.savefig(filename2+".png")
    plt.pause(0.01)
    plt.close()

def plot_results2(models,
                 data,
                 batch_size=128,
                 model_name="vae_mnist"):
    #z0=np.array([-3.7,-2.9])
    #z1=np.array([0.4,2.7])
    
    for t in range(360):
        s=t/360
        #z_sample=np.array([s*z0+(1-s)*z1])
        z_sample=np.array([[2*np.cos(2*s*np.pi),2*np.sin(2*s*np.pi)]])
        x_decoded = decoder.predict(z_sample)
        plt.imshow(x_decoded.reshape(28, 28))
        plt.title("z_sample="+str(t))
        plt.savefig('./mnist1000/360/z_sample_t{}'.format(t))
        plt.pause(0.01)
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
intermediate_dim = 512
batch_size = 128
latent_dim = 2
epochs = 20
filters = 16
kernel_size = 3

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
x_train1 = x_train  #[y_train==0] #7_8_4
x_test1 = x_test  #[y_test==0] #7_8_4
print(len(x_train1),len(x_test1))

#vae.load_weights('vae_mnist_weights_100.h5')
#encoder.load_weights('./mnist1000/encoder_mnist_weightsL2_20.h5')
#decoder.load_weights('./mnist1000/decoder_mnist_weightsL2_20.h5')

# autoencoderの実行
vae.fit(x_train1,x_train1,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(x_test1, x_test1))

models = (encoder, decoder)
data = (x_test, y_test)

plot_results(models,
             data,
             batch_size=batch_size,
             model_name="vae_mlp")

plot_results2(models,
             data,
             batch_size=batch_size,
             model_name="vae_mlp")

vae.save_weights('./mnist1000/vae_mnist_weightsL2_20.h5')
encoder.save_weights('./mnist1000/encoder_mnist_weightsL2_20.h5')
decoder.save_weights('./mnist1000/decoder_mnist_weightsL2_20.h5')

# 実行結果の表示
n = 20
decoded_imgs = vae.predict(x_test[:n])

plt.figure(figsize=(20, 4))
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
plt.savefig("./mnist1000/autodetectL48_5.jpg")
plt.pause(1)
plt.close()

# 学習結果の保存
vae.save('./ae_mnist100L256_2.h5')

# json and weights
model_json = vae.to_json()
with open('ae_mnist.json', 'w') as json_file:
    json_file.write(model_json)
vae.save_weights('ae_mnist_weights100L10_2.h5')