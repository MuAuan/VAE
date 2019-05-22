from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Reshape, Embedding,InputLayer

def plot_fig(x_test, decoded_imgs, encoded_imgs,k):
    n = 10
    plt.figure(figsize=(10, 16))
    for j in range(0,n):
        for i in range(1,n+1):
            # display original
            ax1 = plt.subplot(20, n*1, i+10*2*j)
            ax1.imshow(x_test[i+10*j].reshape(28, 28))
            plt.gray()
            ax1.get_xaxis().set_visible(False)
            ax1.get_yaxis().set_visible(False)

            # display reconstruction
            ax2 = plt.subplot(20, n*1, i + (2*j+1)*10)
            ax2.imshow(decoded_imgs[i+10*j].reshape(28, 28))
            plt.gray()
            ax2.get_xaxis().set_visible(False)
            ax2.get_yaxis().set_visible(False)
    plt.savefig("./mnist1000/mnist_training_by_100_10_{}".format(k))    
    plt.pause(0.01)
    plt.close()

    n = 100
    plt.figure(figsize=(10, 16))
    for i in range(1,n+1):
        ax = plt.subplot(10, n*0.1, i)
        plt.imshow(encoded_imgs[i].reshape(8, 2 * 8).T)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.savefig("./mnist1000/mnist_intermid_training_by_100_10_{}".format(k))    
    plt.pause(0.01)
    plt.close()

input_img = Input(shape=(28, 28, 1))  # adapt this if using `channels_first` image data format

x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same',name='encoded')(x)
encoder=Model(input_img, encoded)
encoder.summary()

x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
autoencoder.summary()

from keras.datasets import mnist
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train[:1000].astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train[:1000], (len(x_train[:1000]), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))
y_train=y_train[:1000]

for j in range(10):
    x_train1 = x_train
    x_test1 = x_test

    autoencoder.fit(x_train1, x_train1,
                epochs=10,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test1, x_test1)
                )

    decoded_imgs = autoencoder.predict(x_test)
    encoded_imgs = encoder.predict(x_test)
        
    plot_fig(x_test,decoded_imgs,encoded_imgs,j)
   

