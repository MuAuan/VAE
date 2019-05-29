

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

input_shape = (28, 28, 1)
latent_dim = 2

def model(latent_dim, name='decoder1'):
    # build decoder model
    # decoder
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = Dense(7 * 7 * 64, activation='relu')(latent_inputs)
    x = Reshape((7, 7, 64))(x)
    x = Conv2DTranspose(64, (3, 3), activation='relu', strides=2, padding='same')(x)
    x = Conv2DTranspose(32, (3, 3), activation='relu', strides=2, padding='same')(x)
    outputs = Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='same')(x)
    return Model(latent_inputs, outputs, name=name)

# instantiate decoder model
decoder0 = model(latent_dim, name='decoder0')
decoder1 = model(latent_dim, name='decoder1')
decoder2 = model(latent_dim, name='decoder2')
decoder3 = model(latent_dim, name='decoder3')
decoder4 = model(latent_dim, name='decoder4')
decoder5 = model(latent_dim, name='decoder5')
decoder6 = model(latent_dim, name='decoder6')
decoder7 = model(latent_dim, name='decoder7')
decoder8 = model(latent_dim, name='decoder8')
decoder9 = model(latent_dim, name='decoder9')
decoder1.summary()
decoder0.load_weights('./mnist1000/0/decoder_mnist_weightsL2_20.h5')
decoder1.load_weights('./mnist1000/1/decoder_mnist_weightsL2_20.h5')
decoder2.load_weights('./mnist1000/2/decoder_mnist_weightsL2_20.h5')
decoder3.load_weights('./mnist1000/3/decoder_mnist_weightsL2_20.h5')
decoder4.load_weights('./mnist1000/4/decoder_mnist_weightsL2_20.h5')
decoder5.load_weights('./mnist1000/5/decoder_mnist_weightsL2_20.h5')
decoder6.load_weights('./mnist1000/6/decoder_mnist_weightsL2_20.h5')
decoder7.load_weights('./mnist1000/7/decoder_mnist_weightsL2_20.h5')
decoder8.load_weights('./mnist1000/8/decoder_mnist_weightsL2_20.h5')
decoder9.load_weights('./mnist1000/9/decoder_mnist_weightsL2_20.h5')

def plot_results(z0,z1,i):
    z_sample=np.array([[np.array(z0),np.array(z1)]])
    x_decoded0 = decoder0.predict(z_sample)    
    x_decoded1 = decoder1.predict(z_sample)
    x_decoded2 = decoder2.predict(z_sample)
    x_decoded3 = decoder3.predict(z_sample)
    x_decoded4 = decoder4.predict(z_sample)
    x_decoded5 = decoder5.predict(z_sample)
    x_decoded6 = decoder6.predict(z_sample)
    x_decoded7 = decoder7.predict(z_sample)
    x_decoded8 = decoder8.predict(z_sample)
    x_decoded9 = decoder9.predict(z_sample)        
    fig=plt.figure()
    ax0=fig.add_subplot(2,5,1)    
    ax1=fig.add_subplot(2,5,2)
    ax2=fig.add_subplot(2,5,3)
    ax3=fig.add_subplot(2,5,4)
    ax4=fig.add_subplot(2,5,5)
    ax5=fig.add_subplot(2,5,6)
    ax6=fig.add_subplot(2,5,7)
    ax7=fig.add_subplot(2,5,8)
    ax8=fig.add_subplot(2,5,9)
    ax9=fig.add_subplot(2,5,10)    
    size=(280,280)
    imgs0=x_decoded0.reshape(28, 28)
    imgs1=x_decoded1.reshape(28, 28)
    imgs2=x_decoded2.reshape(28, 28)
    imgs3=x_decoded3.reshape(28, 28)
    imgs4=x_decoded4.reshape(28, 28)
    imgs5=x_decoded5.reshape(28, 28)
    imgs6=x_decoded6.reshape(28, 28)
    imgs7=x_decoded7.reshape(28, 28)
    imgs8=x_decoded8.reshape(28, 28)
    imgs9=x_decoded9.reshape(28, 28)
    imgs0=cv2.resize(imgs0, size,interpolation = cv2.INTER_CUBIC)    
    imgs1=cv2.resize(imgs1, size,interpolation = cv2.INTER_CUBIC)
    imgs2=cv2.resize(imgs2, size,interpolation = cv2.INTER_CUBIC)
    imgs3=cv2.resize(imgs3, size,interpolation = cv2.INTER_CUBIC)
    imgs4=cv2.resize(imgs4, size,interpolation = cv2.INTER_CUBIC)
    imgs5=cv2.resize(imgs5, size,interpolation = cv2.INTER_CUBIC)
    imgs6=cv2.resize(imgs6, size,interpolation = cv2.INTER_CUBIC)
    imgs7=cv2.resize(imgs7, size,interpolation = cv2.INTER_CUBIC)
    imgs8=cv2.resize(imgs8, size,interpolation = cv2.INTER_CUBIC)
    imgs9=cv2.resize(imgs9, size,interpolation = cv2.INTER_CUBIC)

    cv2.imshow("z_sample=[{0:6.3f},{1:6.3f}]".format(z0,z1),imgs0)
    cv2.imshow("z_sample=[{0:6.3f},{1:6.3f}]".format(z0,z1),imgs1)
    cv2.imshow("z_sample=[{0:6.3f},{1:6.3f}]".format(z0,z1),imgs2)
    cv2.imshow("z_sample=[{0:6.3f},{1:6.3f}]".format(z0,z1),imgs3)
    cv2.imshow("z_sample=[{0:6.3f},{1:6.3f}]".format(z0,z1),imgs4)
    cv2.imshow("z_sample=[{0:6.3f},{1:6.3f}]".format(z0,z1),imgs5)
    cv2.imshow("z_sample=[{0:6.3f},{1:6.3f}]".format(z0,z1),imgs6)
    cv2.imshow("z_sample=[{0:6.3f},{1:6.3f}]".format(z0,z1),imgs7)
    cv2.imshow("z_sample=[{0:6.3f},{1:6.3f}]".format(z0,z1),imgs8)    
    cv2.imshow("z_sample=[{0:6.3f},{1:6.3f}]".format(z0,z1),imgs9)
    ax1.set_title("z_sample=[{0:6.3f},{1:6.3f}]".format(z0,z1))
    ax0.imshow(imgs0)
    ax1.imshow(imgs1)
    ax2.imshow(imgs2)
    ax3.imshow(imgs3)
    ax4.imshow(imgs4)
    ax5.imshow(imgs5)
    ax6.imshow(imgs6)
    ax7.imshow(imgs7)
    ax8.imshow(imgs8) 
    ax9.imshow(imgs9)    
    plt.savefig("./mnist1000/3/3_1_[{0:6.3f},{1:6.3f}].png".format(z0,z1))
    
    
z0=0
z1=0
s=0
while True:
    img=cv2.imread("./mnist1000/"+str(3)+"/digits_over_latent100L2_3.png")
    cv2.circle(img,(int(300+80*(z0)),int(650-70*(z1))), 10, (0,255,255), -1)
    plot_results(z0,z1,s)
    cv2.imshow('image',img)
    s+=1
    k = cv2.waitKey(0)
    if k == 27:         # wait for ESC key to exit
        cv2.destroyAllWindows()
        break
    elif k == ord('j'): # 
        cv2.imwrite('./mnist1000/3/z[{0:6.3f},{1:6.3f}].png'.format(z0,z1),img)
        z0=z0+0.1
    elif k == ord('h'): # 
        cv2.imwrite('./mnist1000/3/z[{0:6.3f},{1:6.3f}].png'.format(z0,z1),img)
        z0=z0-0.1
    elif k == ord('u'): # 
        cv2.imwrite('./mnist1000/3/z[{0:6.3f},{1:6.3f}].png'.format(z0,z1),img)
        z1=z1+0.1
    elif k == ord('n'): # wait for 's' key to save and exit
        cv2.imwrite('./mnist1000/3/z[{0:6.3f},{1:6.3f}].png'.format(z0,z1),img)
        z1=z1-0.1
        