

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

# build decoder model
# decoder
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(7 * 7 * 64, activation='relu')(latent_inputs)
x = Reshape((7, 7, 64))(x)
x = Conv2DTranspose(64, (3, 3), activation='relu', strides=2, padding='same')(x)
x = Conv2DTranspose(32, (3, 3), activation='relu', strides=2, padding='same')(x)
outputs = Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='same')(x)

# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()
decoder.load_weights('./mnist1000/3/decoder_mnist_weightsL2_20.h5')

def plot_results(z0,z1,i):
    z_sample=np.array([[np.array(z0),np.array(z1)]])
    x_decoded = decoder.predict(z_sample)
    fig=plt.figure()
    ax=fig.add_subplot(111)
    size=(280,280)
    imgs=x_decoded.reshape(28, 28)
    imgs=cv2.resize(imgs, size,interpolation = cv2.INTER_CUBIC)
    
    cv2.imshow("z_sample=[{0:6.3f},{1:6.3f}]".format(z0,z1),imgs)
    ax.set_title("z_sample=[{0:6.3f},{1:6.3f}]".format(z0,z1))
    ax.imshow(imgs)
    plt.savefig("./mnist1000/3/3_[{0:6.3f},{1:6.3f}].png".format(z0,z1))
    
    #cv2.imwrite("./mnist1000/3/3_z[{0:6.3f},{1:6.3f}].jpg".format(z0,z1),imgs)    

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
        