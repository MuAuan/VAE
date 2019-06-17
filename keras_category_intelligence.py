'''Trains a simple deep NN on the MNIST dataset.
Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function

import keras
from keras.datasets import mnist
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
epochs = 5

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
s=10
x_train = x_train.reshape(60000, 28,28,1)
x_test = x_test.reshape(10000, 28,28,1)
x_train = x_train.astype('float32')/255.
x_test = x_test.astype('float32')/255.
#x_train /= 255
#x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
x_train1 = x_train #[y_train==s] #7_8_4
x_test1 = x_test #[y_test==s] #7_8_4
y_train1 = y_train #[y_train==s] #7_8_4
y_test1 = y_test #[y_test==s] #7_8_4


# convert class vectors to binary class matrices
y_train1 = keras.utils.to_categorical(y_train1, num_classes)
y_test1 = keras.utils.to_categorical(y_test1, num_classes)


def model_cat(input_image=Input(shape=(28,28,1))):
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

latent_dim=2
model = model_cat(input_image=Input(shape=(28,28,1)))
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

model.load_weights("./category/cat_mnist_weights_"+str(10)+".h5")  #s

def encoder_decoder_model(input_image=Input(shape=(28,28,1))):
    x = Conv2D(32, (3, 3), activation='relu', strides=2, padding='same')(input_image)
    x = Conv2D(64, (3, 3), activation='relu', strides=2, padding='same')(x)
    x = Flatten()(x)
    #z_mean = Dense(latent_dim, name='z_mean')(x)
    # build decoder model
    #x = Dense(7 * 7 * 64, activation='relu')(x)  #(latent_inputs)
    x = Reshape((7, 7, 64))(x)
    x = Conv2DTranspose(64, (3, 3), activation='relu', strides=2, padding='same')(x)
    x = Conv2DTranspose(32, (3, 3), activation='relu', strides=2, padding='same')(x)
    outputs = Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='same')(x)
    return Model(input_image, outputs)

item='decoder'
#latent_dim=2
encoder_decoder = encoder_decoder_model(input_image=Input(shape=(28,28,1)))
#encoder_decoder.summary()

encoder_decoder.compile(loss='binary_crossentropy',optimizer='adam')

#decoder=("decoder0","decoder1","decoder2","decoder3","decoder4","decoder5","decoder6","decoder7","decoder8","decoder9")
encoder_decoders=[]
for s in range(10):
    encoder_decoder.load_weights("./category/encoder_decoder_mnist_weights_"+str(s)+".h5")
    encoder_decoders.append(encoder_decoder)

encoder_decoders[0].summary()

fig=plt.figure(figsize=(32, 8))
size=(280,280)
n=10
for i in range(n):
    img=x_test1[i]
    cat=model.predict(img.reshape(1,28,28,1))
    s=np.argmax(cat)
    cat=encoder_decoders[s].predict(img.reshape(1,28,28,1))/255
    ax=fig.add_subplot(3,n,i+1)
    ax.set_title("cat_"+str(s),size=40)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    img=cv2.resize(img.reshape(28,28), size,interpolation = cv2.INTER_CUBIC) 

    ax.imshow(img)
    ax=fig.add_subplot(3,n,i+n+1)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    cat=cv2.resize(cat.reshape(28,28), size,interpolation = cv2.INTER_CUBIC) 
    ax.imshow(cat)
    orig_img=img
    reconst_img=cat
    diff_img = ((orig_img - reconst_img)+2)/4
    #diff_img_color = cv2.applyColorMap(diff_img, cv2.COLORMAP_JET)
    ax = plt.subplot(3, n, i + n*2 + 1)
    #diff_img=cv2.resize(diff_img.reshape(28,28), size,interpolation = cv2.INTER_CUBIC) 
    plt.imshow(diff_img, cmap=plt.cm.jet)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.savefig("./category/autodetect_mnist.jpg")

plt.show()
plt.close()
