from __future__ import print_function

import keras
from keras.datasets import mnist, cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.layers import Lambda, Input, Dense, Conv2DTranspose
from keras.models import Model
from keras.layers import *
import matplotlib.pyplot as plt

batch_size = 128
num_classes = 10
epochs = 100

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
s=9
x_train = x_train.reshape(50000, 32, 32,3 ) #28,28,1)
x_test = x_test.reshape(10000, 32, 32,3 ) #28,28,1)
x_train = x_train.astype('float32')/255.
x_test = x_test.astype('float32')/255.
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
# convert class vectors to binary class matrices
#y_train = keras.utils.to_categorical(y_train, num_classes)
#y_test = keras.utils.to_categorical(y_test, num_classes)
x_train1=[]
y_train1=[]
x_test1=[]
y_test1=[]
for i in range(len(x_train)):
    if y_train[i]==s:
        img = x_train[i]  #[y_train==s] #7_8_4
        x_train1.append(img)
        img = y_train[i]  #[y_train==s] #7_8_4
        y_train1.append(img)
for i in range(len(x_test)):
    if y_test[i]==s:
        img = x_test[i] #[y_test==s] #7_8_4
        x_test1.append(img)
        img = y_test[i] #[y_test==s] #7_8_4
        y_test1.append(img)
x_train1=np.array(x_train1)
x_test1=np.array(x_test1)
print(x_train1.shape[0], 'train samples')
print(x_test1.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train1 = keras.utils.to_categorical(y_train1, num_classes)
y_test1 = keras.utils.to_categorical(y_test1, num_classes)


def model_cat(input_image=Input(shape=(32, 32,3 ))): #28,28,1))):
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

model = model_cat(input_image=Input(shape=(32, 32,3 ))) #28,28,1)))
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

#model.load_weights("./category/cifar10/cat_mnist_weights_"+str(s)+".h5")  #s

history = model.fit(x_train1, y_train1,
                    batch_size=batch_size,
                    epochs=5,
                    verbose=1,
                    validation_data=(x_test1, y_test1))

model.save_weights("./category/cifar10/cat_mnist_weights_"+str(s)+".h5")

#score = model.evaluate(x_test1, y_test1, verbose=0)
#print('Test loss:', score[0])
#print('Test accuracy:', score[1])
img=x_test1[0]
cat=model.predict(img.reshape(1,32, 32,3 ))  #28,28,1))
plt.imshow(img.reshape(32,32,3))  #28,28))
plt.pause(1)
plt.close()
s=np.argmax(cat)
print(np.argmax(s),np.argmax(y_test1[0]))
"""
for i in range(10):
    cat=model.predict(x_test1[i].reshape(1,32, 32,3 ))  #28,28,1))
    plt.imshow(x_test1[i].reshape(32, 32,3 )) #28,28))
    plt.pause(1)
    print(np.argmax(cat),np.argmax(y_test1[i]))
"""

def encoder_decoder_model(input_image=Input(shape=(32, 32,3 ))):  #28,28,1))):
    x = Conv2D(32, (3, 3), activation='relu', strides=2, padding='same')(input_image)
    x = Conv2D(64, (3, 3), activation='relu', strides=2, padding='same')(x)
    x = Flatten()(x)
    #z_mean = Dense(latent_dim, name='z_mean')(x)
    # build decoder model
    #x = Dense(8 * 8 * 64, activation='relu')(x)  #(latent_inputs)
    x = Reshape((8, 8, 64))(x)
    x = Conv2DTranspose(64, (3, 3), activation='relu', strides=2, padding='same')(x)
    x = Conv2DTranspose(32, (3, 3), activation='relu', strides=2, padding='same')(x)
    outputs = Conv2DTranspose(3, (3, 3), activation='sigmoid', padding='same')(x)  #1
    return Model(input_image, outputs)

item='decoder'
#latent_dim=2
encoder_decoder = encoder_decoder_model(input_image=Input(shape=(32, 32,3 )))  #28,28,1)))
encoder_decoder.summary()

encoder_decoder.compile(loss='binary_crossentropy',optimizer='adam')

#encoder_decoder.load_weights("./category/cifar10/encoder_decoder_mnist_weights_"+str(s)+".h5")

history = encoder_decoder.fit(x_train1, x_train1,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test1, x_test1))

encoder_decoder.save_weights("./category/cifar10/encoder_decoder_mnist_weights_"+str(s)+".h5")

#score = encoder_decoder.evaluate(x_test1, x_test1, verbose=0)
#print('Test loss:', score[0])
#print('Test accuracy:', score[1])

cat=encoder_decoder.predict(img.reshape(1,32, 32,3 )) #/255   #28,28,1))/255
plt.imshow(img.reshape(32, 32,3 ))  #28,28))
plt.pause(1)
plt.imshow(cat.reshape(32, 32,3 ))  #28,28))
plt.pause(1)
print(np.argmax(y_test1[0]))

"""
for i in range(10):
    cat=encoder_decoder.predict(x_test1[i].reshape(1,28,28,1))/255
    plt.imshow(x_test1[i].reshape(28,28))
    plt.pause(1)
    plt.imshow(cat.reshape(28,28))
    plt.pause(1)
    print(np.argmax(y_test1[i]))
"""    