from keras.models import Sequential  #用來啟動 NN
from keras.layers import Conv2D  # Convolution Operation
from keras.layers import MaxPooling2D # Pooling
from keras.layers import Flatten
from keras.layers import Dense # Fully Connected Networks
from keras.layers import Dropout
from keras.layers import LeakyReLU
from keras.layers import BatchNormalization
import tensorflow as tf
import tensorflow_addons as tfa



class Model():
    def __init__():
        pass

    # Build CNN
    def net():
        net = Sequential()
        net.add(Conv2D(32,(3,3), strides=(2,2), activation='relu', padding="same", use_bias=False,input_shape=(15,15,1)))
        net.add(Conv2D(64,(3,3), strides=(2,2), activation='relu', padding="same", use_bias=False))
        net.add(MaxPooling2D(2, strides=2))
        net.add(Dropout(0.25))
        net.add(Flatten())
        net.add(Dense(128,activation = None))
        net.add(Dropout(0.5))
        net.add(Dense(3,activation='softmax'))
        net.compile(loss="categorical_crossentropy", optimizer = "adam", metrics=["acc"])
        return net

    def net1(lr, total_step, warmup_proportion, min_lr):
        # Ranger optimizer
        radam = tfa.optimizers.RectifiedAdam(lr=lr, total_steps=total_step, warmup_proportion=warmup_proportion, min_lr=min_lr)
        ranger = tfa.optimizers.Lookahead(radam, sync_period=6, slow_step_size=0.5)
        net1 = Sequential()
        net1.add(Conv2D(32,(3,3),strides=(2,2),activation=LeakyReLU(alpha=0.1),padding="same", use_bias=False,input_shape=(15,15,1)))
        net1.add(BatchNormalization())
        # net1.add(Dropout(0.25))
        net1.add(Conv2D(64,(3,3),strides=(2,2),activation=LeakyReLU(alpha=0.1),padding="same", use_bias=False))
        net1.add(BatchNormalization())
        net1.add(Dropout(0.25))
        net1.add(Conv2D(128,(3,3),strides=(2,2),activation=LeakyReLU(alpha=0.1),padding="same", use_bias=False))
        net1.add(BatchNormalization())
        # net1.add(Dropout(0.25))
        net1.add(MaxPooling2D(2,strides=2))
        net1.add(Dropout(0.25))
        net1.add(Flatten())
        net1.add(Dense(128,activation = None))
        net1.add(Dropout(0.5))
        net1.add(Dense(3,activation='softmax'))
        net1.compile(loss="categorical_crossentropy", optimizer = ranger, metrics=["acc"])
        return net1
    
    