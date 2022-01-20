import tensorflow as tf
from tensorflow.keras.layers import \
    Conv2D, MaxPool2D, Dropout, Flatten, Dense, Activation, BatchNormalization,Reshape,Conv2DTranspose,UpSampling2D

import hyperparameters as hp

class YourModel1(tf.keras.Model):
    """ Your own neural network model. """

    def __init__(self):
        super(YourModel1, self).__init__()

      

        self.optimizer = 'adam'


        
        self.architecture = [
              Conv2D(64, 3, 1, padding="same",activation="relu", name="block1_conv1"),
              MaxPool2D(2, name="block1_pool"),
              Conv2D(128, 3, 1, padding="same",activation="relu", name="block2_conv1"),
              MaxPool2D(2, name="block2_pool"),
              Conv2D(256, 3, 1, padding="same",activation="relu", name="block3_conv1"),
              MaxPool2D(2, name="block3_pool"),
              Conv2D(512, 3, 1, padding="same",activation="relu", name="block4_conv1"),
              MaxPool2D(2, name="block4_pool"),

              Dropout(.3),
              Activation('relu'),
              BatchNormalization(),
              Conv2D(512, 3, 1, padding="same",activation="relu"),
              UpSampling2D(2),
              Conv2D(256, 3, 1, padding="same",activation="relu"),
              UpSampling2D(2),
              Conv2D(128, 3, 1, padding="same",activation="relu"),
              UpSampling2D(2),
              Conv2D(64, 3, 1, padding="same",activation="relu"),
              UpSampling2D(2),
              Conv2D(1, 3, 1, padding="same",activation="relu"),
              #Dense(1,activation="relu"),
       ]

        self.architecture = tf.keras.Sequential(self.architecture, name="your_model1")
       
    def call(self, x):
        """ Passes input image through the network. """

        x = self.architecture(x)

        return x

    @staticmethod
    def loss_fn(labels, predictions):
        """ Loss function for the model. """

        
        loss = tf.keras.losses.MeanSquaredError()
        return loss(labels, predictions)

class YourModel2(tf.keras.Model):
    """ Your own neural network model. """

    def __init__(self):
        super(YourModel2, self).__init__()

        

        self.optimizer = 'adam'


        self.architecture = [
              Conv2D(64, 5, 1, padding="same",activation="relu", name="block1_conv1"),
              MaxPool2D(2, name="block1_pool"),
              BatchNormalization(),
              Conv2D(128, 3, 1, padding="same",activation="relu", name="block2_conv1"),
              MaxPool2D(2, name="block2_pool"),
              BatchNormalization(),
              Conv2D(256, 3, 1, padding="same",activation="relu", name="block3_conv1"),
              MaxPool2D(2, name="block3_pool"),
              BatchNormalization(),
              Conv2D(512, 3, 1, padding="same",activation="relu", name="block4_conv1"),
              MaxPool2D(2, name="block4_pool"),
              BatchNormalization(),
              Conv2D(1024, 3, 1, padding="same",activation="relu", name="block5_conv1"),
              MaxPool2D(2, name="block5_pool"),
              
              
              Dropout(0.3),
              Activation('relu'),
              BatchNormalization(),

              Conv2D(512, 3, 1, padding="same",activation="relu"),
              UpSampling2D(2),
              Conv2D(512, 3, 1, padding="same",activation="relu"),
              UpSampling2D(2),
              Conv2D(256, 3, 1, padding="same",activation="relu"),
              UpSampling2D(2),
              Conv2D(128, 3, 1, padding="same",activation="relu"),
              UpSampling2D(2),
              Conv2D(64, 3, 1, padding="same",activation="relu"),
              UpSampling2D(2),
              Conv2D(1, 3, 1, padding="same",activation="relu"),
              #Dense(1,activation="relu"),
       ]

        self.architecture = tf.keras.Sequential(self.architecture, name="your_model2")
       
    def call(self, x):
        """ Passes input image through the network. """

        x = self.architecture(x)

        return x

    @staticmethod
    def loss_fn(labels, predictions):
        """ Loss function for the model. """

      
        loss = tf.keras.losses.MeanSquaredError()
        return loss(labels, predictions)
        


class VGGModel(tf.keras.Model):
    def __init__(self):
        super(VGGModel, self).__init__()

        

        self.optimizer = 'adam'

        # Don't change the below:

        self.vgg16 = [
            # Block 1
            Conv2D(64, 3, 1, padding="same",
                   activation="relu", name="block1_conv1"),
            Conv2D(64, 3, 1, padding="same",
                   activation="relu", name="block1_conv2"),
            MaxPool2D(2, name="block1_pool"),
            # Block 2
            Conv2D(128, 3, 1, padding="same",
                   activation="relu", name="block2_conv1"),
            Conv2D(128, 3, 1, padding="same",
                   activation="relu", name="block2_conv2"),
            MaxPool2D(2, name="block2_pool"),
            # Block 3
            Conv2D(256, 3, 1, padding="same",
                   activation="relu", name="block3_conv1"),
            Conv2D(256, 3, 1, padding="same",
                   activation="relu", name="block3_conv2"),
            Conv2D(256, 3, 1, padding="same",
                   activation="relu", name="block3_conv3"),
            MaxPool2D(2, name="block3_pool"),
            # Block 4
            Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block4_conv1"),
            Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block4_conv2"),
            Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block4_conv3"),
            MaxPool2D(2, name="block4_pool"),
            # Block 5
            Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block5_conv1"),
            Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block5_conv2"),
            Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block5_conv3"),
            MaxPool2D(2, name="block5_pool")
        ]

        
        for layer in self.vgg16:
            layer.trainable = False
        

        self.head = [
       
            Dropout(0.3),
            Activation('relu'),
            BatchNormalization(),

            Conv2D(512, 3, 1, padding="same",activation="relu"),
            UpSampling2D(2),
            Conv2D(512, 3, 1, padding="same",activation="relu"),
            UpSampling2D(2),
            Conv2D(256, 3, 1, padding="same",activation="relu"),
            UpSampling2D(2),
            Conv2D(128, 3, 1, padding="same",activation="relu"),
            UpSampling2D(2),
            Conv2D(64, 3, 1, padding="same",activation="relu"),
            UpSampling2D(2),
            Conv2D(1, 3, 1, padding="same",activation="relu"),
            #Dense(1,activation="relu"),
            
        ]

        # Don't change the below:
        self.vgg16 = tf.keras.Sequential(self.vgg16, name="vgg_base")
        self.head = tf.keras.Sequential(self.head, name="vgg_head")

    def call(self, x):
        """ Passes the image through the network. """

        x = self.vgg16(x)
        x = self.head(x)

        return x

    @staticmethod
    def loss_fn(labels, predictions):
        """ Loss function for model. """

        loss = tf.keras.losses.MeanSquaredError()
        return loss(labels, predictions)
        
