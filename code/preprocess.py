import os
import random
import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow_datasets as tfds
import hyperparameters as hp


class Datasets():
    """ Class for containing the training and test sets as well as
    other useful data-related information. Contains the functions
    for preprocessing.
    """

    def __init__(self, task):

        self.task = task

        train_ds, val_ds, test_ds  = tfds.load('nyu_depth_v2', split=['train[:10%]', 'validation[:20%]','train[-5:]'], data_dir='data', as_supervised=True, batch_size=hp.batch_size)
       
        self.train_data = self.get_data(
            train_ds, True, True)
        self.val_data = self.get_data(
            val_ds, False, True)
        self.test_data = self.get_data(
            test_ds, False, True)
        
    
    
    def get_data(self, dataset, shuffle, augment):
        AUTOTUNE = tf.data.AUTOTUNE
        resize = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.Resizing(hp.img_size, hp.img_size),
            
        ])
        
        dataset = dataset.map(tf.autograph.experimental.do_not_convert(lambda x, y: (resize(x), resize(tf.expand_dims(y, -1)))),num_parallel_calls=AUTOTUNE)
        
        if shuffle:
            dataset = dataset.shuffle(10)
        
        
        


        return dataset.prefetch(buffer_size=AUTOTUNE)
