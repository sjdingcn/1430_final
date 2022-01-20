import os
import sys
import argparse
import re
from datetime import datetime
import tensorflow as tf

import hyperparameters as hp
from models import YourModel1, VGGModel, YourModel2
from preprocess import Datasets
from skimage.transform import resize
from tensorboard_utils import \
        ImageLabelingLogger, ConfusionMatrixLogger, CustomModelSaver

from skimage.io import imread
from PIL import Image
from lime import lime_image
from skimage.segmentation import mark_boundaries
from matplotlib import pyplot as plt
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def parse_args():
    """ Perform command-line argument parsing. """

    parser = argparse.ArgumentParser(
        description="Let's train some neural nets!",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--task',
        required=True,
        choices=['1', '2', '3'],
        help='''Which task of the assignment to run -
        training from scratch (1), or fine tuning VGG-16 (3).''')

    parser.add_argument(
        '--load-vgg',
        default='vgg16_imagenet.h5',
        help='''Path to pre-trained VGG-16 file (only applicable to
        task 3).''')
    parser.add_argument(
        '--load-checkpoint',
        default=None,
        help='''Path to model checkpoint file (should end with the
        extension .h5). Checkpoints are automatically saved when you
        train your model. If you want to continue training from where
        you left off, this is how you would load your weights.''')
    parser.add_argument(
        '--confusion',
        action='store_true',
        help='''Log a confusion matrix at the end of each
        epoch (viewable in Tensorboard). This is turned off
        by default as it takes a little bit of time to complete.''')
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='''Skips training and evaluates on the test set once.
        You can use this to test an already trained model by loading
        its checkpoint.''')


    return parser.parse_args()


def analyzer(model, dataset):
    

    image, depth = next(iter(dataset))
    prediction = model.predict(image)
    
    if len(image.shape) == 2:
        image = np.stack([image, image, image], axis=-1)

    
    image = resize(image[1], (hp.img_size, hp.img_size))
    depth = resize(depth[1], (hp.img_size, hp.img_size))/10
    prediction = resize(prediction[1], (hp.img_size, hp.img_size))/10

    cm = plt.get_cmap('gist_rainbow')
    depth_cm = cm(depth).squeeze()
    prediction_cm = cm(prediction).squeeze()

    Image.fromarray(image.astype(np.uint8)).save("results/image.png")
    Image.fromarray(depth_cm.astype(np.uint8)*255).save('results/depth.png')
    Image.fromarray(prediction_cm.astype(np.uint8)*255).save("results/prediction.png")

    
    _, ax = plt.subplots(1,3)
    
    ax[0].imshow(image/255)
    ax[0].set_title("RGB image")
    ax[1].imshow(prediction_cm)
    ax[1].set_title("predition")
    ax[2].imshow(depth_cm)
    ax[2].set_title("ground truth")
    
    ax[0].axis('off')
    ax[1].axis('off')
    ax[2].axis('off')
    plt.savefig('results/analyze.png')



def train(model, datasets, checkpoint_path, logs_path, init_epoch):
    """ Training routine. """

    # Keras callbacks for training
    callback_list = [
        tf.keras.callbacks.TensorBoard(
            log_dir=logs_path,
            update_freq='batch',
            profile_batch=0),
        ImageLabelingLogger(logs_path, datasets),
        CustomModelSaver(checkpoint_path, ARGS.task, hp.max_num_weights)
    ]

    # Include confusion logger in callbacks if flag set
    if ARGS.confusion:
        callback_list.append(ConfusionMatrixLogger(logs_path, datasets))

    # Begin training
    model.fit(
        x=datasets.train_data,
        validation_data=datasets.val_data,
        epochs=hp.num_epochs,
        batch_size=None,
        callbacks=callback_list,
        initial_epoch=init_epoch,
    )


def test(model, test_data):
    """ Testing routine. """

    # Run model on test set
    model.evaluate(
        x=test_data,
        verbose=1,
    )


def main():
    """ Main function. """

    time_now = datetime.now()
    timestamp = time_now.strftime("%m%d%y-%H%M%S")
    init_epoch = 0

    # If loading from a checkpoint, the loaded checkpoint's directory
    # will be used for future checkpoints
    if ARGS.load_checkpoint is not None:
        ARGS.load_checkpoint = os.path.abspath(ARGS.load_checkpoint)

        # Get timestamp and epoch from filename
        regex = r"(?:.+)(?:\.e)(\d+)(?:.+)(?:.h5)"
        init_epoch = int(re.match(regex, ARGS.load_checkpoint).group(1)) + 1
        timestamp = os.path.basename(os.path.dirname(ARGS.load_checkpoint))

    # If paths provided by program arguments are accurate, then this will
    # ensure they are used. If not, these directories/files will be
    # set relative to the directory of run.py
    # if os.path.exists(ARGS.data):
    #     ARGS.data = os.path.abspath(ARGS.data)
    if os.path.exists(ARGS.load_vgg):
        ARGS.load_vgg = os.path.abspath(ARGS.load_vgg)

    # Run script from location of run.py
    os.chdir(sys.path[0])

    datasets = Datasets(ARGS.task)

    if ARGS.task == '1':
        model = YourModel1()
        model(tf.keras.Input(shape=(hp.img_size, hp.img_size, 3)))
        checkpoint_path = "checkpoints" + os.sep + \
            "your_model1" + os.sep + timestamp + os.sep
        logs_path = "logs" + os.sep + "your_model1" + \
            os.sep + timestamp + os.sep

        # Print summary of model
        model.summary()
    elif ARGS.task == '2':
        model = YourModel2()
        model(tf.keras.Input(shape=(hp.img_size, hp.img_size, 3)))
        checkpoint_path = "checkpoints" + os.sep + \
            "your_model2" + os.sep + timestamp + os.sep
        logs_path = "logs" + os.sep + "your_model2" + \
            os.sep + timestamp + os.sep

        # Print summary of model
        model.summary()
    else:
        model = VGGModel()
        checkpoint_path = "checkpoints" + os.sep + \
            "vgg_model" + os.sep + timestamp + os.sep
        logs_path = "logs" + os.sep + "vgg_model" + \
            os.sep + timestamp + os.sep
        model(tf.keras.Input(shape=(224, 224, 3)))

        # Print summaries for both parts of the model
        model.vgg16.summary()
        model.head.summary()

        # Load base of VGG model
        model.vgg16.load_weights(ARGS.load_vgg, by_name=True)

    # Load checkpoints
    if ARGS.load_checkpoint is not None:
        if ARGS.task == '1' or ARGS.task == '2':
            model.load_weights(ARGS.load_checkpoint, by_name=False)
        else:
            model.head.load_weights(ARGS.load_checkpoint, by_name=False)

    # Make checkpoint directory if needed
    if not ARGS.evaluate and not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    # Compile model graph
    model.compile(
        optimizer=model.optimizer,
        loss=model.loss_fn,
        metrics=["accuracy"])

    if ARGS.evaluate:
        test(model, datasets.test_data)

        
        analyzer(model, datasets.test_data)
    else:
        train(model, datasets, checkpoint_path, logs_path, init_epoch)


# Make arguments global
ARGS = parse_args()

main()
