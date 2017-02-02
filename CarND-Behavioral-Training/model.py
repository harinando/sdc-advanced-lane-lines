# Import basic
import logging

# from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.layers import Input, Dense, GlobalAveragePooling2D, Flatten,Lambda,ELU
from keras.models import Model, Sequential
from keras.regularizers import l2
import argparse


from transformations import Preprocess, Normalizer, Resize, Crop
from loader import __train_test_split, generate_batches

""" Usefeful link
		ImageDataGenerator 		- https://keras.io/preprocessing/image/
		Saving / Loading model  - http://machinelearningmastery.com/save-load-keras-deep-learning-models/
		NVIDIA					- https://arxiv.org/pdf/1604.07316v1.pdf
		Features Extraction     - https://keras.io/applications/
		ewma					- http://pandas.pydata.org/pandas-docs/version/0.17.0/generated/pandas.ewma.html
		Callbacks				- https://keras.io/callbacks/

		Dropout 5x5
"""


def NvidiaModel(learning_rate, dropout):
    input_model = Input(shape=(WIDTH, HEIGHT, DEPTH))
    x = Convolution2D(24, 5, 5, border_mode='valid', subsample=(2, 2), W_regularizer=l2(learning_rate))(input_model)
    x = ELU()(x)
    # x = Dropout(dropout)(x)
    x = Convolution2D(36, 5, 5, border_mode='valid', subsample=(2, 2), W_regularizer=l2(learning_rate))(x)
    x = ELU()(x)
    # x = Dropout(dropout)(x)
    x = Convolution2D(48, 5, 5, border_mode='valid', subsample=(2, 2), W_regularizer=l2(learning_rate))(x)
    x = ELU()(x)
    # x = Dropout(dropout)(x)
    x = Convolution2D(64, 3, 3, border_mode='valid', subsample=(1, 1), W_regularizer=l2(learning_rate))(x)
    x = ELU()(x)
    # x = Dropout(dropout)(x)
    x = Convolution2D(64, 3, 3, border_mode='valid', subsample=(1, 1), W_regularizer=l2(learning_rate))(x)
    x = ELU()(x)
    # x = Dropout(dropout)(x)
    x = Flatten()(x)
    x = Dense(100)(x)
    x = ELU()(x)
    x = Dropout(dropout)(x)
    x = Dense(50)(x)
    x = ELU()(x)
    x = Dropout(dropout)(x)
    x = Dense(10)(x)
    x = ELU()(x)
    predictions = Dense(1)(x)
    model = Model(input=input_model, output=predictions)
    model.compile(optimizer='adam', loss='mse')
    print(model.summary())
    return model

BATCH_SIZE = 128
EPOCHS = 30
FORMAT = '%(asctime)-15s %(clientip)s %(user)-8s %(message)s'
WIDTH = 66
HEIGHT = 200
DEPTH = 3
ALPHA = 0.01
DROPOUT = 0.5

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Steering angle model trainer')
    parser.add_argument('--batch', type=int, default=BATCH_SIZE, help='Batch size.')
    parser.add_argument('--epoch', type=int, default=EPOCHS, help='Number of epochs.')
    parser.add_argument('--alpha', type=float, default=ALPHA, help='Learning rate')
    parser.add_argument('--dropout', type=float, default=DROPOUT, help='Dropout rate')
    parser.add_argument('--loadWeights', type=bool, default=False, help='Load weights')
    args = parser.parse_args()

    print('-------------')
    print('BATCH: {}'.format(args.batch))
    print('EPOCH: {}'.format(args.epoch))
    print('ALPA: {}'.format(args.alpha))
    print('DROPOUT: {}'.format(args.dropout))
    print('Load Weights?: {}'.format(args.loadWeights))
    print('-------------')

    # split data into training and testing
    df_train, df_val = __train_test_split('data/driving_log.csv')

    print('TRAIN:', len(df_train))
    print('VALIDATION:', len(df_val))

    model = NvidiaModel(args.alpha, args.dropout)

    # Saves the model...
    with open('.hdf5_checkpoints/model.json', 'w') as f:
        f.write(model.to_json())

    try:
        if args.loadWeights:
            print('Loading weights from file ...')
            model.load_weights('model.h5')
    except IOError:
        print("No model found")

    checkpointer = ModelCheckpoint('.hdf5_checkpoints/weights.{epoch:02d}-{val_loss:.3f}.hdf5')
    early_stop = EarlyStopping(monitor='val_loss', patience=2, verbose=0, mode='auto')
    logger = CSVLogger(filename='.hdf5_checkpoints/history.csv')

    history = model.fit_generator(
        generate_batches(df_train, BATCH_SIZE),
        nb_epoch=args.epoch,
        samples_per_epoch=400*args.batch,
        validation_data=generate_batches(df_val, args.batch/4),
        nb_val_samples=100*BATCH_SIZE,
        callbacks=[checkpointer, early_stop, logger]
    )