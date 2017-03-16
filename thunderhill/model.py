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
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.layers import LSTM
import argparse
import os
from loader import __train_test_split, generate_batches, generate_thunderhill_batches, getDataFromFolder
from config import *
from keras import backend as K


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
    input_model = Input(shape=(HEIGHT, WIDTH, DEPTH))
    x = Convolution2D(24, 5, 5, border_mode='valid', subsample=(2, 2), W_regularizer=l2(learning_rate))(input_model)
    x = ELU()(x)
    x = Convolution2D(36, 5, 5, border_mode='valid', subsample=(2, 2), W_regularizer=l2(learning_rate))(x)
    x = ELU()(x)
    x = Convolution2D(48, 5, 5, border_mode='valid', subsample=(2, 2), W_regularizer=l2(learning_rate))(x)
    x = ELU()(x)
    x = Convolution2D(64, 3, 3, border_mode='valid', subsample=(1, 1), W_regularizer=l2(learning_rate))(x)
    x = ELU()(x)
    x = Convolution2D(64, 3, 3, border_mode='valid', subsample=(1, 1), W_regularizer=l2(learning_rate))(x)
    x = ELU()(x)
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


def Nvidia(weights=None, include_top=True, dropout=0.5):
    input_model = Input(shape=(HEIGHT, WIDTH, DEPTH))
    # block #1
    x = Convolution2D(24, 5, 5, activation='elu', border_mode='same', subsample=(4, 4), init='he_normal', name='block1_conv1')(input_model)
    # block #2
    x = Convolution2D(36, 5, 5, activation='elu', border_mode='same', subsample=(2, 2), init='he_normal', name='block2_conv1')(x)
    # block #3
    x = Convolution2D(48, 5, 5, activation='elu', border_mode='same', subsample=(2, 2), init='he_normal', name='block3_conv1')(x)
    # block #4
    x = Convolution2D(64, 3, 3, activation='elu', border_mode='same', subsample=(1, 1), init='he_normal', name='block4_conv1')(x)
    x = Convolution2D(64, 3, 3, activation='elu', border_mode='same', subsample=(1, 1), init='he_normal', name='block4_conv2')(x)

    if include_top:
        x = Flatten()(x)
        x = Dense(100, activation='elu', init='he_normal', name='fc1')(x)
        x = Dense(50, activation='elu', init='he_normal', name='fc2')(x)
        x = Dense(10, activation='elu', init='he_normal', name='fc3')(x)
        x = Dense(1, name='prediction')(x)

    model = Model(input=input_model, output=x, name='nvidia')
    model.compile(optimizer=Adam(lr=1e-4), loss='mse')

    if weights:
        model.load_weights(weights)

    print(model.summary())
    return model

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Steering angle model trainer')
    parser.add_argument('--batch', type=int, default=BATCH_SIZE, help='Batch size.')
    parser.add_argument('--epoch', type=int, default=EPOCHS, help='Number of epochs.')
    parser.add_argument('--alpha', type=float, default=ALPHA, help='Learning rate')
    parser.add_argument('--dropout', type=float, default=DROPOUT, help='Dropout rate')
    parser.add_argument('--width', type=int, default=WIDTH, help='Save model here')
    parser.add_argument('--height', type=int, default=HEIGHT, help='Save model here')
    parser.add_argument('--depth', type=int, default=DEPTH, help='Save model here')
    parser.add_argument('--adjustement', type=float, default=ADJUSTMENT, help='x per pixel')
    parser.add_argument('--weights', type=str, help='Load weights')
    parser.add_argument('--model', type=int, default=0, help='Chose a model')
    parser.add_argument('--dataset', type=str, required=True, help='Get dataset here')
    parser.add_argument('--output', type=str, required=True, help='Save model here')
    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    print('-------------')
    print('BATCH        : {}'.format(args.batch))
    print('EPOCH        : {}'.format(args.epoch))
    print('ALPA         : {}'.format(args.alpha))
    print('DROPOUT      : {}'.format(args.dropout))
    print('Load Weights?: {}'.format(args.weights))
    print('Dataset      : {}'.format(args.dataset))
    print('OUTPUT       : {}'.format(args.output))
    print('MODEL        ; {}'.format(args.model))
    print('-------------')

    df_train, df_val = getDataFromFolder(args.dataset, args.output)
    print('TRAIN:', len(df_train))
    print('VALIDATION:', len(df_val))

    model = Nvidia()

    # Saves the model...
    with open(os.path.join(args.output, 'model.json'), 'w') as f:
        f.write(model.to_json())

    try:
        if args.weights:
            print('Loading weights from file ...')
            model.load_weights(args.weights)
    except IOError:
        print("No model found")

    checkpointer = ModelCheckpoint(os.path.join(args.output, 'weights.{epoch:02d}-{val_loss:.3f}.hdf5'))
    early_stop = EarlyStopping(monitor='val_loss', patience=2, verbose=0, mode='auto')
    logger = CSVLogger(filename=os.path.join(args.output, 'history.csv'))

    history = model.fit_generator(
        generate_thunderhill_batches(df_train, args),
        nb_epoch=args.epoch,
        samples_per_epoch=400*args.batch,
        validation_data=generate_thunderhill_batches(df_val, args),
        nb_val_samples=100*args.batch,
        callbacks=[checkpointer, early_stop, logger]
    )
