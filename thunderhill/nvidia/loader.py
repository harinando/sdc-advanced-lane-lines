import numpy as np
import pandas as pd
import cv2
import glob
import os
import pickle

# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from transformations import Preproc, RandomShift, RandomFlip, RandomBrightness, RandomRotation, RandomBlur, Resize
from config import *

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


def ReadImg(path):
    return np.array(cv2.cvtColor(cv2.imread(path.strip()), code=cv2.COLOR_BGR2RGB))


def generate_thunderhill_batches(df, args):

    while True:
        batch_x = []
        batch_y = []

        for idx, row in df.iterrows():
            steering_angle = row['steering']
            img = ReadImg(row['center'])

            if '320x160' in row['center']:
                img = img[20:140, :, :]

            img, steering_angle = RandomShift(img, steering_angle, args.adjustement)
            img, steering_angle = RandomFlip(img, steering_angle)
            img, steering_angle = RandomBrightness(img, steering_angle)
            img, steering_angle = RandomRotation(img, steering_angle)
            img, steering_angle = RandomBlur(img, steering_angle)

            # Preproc is after ....
            img = Preproc(img)

            batch_x.append(np.reshape(img, (1, HEIGHT, WIDTH, DEPTH)))
            batch_y.append([steering_angle])

            if len(batch_x) == args.batch:
                batch_x, batch_y = shuffle(batch_x, batch_y)
                yield np.vstack(batch_x), np.vstack(batch_y)
                batch_x = []
                batch_y = []


def generate_lstm_batches(df, features_extractor, seq_length, batch_size):
    CNN_INPUT_SIZE = features_extractor.layers[11].output_shape[1]

    batch_x = []
    batch_y = []
    while True:
        X = []
        i = np.random.randint(0, df.shape[0] - seq_length)
        y = df.iloc[i+seq_length]['steering']

        for idx, row in df.iloc[i: i + seq_length].iterrows():
            img = ReadImg(row['center'])
            if '320x160' in row['center']:
                img = img[20:140, :, :]
            img = Preproc(img)
            cnn_features = features_extractor.predict(np.reshape(img, (1, HEIGHT, WIDTH, DEPTH)))[0]
            X.append(cnn_features)
        # batch_x.append(np.random.rand(1, seq_length, CNN_INPUT_SIZE))
        batch_x.append(np.reshape(X, (1, seq_length, CNN_INPUT_SIZE)))
        batch_y.append(y)

        if len(batch_x) == batch_size:
            yield np.vstack(batch_x), np.vstack(batch_y)
            batch_x = []
            batch_y = []


# def extractFeatures(df)

"""
Randomly split the dataset
"""
def __train_test_split(csvpath, balance=True):
    df = pd.read_csv(csvpath)
    if balance:
        zeros = df.where(df['steering'] == 0).dropna(axis=0).sample(1500)
        non_zeros = df.where(df['steering'] != 0).dropna(axis=0)
        df = zeros.append(non_zeros)
    df = shuffle(df)
    return train_test_split(df, test_size=0.2, random_state=42)


def getDataFromFolder(folder, output, normalize=False, randomize=True, balance=True, split=True):
    data = pd.DataFrame(columns=COLUMNS)
    for csvpath in glob.glob('{}/**/driving_log.csv'.format(folder)):
        df = pd.read_csv(csvpath)
        df.columns = COLUMNS

        skip = False
        for toSkip in SKIP:
            if toSkip in csvpath:
                skip = True
        if skip:
            continue
        basename = os.path.dirname(csvpath)
        df['center'] = basename + '/' + df['center']
        df['positionX'], df['positionY'], df['positionZ'] = df['position'].str.split(':', 2).str
        df['rotationX'], df['rotationY'], df['rotationZ'] = df['rotation'].str.split(':', 2).str
        df[COLUMNS_TO_NORMALIZE] = df[COLUMNS_TO_NORMALIZE].astype(float)
        data = data.append(df)

    data = data.drop(['right', 'left'], 1)

    if balance:
        # data = data.where(data['steering'] != 0).dropna(axis=0)
        zeros = data.where(data['steering'] == 0).dropna(axis=0).sample(1000)
        non_zeros = data.where(data['steering'] != 0).dropna(axis=0)
        data = zeros.append(non_zeros)

    if randomize:
        data = shuffle(data)

    if normalize:
        scaler = StandardScaler()
        data[COLUMNS_TO_NORMALIZE] = scaler.fit_transform(data[COLUMNS_TO_NORMALIZE])
        pickle.dump(scaler, open(os.path.join(output, SCALER), 'wb'))

    if split:
        return train_test_split(data, test_size=0.2, random_state=42)
    return data