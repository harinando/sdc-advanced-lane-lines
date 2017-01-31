import numpy as np
import pandas as pd
import cv2

# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from transformations import Preproc, RandomShift

def ReadImg(path):
    return np.array(cv2.cvtColor(cv2.imread(path.strip()), code=cv2.COLOR_BGR2RGB))


def generate_batches(df, batch_size):

    while True:

        batch_x = []
        batch_y = []

        for idx, row in df.iterrows():
            camera = np.random.choice(['left', 'center', 'right'])
            basename = 'data/{}'.format(row[camera].strip())
            img = ReadImg(basename)

            if camera == 'left':
                steering_angle = row['steering'] + .25
            elif camera == 'center':
                steering_angle = row['steering']
            elif camera == 'right':
                steering_angle = row['steering'] - .25

            img, steering_angle = RandomShift(img, steering_angle)
            img = Preproc(img)

            batch_x.append(np.reshape(img, (1, 66, 200, 3)))
            batch_y.append([steering_angle])

            if len(batch_x) == batch_size:
                batch_x, batch_y = shuffle(batch_x, batch_y)
                yield np.vstack(batch_x), np.vstack(batch_y)
                batch_x = []
                batch_y = []


def __train_test_split(csvpath):
    df = pd.read_csv(csvpath)
    df = shuffle(df)
    return train_test_split(df, test_size=0.2, random_state=42)

""" Private functions
"""


def __generate_arrays_from_dataframe(df, batch_size=32, generateData=False):
    for offset in range(0, len(df), batch_size):
        batchX = []
        batchY = []

        samples = df[offset:offset+batch_size]
        for idx, row in samples.iterrows():
            camera = np.random.choice(['left', 'center', 'right'])
            basename = 'data/{}'.format(row[camera].strip())
            img = ReadImg(basename)

            if camera == 'left':
                steering_angle = row['steering'] + .25
            elif camera == 'center':
                steering_angle = row['steering'] # + np.random.normal(loc=0, scale=0.2)
            elif camera == 'right':
                steering_angle = row['steering'] - .25

            img, steering_angle = RandomShift(img, steering_angle)

            img = Preproc(img)

            batchX.append(np.reshape(img, (1, 66, 200, 3)))
            batchY.append([steering_angle])

        return np.vstack(batchX), np.vstack(batchY)


def __generate_arrays_from_dataframe_old(df, batch_size=32, generateData=False):
    std = np.std(df['steering'])
    for offset in range(0, len(df), batch_size):
        batch = df[offset:offset+batch_size]
        if generateData:
            alpha = np.random.rand()
            if alpha < 0.25:
                batch = batch.append(
                    df[df['steering'] > std].sample(batch_size, replace=True))
            elif alpha < 0.5:
                batch = batch.append(df[df['steering'] < -std].sample(batch_size/2, replace=True))
        X = np.array([cv2.imread('data/{}'.format(basename)) for basename in batch['center'].values])
        y = batch['steering'].values
        return X, y