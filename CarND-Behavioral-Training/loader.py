import numpy as np
import pandas as pd
import cv2

# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from transformations import Preproc, RandomShift, RandomFlip, RandomBrightness


def ReadImg(path):
    return np.array(cv2.cvtColor(cv2.imread(path.strip()), code=cv2.COLOR_BGR2RGB))


def generate_validation(df):
    batch_x, batch_y = [], []
    for idx, row in df.iterrows():
        basename = 'data/{}'.format(row['center'].strip())
        steering_angle = row['steering']
        img = ReadImg(basename)
        img = Preproc(img)
        batch_x.append(np.reshape(img, (1, 66, 200, 3)))
        batch_y.append([steering_angle])
    return batch_x, batch_y


def generate_batches(df, batch_size):

    while True:

        batch_x = []
        batch_y = []

        for idx, row in df.iterrows():
            camera = np.random.choice(['left', 'center', 'right'])
            basename = 'data/{}'.format(row[camera].strip())
            img = ReadImg(basename)

            if camera == 'left':
                steering_angle = min(row['steering'] + .20, 1)
            elif camera == 'center':
                steering_angle = row['steering']
            elif camera == 'right':
                steering_angle = max(row['steering'] - .20, -1)

            img, steering_angle = RandomShift(img, steering_angle)
            img, steering_angle = RandomFlip(img, steering_angle)
            img, steering_angle = RandomBrightness(img, steering_angle)

            img = Preproc(img)
            batch_x.append(np.reshape(img, (1, 66, 200, 3)))
            batch_y.append([steering_angle])

            if len(batch_x) == batch_size:
                batch_x, batch_y = shuffle(batch_x, batch_y)
                yield np.vstack(batch_x), np.vstack(batch_y)
                batch_x = []
                batch_y = []


def __train_test_split(csvpath, balance=True):
    df = pd.read_csv(csvpath)
    if balance:
        zeros = df.where(df['steering'] == 0).dropna(axis=0).sample(1500)
        non_zeros = df.where(df['steering'] != 0).dropna(axis=0)
        df = zeros.append(non_zeros)
    df = shuffle(df)
    return train_test_split(df, test_size=0.2, random_state=42)