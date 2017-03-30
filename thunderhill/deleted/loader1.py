import csv

import numpy as np
import cv2


# Sklearn
from sklearn.utils import shuffle
from lstm.transformations import Preproc, RandomShift, RandomFlip, RandomBrightness, RandomBlur, RandomRotation
from nvidia3.config import *

show_images=True
show_images=False


def ReadImg(path):
    return np.array(cv2.cvtColor(cv2.imread(path.strip()), code=cv2.COLOR_BGR2RGB))


def generate_thunderhill_batches(gen, batch_size):
    batch_x = []
    batch_y = []
    while True:
        for img, steering_angle in gen:
            img, steering_angle = RandomShift(img, steering_angle)
            img, steering_angle = RandomFlip(img, steering_angle)
            img, steering_angle = RandomBrightness(img, steering_angle)
            img, steering_angle = RandomRotation(img, steering_angle)
            img, steering_angle = RandomBlur(img, steering_angle)

            steering_angle = np.min([1, steering_angle])
            steering_angle = np.max([-1, steering_angle])
            # Preproc is after ....
            img = Preproc(img)
            batch_x.append(np.reshape(img, (1, HEIGHT, WIDTH, DEPTH)))
            batch_y.append([steering_angle])

            if len(batch_x) == batch_size:
                batch_x, batch_y = shuffle(batch_x, batch_y)
                yield np.vstack(batch_x), np.vstack(batch_y)
                batch_x = []
                batch_y = []


def getSession5(repo):
    csvfname = os.path.join(repo, 'dataset_session_5/output.csv')
    data = None
    with open(csvfname, 'r') as csvfile:
        data = list(csv.reader(csvfile, delimiter=','))
        dd='/'.join(csvfname.split('/')[:-1])
        data = [(dd+'/'+x[0],float(x[3]),float(x[4]),float(x[5]),float(x[6])) for x in data]
    return data

sim320csvs = [
    'dataset_sim_000_km_few_laps/driving_log.csv',
    'dataset_sim_001_km_320x160/driving_log.csv',
    'dataset_sim_002_km_320x160_recovery/driving_log.csv',
    'dataset_sim_003_km_320x160/driving_log.csv'
]

polysynccsvs = ['dataset_polysync_1464466368552019/output.txt', '', 'dataset_polysync_1464552951979919/output.txt']

def getSim320(repo,rel_csv):
    csvfname = repo + rel_csv
    data=[]
    with open(csvfname, 'r') as csvfile:
        data_tmp = list(csv.reader(csvfile, delimiter=','))
        for row in data_tmp:
            x7 = [float(x) for x in row[7].split(':')]
            x8 = [float(x) for x in row[8].split(':')]
            dd='/'.join(csvfname.split('/')[:-1])
            data.append((dd+'/'+row[0],float(row[3]),float(row[4]),float(row[5]),float(row[6])))
    return data


def genSession5(folder):
    data = shuffle(getSession5(folder))
    while True:
        for row in data:
            img = ReadImg(row[0])[:200, 200:-200, :]
            steering_angle = row[1]
            yield img, steering_angle/2.0



def genSim001(folder):
    data = shuffle(getSim320(folder, sim320csvs[1]))
    while True:
        for row in data:
            img = ReadImg(row[0])[20:140, :, :]
            steering_angle = row[1]
            yield img, steering_angle



def genSim002(folder):
    data = shuffle(getSim320(folder, sim320csvs[2]))
    while True:
        for row in data:
            img = ReadImg(row[0])[20:140, :, :]
            steering_angle = row[1]
            yield img, steering_angle



def getPolysync(repo, rel_csv):
    csvfname = os.path.join(repo, rel_csv)
    data = []
    with open(csvfname, 'r') as csvfile:
        data_tmp = list(csv.reader(csvfile, delimiter=','))[1:]
        dd='/'.join(csvfname.split('/')[:-1])
        for row in data_tmp:
            data.append((dd+'/'+row[0], float(row[-3]), float(row[-2]), float(row[-1]), float(row[-6])))
    return data


def genPolysync0(folder):
    data = shuffle(getPolysync(folder, polysynccsvs[0])[500:-500])
    while True:
        for row in data:
            img = ReadImg(row[0])[-250:1:-1, 300:-300, :]
            steering_angle = row[1]
            yield img, steering_angle/5.0


def genPolysync2(folder):
    data=getPolysync(folder, polysynccsvs[2])
    data=shuffle(data[500:2000] + data[3100:-500])
    while True:
        for row in data:
            img = ReadImg(row[0])[-250:1:-1,300:-300,:]
            steering_angle = row[1]
            yield img,steering_angle/5.0

def genAll(folder):
    datasets = []
    datasets.append(shuffle(getPolysync(folder, polysynccsvs[0])[500:-500]))
    data = getPolysync(folder, polysynccsvs[2])
    data = shuffle(data[500:2000] + data[3100:-500])
    datasets.append(data)
    datasets.append(shuffle(getSim320(folder, sim320csvs[2])))
    datasets.append(shuffle(getSim320(folder, sim320csvs[3])))
    datasets.append(shuffle(getSession5(folder)))

    while True:
        nr =0
        ex=int(np.random.uniform() * len(datasets[nr]))
        row=datasets[nr][ex]
        img = ReadImg(row[0])[-250:1:-1,300:-300,:]
        steering_angle = row[1]
        yield img, steering_angle/5.0
        nr=nr+1
        ex=int(np.random.uniform() * len(datasets[nr]))
        row=datasets[nr][ex]
        img = ReadImg(row[0])[-250:1:-1,300:-300,:]
        steering_angle = row[1]
        yield img, steering_angle/5.0
        nr=nr+1
        ex=int(np.random.uniform() * len(datasets[nr]))
        row=datasets[nr][ex]
        img = ReadImg(row[0])[20:140,:,:]
        steering_angle = row[1]
        yield img,steering_angle
        nr=nr+1
        ex=int(np.random.uniform() * len(datasets[nr]))
        row=datasets[nr][ex]
        img = ReadImg(row[0])[20:140,:,:]
        steering_angle = row[1]
        yield img,steering_angle
        nr=nr+1
        ex=int(np.random.uniform() * len(datasets[nr]))
        row=datasets[nr][ex]
        img = ReadImg(row[0])[:200,200:-200,:]
        steering_angle = row[1]
        yield img,steering_angle/2.0