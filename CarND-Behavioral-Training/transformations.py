import cv2
import numpy as np


class Transform:

    def __init__(self):
        pass

    def apply(self):
        pass


class Grayscale(Transform):

    def apply(self, img):
        width, weight, depth = img.shape
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).reshape(width, weight, 1)

    def toString(self):
        return '{}'.format('Grayscale')


class Normalizer(Transform):

    def __init__(self, a=-0.5, b=0.5, min=0, max=255):
        self.a = a
        self.b = b
        self.min = min
        self.max = max

    def apply(self, img):
        return self.a + (((img-self.min)*(self.b-self.a))/(self.max-self.min))

    def toString(self):
        return '{}'.format('Normalize')


class Preprocess(Transform):

    def __init__(self, transforms):
        self.transforms = transforms

    def apply(self, img):
        for trans in self.transforms:
            img = trans.apply(img)
        return img

    def applies(self, X_train):
        return np.array([self.apply(x) for x in X_train])


class Rotate(Transform):
    def __init__(self, angle=None):
        self.angle = angle

    def apply(self, img):
        rows, cols, depth = img.shape
        M = cv2.getRotationMatrix2D((cols/2,rows/2), self.getAngle(), 1)
        return cv2.warpAffine(img, M, (cols, rows))

    def getAngle(self):
        if self.angle is None:
            return np.random.randint(-15, 15)
        return self.angle

    def toString(self):
        return '{} by {}'.format('Rotate', self.angle)


class Translate(Transform):
    def __init__(self, by_x=None, by_y=None):
        self.by_x = by_x
        self.by_y = by_y

    def apply(self, img):
        rows, cols, depth = img.shape
        M = np.float32([[1, 0, self.getX()], [0, 1, self.getY()]])
        return cv2.warpAffine(img, M, (cols, rows))

    def getX(self):
        if self.by_x is None:
            return np.random.randint(-2, 2)
        return self.by_x

    def getY(self):
        if self.by_y is None:
            return np.random.randint(-2, 2)
        return self.by_y

    def toString(self):
        return '{} by X {} by Y {}'.format('Translate', self.by_x, self.by_y)


class Identity(Transform):

    def apply(self, img):
        return img

    def toString(self):
        return '{}'.format('Identity')


class Skew(Transform):

    def __init__(self, pts1, pts2):
        self.pts1 = pts1
        self.pts2 = pts2

    def apply(self, img):
        M = cv2.getPerspectiveTransform(self.pts1, self.pts2)
        dst = cv2.warpPerspective(img,M,(32,32))
        return dst

    def toString(self):
        return '{}'.format('Skew')


class Resize(Transform):

    def __init__(self, width, height):
        self.width = width
        self.height = height

    def apply(self, img):
        return cv2.resize(img, dsize=(self.width, self.height))


class Crop(Transform):

    def __init__(self, x_min, x_max, y_min, y_max):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

    def apply(self, img):
        return img[self.y_min:self.y_max, self.x_min:self.x_max]


class GuassianBlur(Transform):

    def __init__(self, radius=None):
        self.radius = radius

    def apply(self, img):
        return cv2.GaussianBlur(img,(self.getRadius(), self.getRadius()),0)

    def getRadius(self):
        if self.radius is None:
            return np.random.choice([3, 5, 7])
        return self.radius

    def toString(self):
        return '{}'.format('Guassian Blur')


""" http://docs.opencv.org/3.2.0/d5/daf/tutorial_py_histogram_equalization.html
"""
class Equalizer(Transform):

    def __init__(self, clipLimit=2, tileSize=(8, 8)):
        self.trans = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileSize)

    def apply(self, img):
        return self.trans.apply(img)

def Preproc(img):
    if img.size == 0:
        return img

    preproc = Preprocess([
        # Grayscale(),
        # Equalizer(),
        Crop(50, 270, 20, 140),
        Resize(200, 66),
        Normalizer(a=-0.5, b=0.5)
    ])

    return preproc.apply(img)

def Shift(_img, by=10):

    height = _img.shape[0]
    width = _img.shape[1]
    img = Translate(by_x=by, by_y=0).apply(_img)
    if by > 0:
        img = Crop(by, width, 0, height).apply(img)
    else:
        img = Crop(0, width+by, 0, height).apply(img)
    img = Resize(width, height).apply(img)
    return img

def RandomShift(img, steering):
    tx = np.random.randint(-50, 50)
    steering += tx*0.0008
    return Shift(img, tx), steering
