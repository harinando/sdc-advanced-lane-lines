import cv2
import numpy as np
import pickle
import os
from preprocessing import ReadImg

class Undistort():

    def __init__(self, corner_size=(9, 6)):
        self.corner_size = corner_size
        self.objpoints = []   # 3D point in the real word
        self.imgpoints = []   # 2D point in the image plane
        self.images = []

        self.detected_corners = {}

    def fit(self, images):
        objp = np.zeros((self.corner_size[0]*self.corner_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.corner_size[0], 0:self.corner_size[1]].T.reshape(-1, 2)

        for fname in images:
            img = ReadImg(fname)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, self.corner_size, None)
            if ret == True:
                img = cv2.drawChessboardCorners(img, self.corner_size, corners, ret)
                self.objpoints.append(objp)
                self.imgpoints.append(corners)
                self.detected_corners[fname] = img

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, gray.shape[::-1], None, None)
        self.mtx = mtx
        self.dist = dist

    def apply(self, img):
        dst = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
        return cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)

    def saveImages(self, output_dir):
        for fname, img in self.detected_corners.items():
            path = '{}/detected-corners-{}'.format(output_dir, os.path.basename(fname))
            cv2.imwrite(path, img)

    def saveModel(self, fname='./calibration_pickle.p'):
        dist_pickle = {}
        dist_pickle['mtx'] = self.mtx
        dist_pickle['dist'] = self.dist
        pickle.dump(dist_pickle, open(fname, "wb"))