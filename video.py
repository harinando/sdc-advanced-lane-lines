from moviepy.editor import VideoFileClip
from IPython.display import HTML
import numpy as np
import cv2
import pickle
import glob
from tracker import Tracker
import argparse
from config import *

# Read in the saved img points
dist_pickle = pickle.load(open("{}/calibration_pickle.p".format(CAL_DIR), "rb"))
mtx = dist_pickle['mtx']
dist = dist_pickle['dist']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Steering angle model trainer')
    parser.add_argument(
        '--input',
        type=str,
        default='',
        help='Path to image folder. The video will be created from these images.'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='',
        help='Path to image folder. The video will be created from these images.'
    )

    args = parser.parse_args()

    clip1 = VideoFileClip(args.input)
    video_clip = clip1.fl_image(find_lanes)
    video_clip.write_videofile(args.output, audio=False)


