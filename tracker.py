import numpy as np
import matplotlib.pyplot as plt


class Tracker:

    def __init__(self, window_width, window_height, margin, ym=1, xm=1, smooth_factor=15):
        # list that store all the past (left, right) center set values used for smoothing the output
        self.recent_centers = []

        # the window pixel width of the center values, used to count the pixels inside center windows to determine
        # curve values
        self.window_width = window_width

        # the window pixel height of the center values, used to count the pixels inside center windows to determine
        # curve values breaks the image into vertical level
        self.window_height = window_height

        # the pixel distance in both directions to slide (left_window + right_window) template for searching
        self.margin = margin

        self.ym_per_pix = ym  # meter per pixel in vertical axis

        self.xm_per_pix = xm  # meter per pixel in horizontal axis

        self.smooth_factor = smooth_factor

    # store line segment positions
    def find_window_centroid(self, warped):

        self.out_img = np.dstack((warped, warped, warped))*255

        window_width = self.window_width
        window_height = self.window_height
        margin = self.margin

        img_height = warped.shape[0]
        img_width = warped.shape[1]

        window_centroids = []   # store the (left, right) centroid per level
        window = np.ones(window_width) # create window template that we will use in our convolution

        midpoint = int(warped.shape[1]/2)
        midheight = int(3*warped.shape[0]/4)
        nwindows = int(warped.shape[0]/window_height)
        offset = window_width/2

        # sum the quarter buttom of the image to get slice
        l_sum = np.sum(warped[midheight:, :midpoint], axis=0)
        l_center = np.argmax(np.convolve(window, l_sum))-offset
        r_sum = np.sum(warped[midheight:, midpoint:], axis=0)
        r_center = np.argmax(np.convolve(window, r_sum))-offset+midpoint

        # plt.plot(np.sum(warped[midheight:, :], axis=0))

        window_centroids.append((l_center, r_center))

        for level in range(1, nwindows):
            # convolve the window into the vertical slice of the image
            image_layer = np.sum(warped[int(img_height-(level+1)*window_height):int(img_height-(level*window_height)), :], axis=0)
            conv_signal = np.convolve(window, image_layer)
            # Find the best left centroid by using the past left center as reference
            # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of the window
            l_min_index = int(max(l_center+offset-margin, 0))
            l_max_index = int(min(l_center+offset+margin, img_width))
            l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
            # Find the best right centroid by using hte past right center as reference
            r_min_index = int(max(r_center+offset-margin, 0))
            r_max_index = int(min(r_center+offset+margin, img_width))
            r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
            window_centroids.append((l_center, r_center))
        self.recent_centers.append(window_centroids)

        return np.average(self.recent_centers[-self.smooth_factor:], axis=0)

    def curverad(self, ys, xs):
        pass