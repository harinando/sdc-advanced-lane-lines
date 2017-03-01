##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./camera_cal/calibration1.jpg "Undistorted"
[image2]: ./output_images/undistorted-calibration1.jpg "Road Transformed"
[image3]: ./output_images/tracked-test1.jpg "Binary Example"
[image4]: ./output_images/warped-test1.jpg "Warp Example"
[image5]: ./output_images/lanes-test1.jpg "Fit Visual"
[image6]: ./output_images/output-test1.jpg "Output"
[video1]: ./output_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  


###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in calibration.py Undistort.fit method.  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]
####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of color and gradient thresholds to generate a binary image (thresholding steps are in the 7th cell of the ipython notebook data_exploration.pynb`). 
More specifically, 
 - I compute the gradient with respect to y and x and select the pixels that were both detected in both filter.
 - I also select pixels between a specific threshold value HSV and HSL color space. 
 - Finally, I mask the region that are not relevant to the lane finding search.

Here's an example of my output for this step.

![alt text][image3]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `Warp()`, which appears in lines 124 through 133 in the file `preprocessing.py` 
(output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook data_exploration.pynb).  
The `Warp()` function takes as inputs an image (`img`).  I chose the hardcode the source and destination points in the following manner:

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 590, 470      | 260,   0      | 
| 750, 470      | 460,   0      |
| 1190, 700     | 460, 720      |
| 260, 700      | 269, 730      |


The optimal value of the source and destination was computed from a slider widget in the ipython notebook in the ** Compute Perspective transform ** section. 

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial? 

The code for my lane detection is located in the python notebook in the ** Perspective transformation & Lane detection ** section. (findLanes function)
To detect the lanes, I estimated the position of the left and right lanes respectively by computing the histogram of white pixels over the x axis and getting the maximum position with the highest value.
Then, I divide my images by 9 vertically and convolve to detect the lines. I fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in the function `findLanes()` in my ipython notebook.  Here is an example of my result on a test image:

![alt text][image6]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_video.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

In overall, the color thresholding and gradient thresholding combined with perspective transform was a great strategy to detect the lanes. The main challenge is to tune the different parameters of the model manually.

- The perspective transformation is very fragile and might fail for different situation such as hilly road, sharp turns, car driving next on highway because the the source and the destination has been tuned manually using jupyter notebook widget. Instead, I should fit the source and destination of the perspective given the steering angle using neural network for instance.
- Finding the lanes not optional and time consuming because I am always doing a full search, rather I should estimate the lanes in the next frame once I get the confidence that I have correctly detected the lanes by fitting a polynomial to it.


Smoothing the fitted lines was very important while streaming the videos. I decided to keep track of the lanes of the last 15 frames, and averaged the value of the filters. Without smoothing, the lane was very wobbly especially when cars were passing by.


