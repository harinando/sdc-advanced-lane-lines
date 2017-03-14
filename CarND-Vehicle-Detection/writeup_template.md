**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
* Run your pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_not_car.png
[image2]: ./output_images/hog_features_channel.png
[image3]: ./output_images/hog_features_color_space.png
[image4]: ./output_images/sliding_windows.png
[image5]: ./output_images/detected.png
[image6]: ./output_images/label_map_and_detected_car.png
[video1]: ./project_video.out.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  
---
##Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in lines #17 through #33 of the file called `feature_extractor.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here are some visualization of the hog features extracted for different chanel:


![alt text][image2]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and visualized the extracted features in  Jupyter notebook to get a feel on how each parameters is affecting the performance of our model.
Here are some visualization of the hog features extracted for different color space:


![alt text][image3]
 
Then, I fine tuned those parameters by performing a mini grid search using an SVM as classifier. I picked the one with the highest accuracy which you can refer 
to in the file `config.py`.
For a more robust detection, automating this process would be ideal, but for the purpose of this assignment I just used trial and error.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using hog features, histogram color, and a spacial features (the flatten image) as features. I performed a forward features selection where I started to add each feature individually 
and observed its effect of the accuracy of the classifier. The parameter of the hog features has already been selected above. I noticed that adding histogram color 
and a spacial features increases the accuracy of the classifier by 2% from 97% to 99.21%. The code for this step is container in cell #14 training an SCV section of the 
  ipython notebook `data_exploration.ipynb`. As alway, I divided the dataset into training and testing set with random shuffling to ovoid overfitting.
 

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search window positions at scale 1 and 1.5 from 400 to 656px of the image. I tried different value of scale and overlap while plotting the detected
bounding boxes. Since the classifier accuracy was over 99%, I had high confidence that the detection is good. Every parameters was selected manually but the same grid search approach could be 
performed to fine tuned the parameters.

The code for this step is in the jupyter notebook `data_exploration.ipynb` cell #34 in the "Sliding window" section.

![alt text][image4]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  
Here are some example images.

![alt text][image5]
------------------

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_vide.out.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here the resulting bounding boxes are drawn onto the last frame in the series and the corresponding heat map representation:
![alt text][image6]

The code for this pat is found in the jupyter notebook `data_exploration.ipynb` cell #39 in the section detecting false positive part.

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The main challenges during implementation was fine tuning the paramters of the different features I was using in my svm. 

There are few areas of failure in the pipelines:
 - Performance: since I am performing a sliding window, I am taking some performance hit on my localisation. Ideally, I should perform be able to read the image once (using depth first search maybe), 
  detect blob of color of certain size, and then classify the blob as car or not. I am not too sure about the accuracy of this approach but I think it's worth exploring. If I were to detect the car in as stream of video live, 
   it might lag and therefore fail.
 - Improving the accuracy of the classifier: I think that I could improve the performance of the classifier with further parameters tuning. As discussed earlier, The features extraction  
 should be automated so that it can generalises more. I am also interested in exploring other classifiers such as CNN. diversifying the dataset is avery good idea by adding more training set.

