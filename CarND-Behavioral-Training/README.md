#**Behavioral Cloning** 

**Behavrioal Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one and two without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./output/model.png "Model Visualization"
[image2]: ./output/brightness.png "Random Brightness"
[image3]: ./output/cropped.png "Cropped Image"
[image4]: ./output/flip.png "Flipped Image"
[image5]: ./output/rotated.png "Rotated Image"
[image6]: ./output/flip+30.png "Translated & Cropped Image"
[image7]: ./output/left-center-right.png "Left Center Right Image"
[image8]: ./output/hist.png "Distribution of steering angle"
[image9]: ./output/adjusted.png "Adjusted Distribution of steering angle"
[image10]: ./output/loss.png "Learning curve"


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* loader.py containing the script to generate data and select data
* transformation.py are helpers functions to manipulate the images such as rotation, translation ....
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

####2. Submssion includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.json
```

####3. Submssion code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model arcthiecture has been employed

My model consists of a convolution neural network based on the NVIDIA network:
* 5x5 filter sizes and depths between 32 and 64 for the first 3 layers 
* 3x3 filter sizes and depths 64 for the next 2 layers (model.py lines 30-55) 
* 3 denses layers
* 1 prediction layer


The model include ELU layers to introduce nonlinearity, a dropout after each dense layer, and the data is normalized around 0 and standard deviation 0.5 (transfomation.py lines 29-41)

####2. Attempts to reduce overfitting in the model

In order to reduce overfitting the model uses:
* dropout layer after each dense layer
* l2 regularization after each convolution layer with learning rate 0.001.

The model was trained and validated on different data sets to ensure that the model was not overfitting (model.py line 85). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 53).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road to address the unbalanceness of the data. I did not drive in the simulator at all because it was very difficult to control. Instead, I augmented the data using python generator; I used various image processing transformation to simulate various road and weather conditions and increases the dataset from 6000 to 50,000.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was mainly trial and error while fine tuning the model's parameters.

My first step was to use a convolution neural network model similar to the NVIDIA model. I thought this model might be appropriate because the NVIDIA team used the same model to train a self-driving car.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I added dropout layer after each dense layer which worked also pretty well for the "traffic sign recognition" project and l2 regularization after each convolution filters. To optimize the parameters of my regularizer, I performed a grid search with dropout values of {0.2, 0.3, 0.4, 0.5, 0.6} and learning rate of {0.1, 0.01, 0.001} and picked the model with the best mean square error on the validation set.

Fine tuning the steering correction for the left and right images was laso challenging because high correction makes the car zig-zag on track 1. I tried different value of steering correction and added more training set and was able to successfully smoothing out the driving on track 1 at full throttle. Similar approach was used on track 2 with the same results.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track:
* The car could not enter the bridge because of the constrast in the images. To address the issue, I randomly dim/brighten the images while training. This will also help when there are shadow on the road or under different lighting conditions.
* The car could not make turn that does not have explicit red marking or fence. This issue was addressed by using the left/right images as well as data augmentation. We performed random horizontal translation between -30 and 30 degrees of the center image and correct the steering angle by 0.05 degree per pixels.
* To increase randomness we randomly flip the images and negate the steering angle.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.  

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...
* 5x5 filter sizes and depths between 24 
* 5x5 filter sizes and depths between 36 
* 5x5 filter sizes and depths between 48
* 3x3 filter sizes and depths between 64
* Dense layer of size 100
* Dense layer of size 50
* Dense layer of size 10
* Prediction layer

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I augmented the datasets with appropriate transformation thinking about the different road condition referred in 4.1.

![alt text][image7]

I did not drive in track 2 at all. Instead, I added vertical shift to simulate slopes and randomly dimmed the image to address the shadow in the other track. Suprinsingly, it worked prety well since I was able to drive on track 2 without difficulties.

To augment the dataset, I also flipped images and rotated the images thinking that this would give unique data points for my training. For example, here is an image that has then been flipped, rotated, and translated:

![alt text][image4]
![alt text][image5]
![alt text][image6]


After the collection process, I had 50,000 number of data points. I then preprocessed this data by:
* Applying an RBG to HSV transformation.
* Cropped the image to focus on the road.
* Resized the input to 200 by 66 to match the NVIDIA input.
* Normalized the image between -0.5 and 0.5.
![alt text][image3]


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 7 as evidenced by the plot below.
![alt text][image10]

 I used an adam optimizer so that manually training the learning rate wasn't necessary.

 Here are the you tube link of the two run, I recorded it with a camera because using the simulator and a video recording app on my laptop was very slow and made the car zig-zag all over the place.  
 https://youtu.be/eAQ1PHDmhsQ 
 https://youtu.be/0lXM-RIPIGU