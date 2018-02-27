# **Behavioral Cloning** 

## Writeup

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[center]: ./examples/center.jpg "Center"
[left]: ./examples/left.jpg "Left"
[right]: ./examples/right.jpg "Right"
[flipped]: ./examples/flipped.png "Right"
## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* [model.py](https://review.udacity.com/#!/rubrics/432/view) containing the script to create and train the model
* [drive.py](https://review.udacity.com/#!/rubrics/432/view) for driving the car in autonomous mode
* [model.h5](https://review.udacity.com/#!/rubrics/432/view) containing a trained convolution neural network 
* [Readme.md](https://review.udacity.com/#!/rubrics/432/view) summarizing the results
* [output_images.mp4](https://review.udacity.com/#!/rubrics/432/view) a video of the autonomous simulated driving

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```
Please note that I have run my simulator on a Pentium CPU with no graphics chip(Using Microsoft Display Adapter). Hence there is heavy lag (3 to 5 FPS on simulator) so I have reduced the throttle speed from 9 to 4.
#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5, 3x3 filter sizes and depths between 24 and 64 (model.py lines 18-24) Dense layers with 100, 50 and 10 with ELU activation to avoid dead RELU as well as dropouts between Dense layers.

The model includes RELU/ELU layers to introduce nonlinearity (code line 97), and the data is normalized in the model using a Keras lambda layer (code line 96). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 104 to 107). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 28, 86, 87 and 114). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 112).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road as well as the center lr-flipped image with measurements inverted.  


### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to try out various networks architectures and modify them and find out how well it performed on the track.

My first step was to use a  convolution neural network model similar to the NVIDIA architecture. I thought this model might be appropriate because it can pick out features such as lines and edges in its initial layers and bring out classification similar to road, dirt and railings and lane lines.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that it used Dropouts of 0.5 on first 2 Dense layers.
Then I changed the activation on the dense layer to elu to avoid any dead nodes.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I  adjusted the steering correction for the left and right images to about 0.2 as the lag in the machine meant that 0.2 was not responsive enough to keep the car on the road. The throttle speed also needed to be reduced to 4.


At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture ( model.py lines 93-110) consisted of a modified NVIDIA  convolution neural network architecture with the following layers and layer sizes:
The first 3 convolutional layers use a 5x5 kernel with stride of 2 and the last 2 use 3x3 kernel with stride 2x2 and 1x1 respectively. This is followd by flattening and 4 Fully Connected (Dense) layers with 0.5 Dropout after each of the first 2. These layers also use the ELU activation inplace of RELU to avoid Dead RELU's.
The final two layers are not activated.

|Layer (type)                   | Output Shape        | Param \#  |  Connected to |
|:------------------------------|:--------------------|:----------|:--------------|
|cropping2d_1 (Cropping2D)      | (None, 90, 320, 3)  | 0         | cropping2d_input_1[0][0] |
|lambda_1 (Lambda)              | (None, 90, 320, 3)  | 0         | cropping2d_1[0][0] |
|convolution2d_1 (Convolution2D)| (None, 43, 158, 24) | 1824      | lambda_1[0][0] |
|convolution2d_2 (Convolution2D)| (None, 20, 77, 36)  | 21636     | convolution2d_1[0][0] |
|convolution2d_3 (Convolution2D)| (None, 8, 37, 48)   | 43248     | convolution2d_2[0][0] |
|convolution2d_4 (Convolution2D)| (None, 6, 35, 64)   | 27712     | convolution2d_3[0][0] |
|convolution2d_5 (Convolution2D)| (None, 4, 33, 64)   | 36928     | convolution2d_4[0][0] |
|flatten_1 (Flatten)            | (None, 8448)        | 0         | convolution2d_5[0][0] |
|dense_1 (Dense)                | (None, 100)         | 844900    | flatten_1[0][0] |
|dropout_1 (Dropout)            | (None, 100)         | 0         | dense_1[0][0] |
|dense_2 (Dense)                | (None, 50)          | 5050      | dropout_1[0][0] |
|dropout_2 (Dropout)            | (None, 50)          | 0         | dense_2[0][0] |
|dense_3 (Dense)                | (None, 10)          | 510       | dropout_2[0][0] |
|dense_4 (Dense)                | (None, 1)           | 11        | dense_3[0][0] |
-------------------------------
|Total params: 981,819| Trainable params: 981,819 | Non-trainable params: 0 |
____________________________________________________________________________________________________

#### 3. Creation of the Training Set & Training Process

Due to limited processing power I was not able to capture any training images on my machine. Instead I used the sample training data that was available from the project for the training. Here is an example image of center, left and right lane driving:

![alt text][center]
![alt text][left]
![alt text][right]

To augment the data set, I also flipped images and angles thinking that this would reduce the bias on left turns causing it to turn left more aggressively than the right. For example, here is an image that has then been flipped:

![alt text][center]
![alt text][flipped]



After the collection process, I had 32144 number of data points. I then preprocessed this data by Cropping the image 50 pixels from the top and 20 pixels from the bottom. I also applied normalization to range between -0.5 tp 0.5

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 15 as evidenced by the stabilization in reduction of loss by around 12th to 13th epochs. I used an adam optimizer so that manually training the learning rate wasn't necessary.
