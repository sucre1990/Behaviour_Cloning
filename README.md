**Behavioral Cloning** 



---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./pictures_for_readme/center.jpg "center driving"
[image2]: ./pictures_for_readme/Recovering.jpg "recovering driving"



## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5 (I made change to the original code, added a preprocessing part to extract interest area and convert the RGB space to YUV space)
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The design of this driving behavioral cloning is based on [Nvidia's self-driving car model](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/). 

It consists of a convolution neural network with two different filters (5x5 and 3x3) and depths between 24 and 64.

The model includes RELU layers to introduce nonlinearity after each layer (you can find in code the model structure in line 119 of model.py)

Before being fed into the model, the data is first converted from BGR color space to YUV color space and then normalized in the model using a Keras lambda layer (code line 123 of model.py). 



#### 2. Attempts to reduce overfitting in the model

The model contains SpatialDropout2D layers (in convolutional layers) and l2 regularizers (fully connected layers) in order to reduce overfitting (model.py lines 136 and 145). During training, I tested different combinations of regularizer and dropput probability. It is really tricky to learn since it is hard to intuitively find a direction. (Could you provide me some guidances on this fine tuning?)

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 18). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used a sgd optimizer (for introducing momentum to avoid being trapped into local optimization point), and different learning rates for different stages of training. First I used 0.001 and 30 epochs, after I knew where the vehicle might fall off the track, I added more recovering training data in that area and then re-train the model based on previous weights, in the meantime I tuned down both learning rate (0.00001) and epoch value (3). Then I repeated this process until the vehicle could successfully run the full track.



#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving from both tracks, recovering from the left and right sides of the road. In the training process, I included both center images and both left and righ side images (after being turning angle compensated). In addition, the training data set also include horizontal flipped original images and it is for two reasons: 1. data augmentation, 2 balance the turning (since the circle is on one direction, which may make the model have bias. I also filtered a large part of 0 turning data).  

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to train drving behavioral cloning molde, so that it will predict turning angle based on simuator perceived image and make the driving simulator drive automatically.   

My first step was to use a convolution neural network model similar to [Nvidia's self-driving car model](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/). This model and its method has been well-documented, which helps me focus on tuning other parameters.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. At first, I train the model with learning rate of 0.001 and 30 epochs. I checked the result, found where the vehicle fell off the track and added more recovering data in that area. Then I re-train the model based on previous model's weights but tuned down learning rate and epoch value. I repeated the process until the vehicle could successfully run the full track. 


At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 117-163) consisted of a convolution neural network. 

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)


|Layer (type)                     |Output Shape          |Param #     |Connected to|                     
|---------------------------------|:--------------------:|:----------:|:----------:|
|lambda_1 (Lambda)                |(None, 66, 200, 3)    |0           |lambda_input_1[0][0]|             
|convolution2d_1 (Convolution2D)  |(None, 31, 98, 24)    |1824        |lambda_1[0][0]|                   
|activation_1 (Activation: Relu)        |(None, 31, 98, 24)    |0           |convolution2d_1[0][0] |           
|spatialdropout2d_1 (SpatialDropot) |(None, 31, 98, 24)    |0           |activation_1[0][0] |              
|convolution2d_2 (Convolution2D)  |(None, 14, 47, 36)    |21636       |spatialdropout2d_1[0][0]  |       
|activation_2 (Activation: Relu)        |(None, 14, 47, 36)    |0           |convolution2d_2[0][0]|            
|spatialdropout2d_2 (SpatialDropot) |(None, 14, 47, 36)    |0           |activation_2[0][0]|               
|convolution2d_3 (Convolution2D)  |(None, 5, 22, 48)     |43248       |spatialdropout2d_2[0][0]|         
|activation_3 (Activation: Relu)        |(None, 5, 22, 48)     |0           |convolution2d_3[0][0]|            
|spatialdropout2d_3 (SpatialDropot) |(None, 5, 22, 48)     |0           |activation_3[0][0] |              
|convolution2d_4 (Convolution2D)  |(None, 3, 20, 64)     |27712       |spatialdropout2d_3[0][0] |        
|activation_4 (Activation: Relu)        |(None, 3, 20, 64)     |0           |convolution2d_4[0][0] |           
|spatialdropout2d_4 (SpatialDropo |(None, 3, 20, 64)     |0           |activation_4[0][0]   |            
|convolution2d_5 (Convolution2D)  |(None, 1, 18, 64)     |36928       |spatialdropout2d_4[0][0] |        
|activation_5 (Activation: Relu)        |(None, 1, 18, 64)     |0           |convolution2d_5[0][0]  |          
|spatialdropout2d_5 (SpatialDropot) |(None, 1, 18, 64)     |0           |activation_5[0][0]|               
|flatten_1 (Flatten)              |(None, 1152)          |0           |spatialdropout2d_5[0][0]  |       
|dense_1 (Dense)                  |(None, 100)           |115300      |flatten_1[0][0]    |              
|activation_6 (Activation: Relu)        |(None, 100)           |0           |dense_1[0][0]      |              
|dense_2 (Dense)                  |(None, 50)            |5050        |activation_6[0][0]  |             
|activation_7 (Activation: Relu)        |(None, 50)            |0           |dense_2[0][0]       |             
|dense_3 (Dense)                  |(None, 10)            |510         |activation_7[0][0]  |             
|activation_8 (Activation: Relu)        |(None, 10)            |0           |dense_3[0][0]       |                        
|dense_4 (Dense)                  |(None, 1)             |11          |activation_8[0][0]     |             



#### 3. Creation of the Training Set & Training Process

At first, I used mouse as my input device to collect data and collected 31620 images. However, during training, I found out the quality of the data is not very good, since the input is not smooth enough. Then I decided to give up this whole data set and tried different equipments as my input device: Logitec's G27 wheel and ps3's joystick. Finally I chose joystick, as it produced the best result.

To capture good driving behavior, I first recorded multiple laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to truning back to center after it off the center line. These images show what a recovery looks like starting from side:

![alt text][image2]

Then I repeated this process on track two in order to get more data points.

For more data, I include left and right-side images and compensated turning angle for different perceiving position. I used 0.2 for the offset value. (left turning angle = turning angle + 0.2, righ turing angel = turning angle - 0.2) 

To augment the data set, I also flipped images and angles thinking that this would balance the the track left turining data. this process I included in the pre-preocessing part.

After the collection process, I had 61566 number of data points. I then preprocessed this data by corping, deleting top and bottom useless information, converting BGR color space to YUV color space and then resize to 66x200 image (as per Nvidia's model). 

Then I found out there were to many going straight samples, which might make the model not sensitive to the turn. I used a random select exclude 90% of those samples (line 18).

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. 

At first I used learning rate of 0.001 and epochs of 30. Then I re-trained the model after I added more recoving samples within areas that the car fell off the track with learning rate of 0.00001 and epochs of 3. I kept on doing this until car could successfully operate on the whole track.

Here is the link to the video [link to the video](./run1.mp4)