# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)


* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./1.png "Visualization"
[image2]: ./2.png "Histogram"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"


Here is a link to my [project code](./Traffic_Sign_Classifier-Copy1.ipynb.ipynb)

### Data Set Summary & Exploration

Data set consists of 3 pickle files with trainnig,testing & validation data. 
I used the pandas library to import files as a dataframe & calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (34799, 32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

![alt text][image2]

### Design and Test a Model Architecture

As a first step, I decided to convert the images to grayscale because color of sign is not among the 
features for classifying the sign & as model will be taking input as pixel values, the grayscale normalized 
image pixel values will help in providing better results.

Also normalizing image helped in maintaining numerical stabilty in the pixel values,
aiding in avoiding numerical/calculation errors while programming 

I decided to generate additional data because ...
I experianced that accuracy was between 40% to 60% without additional data set and 
I tried tuning learning rate, batch size & epoch size in order to improve the accuracy
Hence I thought of incresing number of training and validation images.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3	    |   Output = 10x10x16                           |
| RELU					|												|
| DropOut				|												|
| Fully Connected		| Input = 120. Output = 84						|
| RELU					|												|
| DropOut				|												|
| Fully Connected   	| Input = 84. Output = 43                       |
| Logits				|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an 'tf.train.AdamOptimizer()' as I found it useful in LeNet 5 assignment as well.I set batch size of 150 with 
number of epochs 10

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

I also shuffled the data set in order to randomly distribute test images with different classes in train,test and validation data
for that I used 'shuffle()' from sklearn library



My final model results were:
* training set accuracy of 98.3 %
* test set accuracy of 92.57 %


 
