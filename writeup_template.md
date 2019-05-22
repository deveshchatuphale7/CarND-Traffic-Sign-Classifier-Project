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

[lenet]: ./lenet.png "lenet"

[60_kmh]: ./test/60_kmh.jpg "60_kmh.jpg"
[left_turn]: ./test/left_turn.jpeg "left_turn.jpeg"
[road_work]: ./test/road_work.jpg "road_work"
[stop_sign]: ./test/stop_sign.jpg "stop_sign"
[yield_sign]: ./test/yield_sign.jpg "yield_sign"


Here is a link to my [project code](./Traffic_Sign_Classifier-Copy1.ipynb)

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

Here is an exploratory visualization of the data set. 

![alt text][image1]

The histogram below shows count of each label in the dataset. 

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
number of epochs 10.I also used ADAM stochiastic optimizer 'optimizer = tf.train.AdamOptimizer(learning_rate = rate)' with learning rate 0.005.
I set batch size to 150, I tested with batch size of 100 & 200 above, but I got the better accuracy at 150.
I kept number of epochs as 10 as I observed that validation accuracy wasn't increasing significantly above number of 10 epochs 

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

![alt text][lenet]


I also shuffled the data set in order to randomly distribute test images with different classes in train,test and validation data
for that I used 'shuffle()' from sklearn library

My final model results were:
* training set accuracy of 98.3 %
* test set accuracy of 92.57 %


As the CNN with 2 convolutional layers, 3 fully connected, a learning rate of 0.005 and a batch size 150 appears to result in the optimal CNN,
 I utilized this for the final model as I found this existing LeNet architecture was able to predict digits with great accuracy in the LeNet labs. after testing and playing around with this parameters.
 I also used ADAM stochiastic optimizer 'optimizer = tf.train.AdamOptimizer(learning_rate = rate)' with learning rate 0.005, for this I referenced LeNet lecture notes & lab code. Along with that I also added dropout layer not to overfit the data.
 I initialy kept learning rate around 0.009 & as a result the accuracy of model on training set was around 85 %. So in order to keep model stable & increase training set accuracy I kept on reducing the learning rate and finally set to 0.005. 


#### 5. Probabilities  of images and classes
 


![alt text][stop_sign]

Image 0 probabilities: [10.616723    3.8455713   2.669806    2.2186196   0.87587404] 
 and predicted classes: [14  0 34 38 15]

![alt text][left_turn]

Image 1 probabilities: [16.987305    8.380401    4.820599    2.2235348   0.91358745] 
 and predicted classes: [38 34 13 14 15]


![alt text][60_kmh]

Image 2 probabilities: [ 5.0054965   4.6089253   1.2927918  -0.76526076 -1.203482  ] 
 and predicted classes: [10  9  3  5 35]


![alt text][yield_sign]

Image 3 probabilities: [27.546165  15.932422   2.9358974  2.172908  -0.5163945] 
 and predicted classes: [13 38 15  2 35]


![alt text][road_work]

Image 4 probabilities: [8.483552  8.009216  2.5893536 2.302762  2.0210102] 
 and predicted classes: [29 18 40 37 12]




As from the above predicted classes I observed that model was not able to clearly distinguish between speed signs
The model was able to detect that it is a speed sign but the value predicted was wrong.

For second image which is keep left sign image the model predicted it as keep right sign turn.
Here model failed to predict exact sign however the keep right sign is second in the predicted classes list.

For the stop sign image & yield sign image, the model was able to successfully predict both of the signs. 

For the road work sign image the model predicted it as Bicycles crossing here, what I observed is model detected the shape of the sign image correctly but failed to predict it based on the actual content in the sign maybe because of complex nature of picture/action in the sign.

So overall, model predicted 2 out of 5 images successfully, for rest of the images model was closer to identify the sign & predicted classes were
closer to actual sign image classes, model worked correctly on 40% of additional pictures as compared to 93% accuracy on test data, model failed to recognize the detailed content in the sign which in my opinion can be improved by increasing number of layers in CNN. or by increasing the dataset for some complex signs.Also I feel there is a need to consider other characteristics like image of sign in different seasons, during different daytimes, image of sign with different angles & dimensions,size, background of the sign, position of sign in the image etc.  
