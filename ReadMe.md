**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_image.png
[image2]: ./output_images/notcar_image.png
[image3]: ./output_images/car_hog_image.png
[image4]: ./output_images/notcar_hog_image.png
[image5]: ./output_images/grid_search_params.png
[image6]: ./output_images/grid_search_result.png
[image7]: ./output_images/scale_1.00_x_4.png
[image8]: ./output_images/scale_1.25_x_4.png
[image9]: ./output_images/scale_1.50_x_4.png
[image10]: ./output_images/scale_1.75_x_4.png
[image11]: ./output_images/scale_2.00_x_4.png
[image12]: ./output_images/original1.jpg
[image13]: ./output_images/detection1.jpg
[image14]: ./output_images/detection_with_heatmap1.jpg
[image15]: ./output_images/original2.jpg
[image16]: ./output_images/detection2.jpg
[image17]: ./output_images/detection_with_heatmap2.jpg
## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in function `get_hog_features` of the file `lesson_functions.py`.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1] ![alt text][image2]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image3]
![alt text][image4]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters. Initially i uses the YUV clor space but it turns out there is some problem that i can not apply it to large dataset. So i change it back to YCrcb space. I just take the value for `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`, 
it turns out it works well so i just use those values. 

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

Firstly a linear SVM is used to the classification. For the training result with sample size around 900 for both car/non-car images. I am able to get around 96.5% accuracy. Then i directly applies it the bit dataset. And i got the training accuracy result around 98.27%. Then i use it with the test
video, it looks ok. But when i applies it to the project video, i got a lot of false positive detection.

Then i tried the non linear SVM classifier. In order to get the bast parameter match, i use the GridSearchCV with the following parameter.

![alt text][image5]

With the setting, i got an optimized parameter set by use the small test set as shown below:

![alt text][image6]

The accuracy is better than the linear SVM classifier.

Then i took the parameter set and apply it to the large data set, i am ablt to get upto 99.1% testing accuracy. After this i store the trained model in a local file named `svcClassifier_full.p`, the related code is inside `search_classify.py`. For future detection, the model is directly loaded from the file. The loading code is in `load_model` function of `lesson_functions` file. 

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

By examing the video frame image. I decide only exame to lower half to reduce the false positive and improve efficiency. From the camera perspective transformation, we know that the same will appear small as it is furthur away from the camera position.
Base on that i decided to use different size of sliding window starts from different Y position. Related codes are in `find_cars_subsamples` function in the `lesson_functions.py`. 
    
    `scales  = [1.0, 1.25, 1.5, 1.75, 2.0]`
    `ystops   = [528, 560, 592, 624, 656]`
    `ystarts = [400, 416, 448, 480, 528]`
The corresponding images for diffent sliding window size are shown below:

![alt text][image7]![alt text][image8]
![alt text][image9]![alt text][image10]
![alt text][image11]

Howeve, after apply it to the project video, i found there are too many false positives from the left side which we know that it's actually not need for our project.
so i add another constraints to romove search features from the left half as shown below:

    `xstops = [1280, 1280, 1280, 1280, 1280]`
    `xstarts = [720, 720, 720, 720, 720]`

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my test video result](./test_video_result.mp4)

Here's a [link to my project video result](./project_video_result.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.
Below are two original test image:

![alt text][image12]![alt text][image13]
![alt text][image15]![alt text][image16]

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

To improve the correctness of the detection, i used a queue fo store the last 16 frames of detection, and sum up all the detected boxes for each frame and filter out 
the regions that are not shown up in half of the tracked frames which is 8. The related code is in `process_image` function of `lesson_functions.py` file. After applying this average technique, i got below output from the above
test images with corresponding heatmaps.

![alt text][image13]
![alt text][image15]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

My approach has two major steps.

Fistly is to find a good classifier. I tried both linear and non-linear svm classifier as well as augment the test image by flip them. It turns out a non linear svc with correct tuned parameter
is crucial for the project. Alought the accuracy only increases by 0.5%, but it dramastically reduces the number of false positive detection.

Secondly how to detect the cars. I choose selected region(lower, right part) for detection to reduce the false positive as well as improve the performance. I use a sliding window technique to find whether a sub-region
is a car or not. And record all the detected region and sum them up as a heat map. In order to reduce the number of false positve detection, a number of historical frames are used so that only the pixels constanly 
detected at least half of the frames will be marked as positive detection. Then i use the `scipy.ndimage.measurements.label()` to find the bounding box for each isolated detection as assume
 that it is right detection and display it.

 There are some potential problems for the approach.
 
 1. The manually selected region of interested. The region is selected due to two reasons:
    1.1  One is to reduce false positive, we can use more training data to improve the accuracy to duce false positive detection
    2.1 Another reason is to improve the detection speed. This can be improved by find a way to extract the feature for subsampled window. Currently each slidnig window we will tract its features. Theoretically that we should have a way that get feature for the whole image and then extract all the subsample window features using some mapping method. By doing this, it should imporve the rendering speed a lot 
 
 2. Another potential problem is that the way it processes the detected car bounding box is too simple. We can have better ways to process the detections liking tracking the detected car center. For any new detected cars, we know that it can not just shown up in the middle of the frame. It must come from either the top or from the bottom first. By using this information, we can identify the cars more accurately.



