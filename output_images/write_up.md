
## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_not_car.png
[image2]: ./output_images/HOG_example.png
[image3]: ./output_images/sliding_windows.png
[image4]: ./output_images/sliding_window.png
[image5]: ./output_images/bboxes_and_heat.png
[image51]: ./output_images/bboxes_and_heat_filtered.png
[image6]: ./output_images/labels_map.png
[image7]: ./output_images/output_bboxes.png

[video1]: ./LaneDetection_video_output.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook (project.ipynb In[12]).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the RGB `B` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(8, 8)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.
Due to the time limit, I do not fully explore the parameters for HOG, and just try some cofigurations as bellow. Ultimatly, I find below settings not only has quick extrated time also makes SVM has a satisfied accuracy.
```python
colorspace = 'YUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 11
pix_per_cell = 16
cell_per_block = 2
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"

```
| Configuration Label | Colorspace | Orientations | Pixels Per Cell | Cells Per Block | HOG Channel | Extract Time |
| :-----------------: | :--------: | :----------: | :-------------: | :-------------: | :---------: | ------------:|
| 1                  | YUV        | 11           | 16              | 2               | 0           | 33.15        |
| 2                  | YUV        | 11           | 4               | 2               | 0           | 73.22        |
| 3                  | YUV        | 11           | 16              | 2               | ALL         | 59.58       |

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).
Codes are in (project.ipynb In[12-22])
I trained a linear SVM by steps:
1. Only extract Hog features of ALL color space for Car and No-Car images, respectively. Because I find Hog features is enough for SVM to get a high accuracy. So I do not use spatial color features and color histogram features to save computing time.
2. Define labels with '1' for car and '0' for no-car.
3. Shuffle data.
4. Split into train and test dataset.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?
Firstly, I define a single function find_cars() that can extract features using hog sub-sampling and make predictions
(project.ipynb In[25]) Then, becuase the size and position of cars in the image will be different depending on their distance from the camera, find_cars will have to be called a few times with different ystart, ystop, and scale values. These next few blocks of code are for determining the values for these parameters that work best. (project.ipynb In[39-42])

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on 8 different settings accroding to ROI (interested of region) and scale, using only YUV 3-channel HOG features as input feature vector, which provided a nice result.  (project.ipynb In[57])

Here are some example images:
![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./Lane_and_vehicle_detection_out.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.


I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here is a 1 frame and its corresponding heatmap:

![alt text][image5]

### Here is the heatmap filtered by threshold:

![alt text][image51]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the test frame:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

1. When I test my algorithm on video streams, I find it cannot **immediately** detect the new vehicle emerging from the rear of our vehicle. When the new vehicle fully (not a part of it) comes into our vision, my algorithm can work well.
2. Another promblem is that my detection window is slightly wobbly or unstable bounding boxes.

I think a more effective representation is great to solve these problems!


```python

```
