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

[image1]: ./writeup_image/calib_image.png "calib_image"
[image2]: ./writeup_image/undistort_testimage.png "undistort_testimage"
<!-- [image2]: ./test_images/test1.jpg "Road Transformed" -->
[image3_1]: ./writeup_image/image_stack.png "image_stack"
[image3_2]: ./writeup_image/combine_threshold.png "combine_threshold"
<!-- [image3]: ./examples/binary_combo_example.jpg "Binary Example" -->
[image4]: ./writeup_image/warpimage.png "Warp Example"
[image4_2]: ./writeup_image/warped2.png "Warp Example"
<!-- [image5]: ./examples/color_fit_lines.jpg "Fit Visual" -->
[image6]: ./writeup_image/finaloutput.png "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the second code cell of the IPython notebook located in "./find_lanes.ipynb".  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

I used cv2.calibrateCamera parameter obtained by chessboard board then applied it to the test image.  

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at 6th code cell ("color and gradient stack" section) in ./find_lanes.ipynb).  Here's an example of my output for this step. Green is a gradient threshold and blue is a color threshold.  
I decided parameter by try and error using ipywidgets.  

![alt text][image3_1]

Here's combined image of color and gradient thresholds.  
![alt text][image3_2]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform appears in 7th code cell ("perspective transform and provide an example of a transformed image" section) in ./find_lanes.ipynb.  
I chose the hardcode the source and destination points by checking test_images/straight_lines1.jpg so that 4 src points were on the line.  


This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 198, 720      | 350, 720      |
| 1110, 720     | 930, 720      |
| 707, 464      | 930, 0        |
| 573, 464      | 350, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

I also checked that curved lines were curved in the warped image.  
![alt text][image4_2]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I did this in lines 261 through 285 in my code in `find_lanes.py`

1. create histogram
2. find two peaks
3. using window sliding around each peak
4. when a fitted line of the previous frame existed, just searched around this lines.
5. collecting line candidate positions through #3 or #4, fitted 2nd order polynomial using `numpy.polyfit(ys, xs, 2)`.


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to the center.

I did this in lines 315 through 324 in my code in `find_lanes.py`
###### curvature
- average left and right curvature
- equation
    - `curvature = ((1 + (2 * a * y + b)^ 2)^ 1.5) / fabs(2 * a)`
    - `f = ay^2 + by + c`

###### position of the vehicle
- I used x positions at left, right and camera center.
- Left and right x positions were calculated by fitting equation at y of bottom image.
- Camera center x position was calculated by the middle of the image that was 640[pixel].



#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 322 through 323 in my code in `find_lanes.py` in the function `process_image()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

When line candidate points are not taken correctly, my pipeline will likely fail.  
To avoid this,
1. color thresholded
2. gradient thresholded
3. line detect
4. recover

those steps are important.

**color threshold**  
I used only s channel of hsv color space. But LUV or LAB might be helpful under difficult light condition.  

**gradient thresholded**  
When line color and ground color are similar, gradient detection might be difficult.  Before applying cv2.COLOR_RGB2GRAY, convert RGB to another color space, then to GRAY might be helpful.

**line detect**  
When noisy line candidate points are collected, curve fitting might fail. So I set window margin size to 50 pixels(smaller than tutorial code).

**recover**  
I stocked previous 10 frames. In case the line was lost, I used the previous best fit parameter.  
I also average 10 frames parameter so that line curve to be stable.  
