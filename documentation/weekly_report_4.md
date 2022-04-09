## What have I done this week?

I managed to do all three different distance calculations and also built functions for different types of sorts. Using python's own "sorted" mechanism and heap structures. I also managed to spend quite a lot of time wondering why my test image set does not have 10k pictures, but I realized that I'm limitting them to 1k per label and the images by numbers aren't spread out evenly in the MNIST dataset. But I think 9786 test images is more than enough for random testing. The training images still hold 10k images as agreed. I also modified image-arrays in a way that I can easily get the label of the image. This has to be done so I can be sure that I can analyze the returned values.

## Process:

I'm almost done from the technical side. I just have to run the different calculations and try results between binary images as they are and edge images. But everything should pretty much done (except for the UI ofcourse). I have 3 different distance calculations and 4 different sorting mechanisms to compare between.

## What was hard/unclear?

Definitely trying to figure out where I could do better. I realized that the calculations are done against 10k images every time when a test is being done, so calculation time should stay pretty stable and I think kd-Tree is one of the fastest methods to do it. And I believe I have covered sorting methods pretty well. I'm pretty happy about the processing times but I have no benchmark to compare to, so I could be totally wrong.

## Next week:

I'm going to do more performance tests and build up the testing coverage with various tests. Also I think I'm going to start working on UI. That is going to be fun. I'm thinking about pretty statistically heavy UI that presents over all statistics for bigger test datasets and then a possibility to run the test for a random test image.



## Time spent this week:

Approx. 20 hours. Didn't track hours this week precisely.
