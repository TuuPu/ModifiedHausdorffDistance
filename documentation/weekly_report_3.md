## What have I done this week/process:

Read a lot about MHD and tried to understand the practical implementation better. First I wasted quite a bit of time by calculating the distances between an image and and image set. Then realised it is the wrong approach and went back to calculating the distance from a pixel to an image.

I created more image processing functions, mainly a function for morphing the images so that only outer lines are left (since the point of the paper from Dubuisson is mainly about edge detection). Then created a function for getting the coordinates of the the pixels on these outer lines of images.

After doing that I went to work on the actual calculations. Right now the mhd.py calculates the D22 distance by first calculating the min distances between points and a picture (both ways from a test image to training and vice versa). Then calculates the d6 distance between two sets and finally takes the f2 function of two distances and returns the max distance from those. Seems to work pretty well, I'm proud of that!

## What did I learn?

The main thing I actually learned was that I finally understood that this is about edge detection mostly. Dubuisson's examples with images of moving cars made me realise this. And that even though MHD isn't a metric, it uses Euclidean distances in the point to set distance calculation. This helped me greatly  and steered me to use the coordinates.

## What was left unclear?

Not much this week. I'm fairly confident that the rest of the distance calculations (D23 as it is and without the 1/N) are fairly simple to implement and also that the k-nearest check is going to be fairly straight forward. I'm only worrying time consumption in the end and hope that everything goes smoothly with it.

## What's next?

Next week I'm going to (hopefully) quickly do the rest of the distance calculations for comparisons and complete the K-nearest algorithm for actual results.

## Working hours this week:

Tuesday: 8 hours (10-18)

Wednesday: 5 hours (10-15)

Friday: 5 hours (10-15)

Total: 18 hours
