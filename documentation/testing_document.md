## Performance tests

Performance tests are up to par now. Explanation below

- Top row, leftmost: Sorting times. Done over 5 loops and looping again over 8 different k-values for one picture (picture changes between the five loops). Every sort for 
one k-value is performed a 100 times again, to get averages from a large amount of samples.
- Top row, middle: Distance calculation times. Iterated 100 times and distance calculation used is D22, a test image against 10k training images on every loop.
- Top row, rightmost: Distance calculation times. Same as before but uses heap structure, so keep in mind that this method also sorts and chooses k-shortest distances. To 
compare this fairly to D22 calculation times, you should add up sort times to D22 calculation times.
- Bottom row, leftmost: Accuracy calculations for three different distance calculation methods: D22, D23 and D23 with no mean (1/N_a) included. Tested over 100 test images 
and calculates accuracies for 8 different k-values. Also this accuracy calculation uses edge images.
- Bottom row, middle: Same as before for accuracy calculations but uses ordinary binary images, no edge images.


![alt text](https://github.com/TuuPu/ModifiedHausdorffDistance/blob/main/documentation/images/PerformanceStats.png)

(Tested on 22.4.2022)

## Notes about performance test results

I haven't found a reason for the jitter in small k-value sort times in the first graph. It comes and goes when I run the tests. This could be some type of overhead thing or 
just purely my computer as a bit of a cough when I run the program. Otherwise the performance tests seem reliable and sensible. I will be updating them every week to have 
the latest tests at hand.

Also I think I will modify the graphs to have more ticks on y-axis. I noticed that reading them can be a bit hard with such a large gap between ticks.

## Coverage report

Coverage report can be found from:

[CodeCov](https://app.codecov.io/gh/TuuPu/ModifiedHausdorffDistance)

Here's a picture of the coverage so far:

![alt text](https://github.com/TuuPu/ModifiedHausdorffDistance/blob/main/documentation/images/UpdatedCoverage.png)

## Unittests

### MNIST

The MNIST dataset has been tested to return in correct shapes (60000, 28, 28) (60000, ) for training images and (10000, 28, 28) (10000, ) for test images

### Images

Images have been tested to return in correct for by having a previously built image and comparing it to the return value of thresholded image.

### Coordinates

Coordinates have been tested by creating a set of edge images and then returning one of them and comparing the shape to set value. This is a bit tricky, since coordinate shapes can differ between images and a general random testing is quite hard.

### Pairwise distance

This is a test between two images. It creates two simple 2x3 matrices and calculates distances between the matrices manually, then calls a function for the matrices and 
compares if the distance calculations match.

### mhd_d22, mhd_d23, mhd_d23 without mean

These have been basically been tested like pairwise distance. Creating two simple matrices and doing the calculations manually and once again comparing the results to the 
functions being tested.

### sorting tests

K-nearest with python's sort has been tested by figuring out a distance list and it's correct order. Then running the sort for that list in no particular order and testing that the return order should be the same. Same has been done for heapq.nsmallest search test. YET TO BE TESTED: Complete heap sort and heapify sort. Kind of ran out of time, so these will be added later.

## How can the test be run?

```
poetry shell
pytest
```

in the root directory.

## Problems with testing

Right now I'm running the tests on specific images and I'm wondering how I could make them more general and be assured everything works correctly. It is hard to find alternative setups for distance calculations where I could give any two images and then compare the distance result to some value it should match.

If there are any tips and tricks for this, I would greatly appreciate them.
