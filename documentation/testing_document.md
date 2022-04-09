## Performance tests

As of now, performance tests have been generally tested with 100 iterations on different parts of the program, using k-value of 5 if not stated otherwise. My plan is to test more with different iterations and k-values in the future. Below you can read an explanation and the results of the tests.

- 100 iterations of calculating distances for one test image against 10k training images
- 100 iterations of sorting with python's sorted function
- 100 iterations of sorting with calling heapq.nsmallest for a list-type structure
- 100 iterations of sorting AND calculating distances with heapq heap.
- 100 iterations of using heapq's heapify and calling for k-shortest distances.

|n=100, k=5     |Mean |Max  |Min  |Avg for pairwise calc.  |
|---------------|-----|-----|-----|------------------------|
|Calc. distance |2.66s|5.59s|2.38s|0.25ms                  |
|Python sort    |3,4ms|5,8ms|3,3ms|NaN                     |
|heapq.nsmallest|0.6ms|1,1ms|0,6ms|NaN                     |
|complete heap  |2.53s|2.77s|2.49s|NaN                     |
|heapify        |1.8ms|5.1ms|1.7ms|NaN                     |

I was surprised about the results (this particular test was done on saturday (9.4)), because usually the avg. for calculating a distance has been lower than using the heap, which calculates and sorts distances. I will do more tests and see what comes up next time. I was particulary surprised about the max. time of distance calculations.

Below you can also see a table of accuracies for different k-values. All accuracy tests have been done over 100 test images compared to 10k training images (NOTE: tests only done with edge-images, full binary image tests later on).

|k/accuracy|%   |
|----------|----|
|k=1       |0.94|
|k=3       |0.95|
|k=5       |0.93|
|k=11      |0.94|
|k=15      |0.94|
|k=21      |0.93|
|k=51      |0.90|
|k=101     |0.88|

EXTRA NOTE: These tests have been ran with D22 calculation. I will be running the other distance calculations in the future to compare accuracies.

## Coverage report

Coverage report can be found from:

[CodeCov](https://app.codecov.io/gh/TuuPu/ModifiedHausdorffDistance)

Here's a picture of the coverage so far:

![alt text](https://github.com/TuuPu/ModifiedHausdorffDistance/blob/main/documentation/images/CoverageReportCodeCov.png)

## Unittests

MNIST

The MNIST dataset has been tested to return in correct shapes (60000, 28, 28) (60000, ) for training images and (10000, 28, 28) (10000, ) for test images

Images

Images have been tested to return in correct for by having a previously built image and comparing it to the return value of thresholded image.

Coordinates

Coordinates have been tested by creating a set of edge images and then returning one of them and comparing the shape to set value. This is a bit tricky, since coordinate shapes can differ between images and a general random testing is quite hard.

Pairwise distance

This is a test between two images. It has been tested by running the function manually for a picture and then calling the actual function and testing if they return the same values. Once again, testing this randomly is pretty hard. I could try modifying the tested values to really simple values so that the calculations could be tested easily.

mhd_d22, mhd_d23, mhd_d23 without mean

These have been basically been tested like pairwise distance. Running the functions manually for two images, then calling the functions and comparing the images. Once again running some simple values where I could calculate the distances easily by hand and then running those values to test them would probably be better.

sorting tests

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
