## Performance tests

As of now, performance tests have been generally tested with 100 iterations on different parts of the program, using k-value of 5 if not stated otherwise. My plan is to test more with different iterations and k-values in the future. Below you can read an explanation and the results of the tests.

- 100 iterations of calculating distances for one test image against 10k training images
- 100 iterations of sorting with python's sorted function
- 100 iterations of sorting with calling heapq.nsmallest for a list-type structure
- 100 iterations of sorting AND calculating distances with heapq heap.
- 100 iterations of using heapq's heapify and calling for k-shortest distances.

|n=100, k=5     |Mean |Max  |Min  |
|---------------|-----|-----|-----|
|Calc. distance |2.66s|5.59s|2.38s|
|---------------|-----|-----|-----|
|Python sort    |3,4ms|5,8ms|3,3ms|
|---------------|-----|-----|-----|
|heapq.nsmallest|0.6ms|1,1ms|0,6ms|
|---------------|-----|-----|-----|
|complete heap  |2.53s|2.77s|2.49s|
|---------------|-----|-----|-----|
|heapify        |1.8ms|5.1ms|1.7ms|

I was surprised about the results (this particular test was done on saturday (9.4)), because usually the avg. for calculating a distance has been lower than using the heap, which calculates and sorts distances. I will do more tests and see what comes up next time.

Below you can also see a table of accuracies for different k-values. All accuracy tests have been done over 100 test images compared to 10k training images (NOTE: tests only done with edge-images, full binary image tests later on).

|k/accuracy|%   |
|----------|----|
|k=1       |0.94|
|----------|----|
|k=3       |0.95|
|----------|----|
|k=5       |0.93|
|----------|----|
|k=11      |0.94|
|----------|----|
|k=15      |0.94|
|----------|----|
|k=21      |0.93|
|----------|----|
|k=51      |0.90|
|----------|----|
|k=101     |0.88|

EXTRA NOTE: These tests have been ran with D22 calculation. I will be running the other distance calculations in the future to compare accuracies.

## Coverage report

Coverage report can be found from:

[CodeCov](https://app.codecov.io/gh/TuuPu/ModifiedHausdorffDistance)

Here's a picture of the coverage so far:

![alt text](https://github.com/TuuPu/ModifiedHausdorffDistance/blob/main/documentation/images/CoverageReportCodeCov.png)

## What has been tested so far and how?

Parts of the program I have tested:

1. Loading data
	1. Tested by calling the mnist.load_data() and comparing the shape of the arrays to arrays it should present
2. That test images are in correct shape
	1. Returning an image from the MNIST dataset and comparing it to a previously made image that it should match
3. Testing conditional statements between training and testing set
	1. Testing that the indexes (according to labels) return the correct length arrays.
4. That after converting images to edge images the arrays return in correct form
	1. Returning the dataset and comparing its shape to the value it should be
5. Testing that when forming an image to its coordinates it is in correct form
	1. Returning an image and comparing its shape to a value it should be in
6. Testing the distance between point and a set
	1. Running the calculations "manually" for two images and then calling the function which calculates the distance and comparing that the values match
7. Testing the actual D22 calculation
	1. Running the calculations "manually" for two images and then calling the function which calculates the distance and comparing that the values match

## How can the test be run?

```
poetry shell
pytest
```

in the root directory.

## Problems with testing

Right now I'm running the tests on specific images and I'm wondering how I could make them more general and be assured everything works correctly. It is hard to find alternative setups for distance calculations where I could give any two images and then compare the distance result to some value it should match.

If there are any tips and tricks for this, I would greatly appreciate them.
