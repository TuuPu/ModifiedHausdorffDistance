Yksikkötestauksen kattavuusraportti.
Mitä on testattu, miten tämä tehtiin?
Minkälaisilla syötteillä testaus tehtiin (vertailupainotteisissa töissä tärkeää)?
Miten testit voidaan toistaa?
Ohjelman toiminnan empiirisen testauksen tulosten esittäminen graafisessa muodossa.

## Coverage report

Coverage report can be found from:

[CodeCov](https://app.codecov.io/gh/TuuPu/ModifiedHausdorffDistance)

Here's a picture of the coverage so far:



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
