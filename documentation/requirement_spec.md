# Requirement Specification

**Language:** English

**Programming language:** Python

**Programming languages I know well:** Java

**Study program:** Tietojenkäsittelytieteen kandidaatti

## Subject

The basic idea is to build a program that uses handwritten digits from the MNIST database and performs object recognition to those numbers.

The aim is to first modify the pictures from grey scale to black and white pictures, using a treshold. The actual recognition part is done by using K nearest neighbors (k-NN) algorithm and the nearest neighbors are identified by using the Modified Hausdorff Distance technique ([MHD](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.1.8155&rep=rep1&type=pdf)).

I am going to use the D22 measurement (MHD) and compare it to the D23 measurement without a factor 1/N in d6 formula.

Visualisation has not yet been planned but I would assume it includes the comparison between D22 and D23. On top of that probably some visualisation on the accuracy of predictions between different values of k.

I chose this project because I have an interest towards data science and machine learning. This seemed like a first real opportunity to build such a project.
