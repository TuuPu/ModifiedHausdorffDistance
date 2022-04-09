## Structure

### distance/mhd

Contains calculations of Modified Hausdorff Distance. Includes D22, D23 and D23 without mean. Also has pairwise distance calculations and various sorting mechanisms distance lists.

See [calculations](http://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=9686DAF6AAA355EAE3F31A53B42076EA?doi=10.1.1.1.8155&rep=rep1&type=pdf)

### dataset/image_processing

Used for manipulating images to a binary form and edge images. Also creates arrays with correct coordinates from binary matrices.

### performance/performance_tests

Runs performance tests for various iteration sizes and k-values.

Note: I will add more detail to structure later on. Including function-wise comments.

## Time complexity

For pairwise distance calculations:

Inserting into kd-tree is O(log n)
And searching from kd-tree is also O(log n)

n is the number of pixels in an image.

Sorting:

Sorting with python's sorted is O(n log n)
Sorting with heapq.nsmallest is O(n+k log n)

n is the number of distances calculated
k is the number of distances wanted.

Information about heapq's time complexities were hard to find, I will read about it more in the upcoming weeks.

[kd-tree](https://en.wikipedia.org/wiki/K-d_tree)

[Python's sort](https://drops.dagstuhl.de/opus/volltexte/2018/9467/pdf/LIPIcs-ESA-2018-4.pdf)

[Heapq nsmallest](https://johnlekberg.com/blog/2020-11-01-stdlib-heapq.html)
