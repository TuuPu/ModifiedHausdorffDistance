## What have I done this week?

Most of my time went to building a framework for the UI and learning to use matplotlib in a sensible way (to make the UI work). I also made sure I have all comparable 
methods in use, that is: Three different distance calculations from Dubuisson et. al, D22, D23 and D23 without using mean of the set distances. Also I have three different 
sorting mechanisms in use that I can compare (Python's sorted(), heapq sort against a list and heapifying a list and then sorting). And lastly I can compare accuracies 
between using edge images and normal binary images. I also made the distance tests more sensible, creating a simple 2x3 matrix, calculating distances and comparing it to my 
functions value returned.

## How did I move progres?

Performance tests are pretty much done, technical side is done, UI is almost done. Next week I will finish up the UI and make testing even better and cover the rest of the 
parts that need to be covered.

## What was hard/left unclear?

Right now my only problem is writing a test for the mhd.py def k_nearest_with_complete_heap. I could write a general test where I see if it actually returns the list in 
right order but it seems a bit vague. But it relies so much on the heapq system, that it is hard to break down and test throughoutly. I will see if I come up with something 
next week.

## Next week:

UI and tests are next weeks tasks. I'm planning to run performance tests and saving the results as lists and using them to present general statistics when one runs my 
program. This is sensible, since running the performance tests takes around 40 minutes. Along with the general statistics, there will be an animation which runs the 
program on a random test image and shows the image and below it a bar chart which displays the "voting" of the number to show that it actually works.


## Time spent this week:

Around 16 hours this week.
