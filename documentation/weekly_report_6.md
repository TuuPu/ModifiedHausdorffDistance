## What have I done this week?

Built UI to its final form, it can be now used either by using pre-gathered data (this will be used when doing a demo) or by running the latest data, which means basically 
running performance tests. As my UI is heavily based on graphs. It also has a looping function that updates the view with random images and shows how my program "votes" for 
the images. I also made documentation better and moved a few functions between modules to make the whole program a little more sensible. Testing has been now modified to 
use random 28x28 images for distance calculation tests.

## How did the program evolve?

UI changes and overall better testing and usability. Little changes here and there left only. So mostly working on making it a little more readable.

## What was left unclear?

I got comment's about testing/not testing heap. The basic problem with the heap I'm using is that it's a premade heap from Python's libraries. I wanted to test that it 
works correctly, and I want to use it to see different values in sorting and calculating so I can have a wide selection of datapoints in my UI. But I have decided (for now) 
not to test the heap in my test_mhd.py module. It's a premade heap structure and it seems to return values right, so I think I'll leave it as it is. Overall, it does not 
perform as well as using kd-tree for distance calculations and then sorting the list I get from kd-tree. So the heap just adds value in graphs.

## Next week

I will tidy up the project even more and fix the problem with graph colours changing with every loop (they don't match the legends of the graphs after first iteration). And 
see if there is more to be tested.

## Time spent this week

Total 13 hours
