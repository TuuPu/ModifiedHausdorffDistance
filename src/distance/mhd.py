from scipy.spatial import cKDTree
from dataset import image_processing




#d(a, B) = min_(b in B) || a - b ||
#decided to use a kd-tree to find min distances. This makes the process much faster than
#actually going through every single pixel
#from a test image and then calculating distances to training images.
#kd-tree returns all min distances for pixels
# in and image compared to another image.
def calculate_minimum_distance_pairwise(test_image, training_image):
    coordinates_test_img = image_processing.coordinates(test_image)
    coordinates_training_img = image_processing.coordinates(training_image)
    tree1 = cKDTree(coordinates_test_img)
    tree2 = cKDTree(coordinates_training_img)
    # returns min dist for every pixel of test_image
    distance_1 = tree1.query(coordinates_training_img)[0]
    # returns min dist for every pixel of training_image
    distance_2 = tree2.query(coordinates_test_img)[0]
    return distance_1, distance_2

#Calculates d(A, B), d(B, A) and f2
def mhd_d22(test_image, training_image):
    distance_1, distance_2 = calculate_minimum_distance_pairwise(test_image, training_image)
    # calculates d6 of mhd d(A, B)
    distance_1 = distance_1.mean()
    # calculates d6 of mhd the other way around d(B, A)
    distance_2 = distance_2.mean()
    # calculates f2 of mhd. max(d(A,B), d(B,A))
    return max(distance_1, distance_2)







#Tuesday: 10-18 Trying to figure out how to calculate MHD,
# failed miserably at first by trying to apply the calculation
#from a picture over a picture set,
# before realising it is the wrong way and now trying to figure out
# how to calculate the
#the distances over coordinates of pixels. Hope this is right.

#Wednesday 10 - 15 Built a function to get edge
# images for easier distance calculations.
# Once I got that done and made
# sure it works, I started building the actual calculations for the distances.
# Distance calculation for D22 are now complete and should work.
# Will keep working on unittests later on today or possibly tomorrow.
# This was an intensive couple of hours.
