from scipy.spatial import cKDTree
from dataset import image_processing





def calculate_minimum_distance_pairwise(test_image, training_image):
    '''
    Calculates d(a, B) = min_(b in B) || a - b ||
    for every point in the image using kd-Tree.
    Distance calculations use matrices coordinates.
    Takes in a test image and training image
    and returns min distances forwards and backwards
    '''
    coordinates_test_img = image_processing.coordinates(test_image)
    coordinates_training_img = image_processing.coordinates(training_image)
    tree1 = cKDTree(coordinates_test_img)
    tree2 = cKDTree(coordinates_training_img)
    # returns min dist for every pixel of test_image
    distance_1 = tree1.query(coordinates_training_img)[0]
    # returns min dist for every pixel of training_image
    distance_2 = tree2.query(coordinates_test_img)[0]
    return distance_1, distance_2


def mhd_d22(test_image, training_image):
    '''
    Calculates
    d6 d(A, B) = 1/N_a Sigma_(a in A) d(a, B)
    and
    f2(d(A, B), d(B, A) = max(d(A, B), d(A, B))
    When given a test image and a training image.
    Also calls a function to calculate the min distance pairwise.
    '''
    distance_1, distance_2 = calculate_minimum_distance_pairwise(test_image, training_image)
    distance_1 = distance_1.mean()
    distance_2 = distance_2.mean()
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

#Friday 10-... Worked on tests, pylint and documentation
