from scipy.spatial import cKDTree
from dataset import image_processing
from operator import itemgetter
import time




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

def mhd_D23(test_image, training_image):
    distance_1, distance_2 = calculate_minimum_distance_pairwise(test_image, training_image)
    distance_1 = distance_1.mean()
    distance_2 = distance_2.mean()
    function_3 = (distance_1 + distance_2)/2
    return function_3

def mhd_D23_without_mean(test_image, training_image):
    distance_1, distance_2 = calculate_minimum_distance_pairwise(test_image, training_image)
    distance_1 = distance_1.sum()
    distance_2 = distance_2.sum()
    function_3 = (distance_1 + distance_2)/2
    return function_3


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

def k_nearest(k, distance_list):
    '''
    Sorts array in ascending order and
    chooses k first elements. Also
    fetches indexes of the images chosen.
    '''
    sorted_list = sorted(distance_list, key = itemgetter(1))
    sorted_list = sorted_list[:k]
    indexes = [i[0] for i in sorted_list]
    return sorted_list, indexes







# Tuesday 9.30 - 17 working on different calculations and reading about k-nearest
# Wednesday 10 - 19 Getting k-nearest to work. And wondering why test set does not return 10k pictures.
