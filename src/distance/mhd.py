from operator import itemgetter
import heapq
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

def mhd_d23(test_image, training_image):
    '''
    Calculates D23 version of distance.
    Differs from D22 only by not taking
    max(d(A, B), d(B, A)) but calculating
    (d(A, B) + d(B, A))/2
    '''
    distance_1, distance_2 = calculate_minimum_distance_pairwise(test_image, training_image)
    distance_1 = distance_1.mean()
    distance_2 = distance_2.mean()
    function_3 = (distance_1 + distance_2)/2
    return function_3

def mhd_d23_without_mean(test_image, training_image):
    '''
    Exactly like D23 but does not take a mean
    of the pairwise distances.
    '''

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
    sorted_list = sorted(distance_list, key = itemgetter(0))
    sorted_list = sorted_list[:k]
    indexes = [i[1] for i in sorted_list]
    return sorted_list, indexes

def k_nearest_with_heap_search(k, distance_list):
    '''Only using heap search'''
    list_of_distances = heapq.nsmallest(k, distance_list)
    indexes = [i[1] for i in list_of_distances]
    return list_of_distances, indexes

def k_nearest_with_complete_heap(k, test_image, training_set):
    '''Uses heap insert and sort
    and tries to limit the amount
    of elements in heap
    '''
    distance_list = []
    for idx, image in enumerate(training_set):
        distance = mhd_d22(test_image, image)
        if idx == 0:
            heapq.heappush(distance_list, distance)
        else:
            if distance < heapq.nsmallest(k, distance_list)[-1]:
                heapq.heappush(distance_list, distance)
    return heapq.nsmallest(k, distance_list)

def k_nearest_with_heapify(k, distance_list):
    '''First turns the list data structure
    to a heap and then fetches
    the k-smallest distances'''
    heapq.heapify(distance_list)
    sorted_distances = heapq.nsmallest(k, distance_list)
    return sorted_distances
