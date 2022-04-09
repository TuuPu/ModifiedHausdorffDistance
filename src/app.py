from collections import Counter
import random
from tensorflow.keras.datasets import mnist # pylint: disable=E0611, E0401
import numpy as np
from dataset import image_processing
from distance import mhd
from performance import performance_tests
# NOTE: Importing the mnist database takes about 7-10 seconds
# the program itself runs in about 1.5 seconds.

(x_train, y_train), (x_test, y_test) = mnist.load_data()

def calculate_distances_for_set(test_image, training_image_set):
    '''
    Calculates distances for a test image against
    10k training images
    '''
    distance_list = []
    for i, image in enumerate(training_image_set):
        distance_list.append([mhd.mhd_d22(test_image, image), i])
    return distance_list

def get_labels(indexes):
    '''
    Fetches labels and most common label
    AKA fetches the number that the  k-NN
    recommends the image to be
    '''
    labels = []
    for i in indexes:
        labels.append(int(i/1000))
    label = Counter(labels)
    label = label.most_common(1)[0][0]
    return labels, label

def main():
    '''
    Runs the program and stress tests.
    Stress tests include:

    running a loop of n iterations and calculating
    the mean, max and min of:
    -calculation time
    -every sorting style I have used

    Accuracy of predictions using the fastest sort.
    Accuracy is calculated for k-values:
    1, 3, 5, 11, 15, 21, 51 and 101.

    Also the program calculates the avg of one
    pairwise distance calculation.

    WARNING: running the program takes several minutes
    because the accuracy test runs a loop of 100 iterations
    and the calculation loops run twice when forming the
    run time averages.
    One calculation takes approx 2.5 secs and the calculations
    run three times, so that's (2.5*100)*3 seconds.

    If you want to test the program with one image, comment everything
    away between lines 84 and 127 and copy this:

    sorted_distances, indexes = mhd.k_nearest_with_heap_search(5, distance_list)
    labels, label = get_labels(indexes)
    print(edge_testing_image) #Image tested
    print(selected_test_labels[random_value]) #The number the image represents
    print(label) #The number k-NN suggests the test image to be.

    If you wish, you can change the value 5 in
    k_nearest_with_heap_search
    to test different k-values.
    '''

    testing_images, selected_test_labels = \
        image_processing.sort_images_and_threshold(x_test, y_test)
    training_images, selected_train_labels = \
        image_processing.sort_images_and_threshold(x_train, y_train)
    edge_training_set = image_processing.create_binary_edge_image(training_images)
    edge_testing_set = image_processing.create_binary_edge_image(testing_images)
    random_value = random.randint(0, 9786)
    edge_testing_image = edge_testing_set[random_value]
    distance_list = calculate_distances_for_set(edge_testing_image, edge_training_set)

    mean_calc, avg_for_one, max_calc, min_calc = performance_tests.time_with_mhd22_calculations\
        (edge_testing_image, edge_training_set, 100)
    print("Calculation time when n=100")
    print("Mean: ", mean_calc)
    print("Max: ", max_calc)
    print("Min: ", min_calc)
    print("Avg for one pair: ", avg_for_one)
    mean_py, max_py, min_py = performance_tests.time_with_pythons_sort(distance_list, 100, 5)
    print("Sorting with python's sorted command n=100, k=5")
    print("Mean: ", mean_py)
    print("Max: ", max_py)
    print("Min: ", min_py)
    heap_mean, heap_max, heap_min = performance_tests.time_with_heap_search(distance_list, 100, 5)
    print("Sorting with heapq.nsmallest function straight from a list n=100, k=5")
    print("Mean: ", heap_mean)
    print("Max: ", heap_max)
    print("Min: ", heap_min)
    comp_mean, comp_max, comp_min = performance_tests.time_with_complete_heap_sort\
        (edge_testing_image, edge_training_set, 100, 5)
    print("Sorting AND calculating distances with heap n=100, k=5")
    print("Mean: ", comp_mean)
    print("Max: ", comp_max)
    print("Min: ", comp_min)
    heapify_mean, heapify_max, heapify_min = performance_tests.time_with_using_heapify\
        (distance_list, 100, 5)
    print("Changing a list to a heap type structure using heapify "
          "and then returning values n=100, k=5")
    print("Mean: ", heapify_mean)
    print("Max: ", heapify_max)
    print("Min: ", heapify_min)

    num_test_images = 100
    k_values = [1, 3, 5, 11, 15, 21, 51, 101]
    hits_table = np.zeros((num_test_images, len(k_values)))
    for i in range(100):
        rand_val = random.randint(0, 9786)
        distance_list = calculate_distances_for_set(edge_testing_set[rand_val], edge_training_set)
        for k_ind, k in enumerate(k_values):
            sorted_distances, indexes = mhd.k_nearest_with_heap_search(k, distance_list)
            labels, most_common_label = get_labels(indexes)
            hits_table[i, k_ind] = int(most_common_label == selected_test_labels[rand_val][0])
    percentages = np.mean(hits_table, axis=0)
    for idx, i in enumerate(percentages):
        print("k-", k_values[idx], ", accuracy%: ", i)

if __name__ == "__main__":
    main()
