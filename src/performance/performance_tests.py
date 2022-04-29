#pylint: disable=W0612
import statistics
import numpy as np
import time
from collections import Counter
import random
from distance import mhd




def calculate_distances_for_set(test_image, training_image_set):
    '''Calculates distances for a test image against 10k
    training images'''
    distance_list = []
    for i, image in enumerate(training_image_set):
        distance_list.append([mhd.mhd_d22(test_image, image), i])
    return distance_list

def calculate_distances_for_set_mhd23(test_image, training_image_set):
    distance_list = []
    for i, image in enumerate(training_image_set):
        distance_list.append([mhd.mhd_d23(test_image, image), i])
    return distance_list

def calculate_distancses_for_set_mhd23_wo_mean(test_image, training_image_set):
    distance_list = []
    for i, image in enumerate(training_image_set):
        distance_list.append([mhd.mhd_d23_without_mean(test_image, image), i])
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
    label_amount = Counter(labels)
    label = label.most_common(1)[0][0]
    return labels, label, label_amount


def time_with_mhd22_calculations(test_image, training_set, iterations):
    '''Tracks time for calculations'''
    math_time_list = []
    for iteration in range(iterations):
        math_time_start = time.time()
        calculate_distances_for_set(test_image, training_set)
        math_time_stop = time.time()
        math_time_list.append(math_time_stop-math_time_start)
    mean_math_time = statistics.mean(math_time_list)
    avg_time_per_calc = mean_math_time/10000
    max_math_time = max(math_time_list)
    min_math_time = min(math_time_list)
    return mean_math_time, avg_time_per_calc, max_math_time, min_math_time

def time_with_pythons_sort(distance_list, iterations, k):
    '''Tracks time for sorting with sorted()'''
    sort_list_python = []
    for iteration in range(iterations):
        sort_time_start = time.time()
        mhd.k_nearest(k, distance_list)
        sort_time_stop = time.time()
        sort_list_python.append(sort_time_stop-sort_time_start)
    mean_sort_time_python = statistics.mean(sort_list_python)
    max_sort_time_python = max(sort_list_python)
    min_sort_time_python = min(sort_list_python)
    return mean_sort_time_python, max_sort_time_python, min_sort_time_python

def time_with_heap_search(distance_list, iterations, k):
    '''Tracks time for sorting with heapq.nsmallest'''
    sort_list_heap_sort = []
    for iteration in range(iterations):
        sort_time_start = time.time()
        mhd.k_nearest_with_heap_search(k, distance_list)
        sort_time_stop = time.time()
        sort_list_heap_sort.append(sort_time_stop-sort_time_start)
    mean_sort_time = statistics.mean(sort_list_heap_sort)
    max_sort_time = max(sort_list_heap_sort)
    min_sort_time = min(sort_list_heap_sort)
    return mean_sort_time, max_sort_time, min_sort_time

def time_with_complete_heap_sort(image, training_set, iterations, k):
    '''Tracks time for sorting and calculating distances
    with complete heap-structure
    '''
    sort_list_complete_heap = []
    for iteration in range(iterations):
        sort_time_start = time.time()
        mhd.k_nearest_with_complete_heap(k, image, training_set)
        sort_time_stop = time.time()
        sort_list_complete_heap.append(sort_time_stop - sort_time_start)
    mean_sort_time = statistics.mean(sort_list_complete_heap)
    max_sort_time = max(sort_list_complete_heap)
    min_sort_time = min(sort_list_complete_heap)
    return mean_sort_time, max_sort_time, min_sort_time

def time_with_using_heapify(distance_list, iterations, k):
    '''Tracks time for modifying the list into a
    heap and then fetching k-smallest distances'''
    sort_list_using_heapify = []
    for iteration in range(iterations):
        sort_time_start = time.time()
        mhd.k_nearest_with_heapify(k, distance_list)
        sort_time_stop = time.time()
        sort_list_using_heapify.append(sort_time_stop-sort_time_start)
    mean_sort = statistics.mean(sort_list_using_heapify)
    max_sort = max(sort_list_using_heapify)
    min_sort = min(sort_list_using_heapify)
    return mean_sort, max_sort, min_sort

def sort_time_means(edge_testing_set, edge_training_set):
    '''
    Calculates average sort times for all sorting methods
    :param edge_testing_set: images from testing set
    :param edge_training_set: images from training set
    :return: averages (time) of different sort methods
    '''
    k_values = [1, 3, 5, 11, 15, 21, 51, 101]
    fivek_mean_py_list = np.zeros((5, len(k_values)))
    fivek_heap_mean_list = np.zeros((5, len(k_values)))
    fivek_heapify_mean_list = np.zeros((5, len(k_values)))
    for i in range(5):
        for ik, k in enumerate(k_values):
            random_val = random.randint(0, 9785)
            edge_testing_img = edge_testing_set[random_val]
            dist_list = \
                calculate_distances_for_set(edge_testing_img, edge_training_set)
            mean_py, max_py, min_py = \
                time_with_pythons_sort(dist_list, 100, k)
            heap_mean, heap_max, heap_min = \
                time_with_heap_search(dist_list, 100, k)
            heapify_mean, heapify_max, heapify_min = \
                time_with_using_heapify(dist_list, 100, k)
            fivek_mean_py_list[i, ik] = mean_py
            fivek_heap_mean_list[i, ik] = heap_mean
            fivek_heapify_mean_list[i, ik] = heapify_mean
    return np.mean(fivek_mean_py_list, axis=0), np.mean(fivek_heap_mean_list, axis=0), \
           np.mean(fivek_heapify_mean_list,  axis=0)

def calculate_times(edge_testing_set, edge_training_set):
    '''
    Calculates time for distance calculations
    :param edge_testing_set:
    :param edge_training_set:
    :return:
    '''
    time_list = []
    time_list_heap = []
    for i in range(1, 101):
        random_val = random.randint(0, 9785)
        edge_testing_img = edge_testing_set[random_val]
        math_time_start = time.time()
        calculate_distances_for_set\
            (edge_testing_img, edge_training_set)
        math_time_stop = time.time()
        heap_time_start = time.time()
        mhd.k_nearest_with_complete_heap\
            (5, edge_testing_img, edge_training_set)
        heap_time_stop = time.time()
        time_list.append\
            (math_time_stop-math_time_start)
        time_list_heap.append\
            (heap_time_stop-heap_time_start)
    return time_list, time_list_heap

def calculate_all_accuracies(edge_testing_set, edge_training_set,
                             testing_images, training_images,
                             selected_test_labels):
    '''
    Calculates accuracies of different distance measures
    :param edge_testing_set: testing images (only edges)
    :param edge_training_set: training images (only edges)
    :param testing_images: full binary testing images
    :param training_images: full binary training images
    :param selected_test_labels: labels of found results
    :return: Accuracies for different distance calculations
    for both, edge and full  binary images
    '''
    k_values = [1, 3, 5, 11, 15, 21, 51, 101]
    num_test_imgs = 100
    hits_table = \
        np.zeros((num_test_imgs, len(k_values)))
    d23_hits_table = \
        np.zeros((num_test_imgs, len(k_values)))
    d23_no_mean_hits_table = \
        np.zeros((num_test_imgs, len(k_values)))
    hits_table_no_edge = \
        np.zeros((num_test_imgs, len(k_values)))
    d23_hits_no_edge = \
        np.zeros((num_test_imgs, len(k_values)))
    no_mean_no_edge_hits = \
        np.zeros((num_test_imgs, len(k_values)))
    for i in range(100):
        rand_val = random.randint(0, 9785)
        distance_list = \
            calculate_distances_for_set\
                (edge_testing_set[rand_val], edge_training_set)
        dist_mhd23_list = \
            calculate_distances_for_set_mhd23\
                (edge_testing_set[rand_val], edge_training_set)
        dist_mhd23_no_mean_list = \
            calculate_distancses_for_set_mhd23_wo_mean\
                (edge_testing_set[rand_val], edge_training_set)

        no_edge_dist = \
            calculate_distances_for_set\
                (testing_images[rand_val], training_images)
        d23_no_edge_dist = \
            calculate_distances_for_set_mhd23\
                (testing_images[rand_val], training_images)
        no_mean_no_edge_dist = \
            calculate_distancses_for_set_mhd23_wo_mean\
                (testing_images[rand_val], training_images)

        for k_ind, k in enumerate(k_values):
            sorted_distances, indexes = \
                mhd.k_nearest_with_heap_search(k, distance_list)
            mhd23_sort, idx23 = \
                mhd.k_nearest_with_heap_search(k, dist_mhd23_list)
            nomean_sort, nomean_idx = \
                mhd.k_nearest_with_heap_search(k, dist_mhd23_no_mean_list)

            no_edge_sorted, no_edge_idxs = \
                mhd.k_nearest_with_heap_search(k, no_edge_dist)
            d23_no_edge_sort, d23_idxs = \
                mhd.k_nearest_with_heap_search(k, d23_no_edge_dist)
            no_mean_sort, no_mean_idxs = \
                mhd.k_nearest_with_heap_search(k, no_mean_no_edge_dist)



            labels, most_common_label, \
            label_amount = get_labels(indexes)
            mhd23_labels, mhd23_most_common, \
            d23_amount = get_labels(idx23)
            nomean_labels, nomean_most_common, \
            nomean_amount = get_labels(nomean_idx)

            no_edge_labels, no_edge_mst_cmn_lbl, \
            no_edge_amount = get_labels(no_edge_idxs)
            d23_no_edge_lbls, d23_no_edge_lbl, \
            d23_no_edge_amount = get_labels(d23_idxs)
            no_mean_lbls, no_mean_lbl, \
            no_mean_amnt = get_labels(no_mean_idxs)

            hits_table_no_edge[i, k_ind] = \
                int(no_edge_mst_cmn_lbl == selected_test_labels[rand_val][0])
            d23_hits_no_edge[i, k_ind] = \
                int(d23_no_edge_lbl == selected_test_labels[rand_val][0])
            no_mean_no_edge_hits[i, k_ind] = \
                int(no_mean_lbl == selected_test_labels[rand_val][0])

            hits_table[i, k_ind] = \
                int(most_common_label == selected_test_labels[rand_val][0])
            d23_hits_table[i, k_ind] = \
                int(mhd23_most_common == selected_test_labels[rand_val][0])
            d23_no_mean_hits_table[i, k_ind] = \
                int(nomean_most_common == selected_test_labels[rand_val][0])
    return np.mean(hits_table, axis=0), np.mean(d23_hits_table, axis=0), \
           np.mean(d23_no_mean_hits_table, axis=0),np.mean(hits_table_no_edge, axis=0), \
           np.mean(d23_hits_no_edge, axis=0), np.mean(d23_no_mean_hits_table, axis=0)