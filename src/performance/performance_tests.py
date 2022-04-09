import statistics
import time
from distance import mhd

def calculate_distances_for_set(test_image, training_image_set):
    distance_list = []
    for i, image in enumerate(training_image_set):
        distance_list.append([mhd.mhd_d22(test_image, image), i])
    return distance_list


def time_with_mhd22_calculations(test_image, training_set, iterations):
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
