#pylint: disable-all
from collections import Counter
import random
from tensorflow.keras.datasets import mnist # pylint: disable=E0611, E0401
import numpy as np
from dataset import image_processing
from distance import mhd
from performance import performance_tests
import matplotlib.pyplot as plt
import time
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

def main():
    testing_images, selected_test_labels = \
        image_processing.sort_images_and_threshold(x_test, y_test)
    training_images, selected_train_labels = \
        image_processing.sort_images_and_threshold(x_train, y_train)
    edge_training_set = image_processing.create_binary_edge_image(training_images)
    edge_testing_set = image_processing.create_binary_edge_image(testing_images)
    #random_value = random.randint(0, 9786)
    #edge_testing_image = edge_testing_set[random_value]
    #distance_list = calculate_distances_for_set(edge_testing_image, edge_training_set)

    '''Just a test for one random picture'''
    '''
    rand_val = random.randint(0, 9785)
    distance_list = \
        calculate_distances_for_set\
            (edge_testing_set[rand_val], edge_training_set)
    sorted_list, indexes = \
        mhd.k_nearest_with_heap_search(20, distance_list)
    labels, label, label_amount = \
        get_labels(indexes)
    for k in label_amount:
        print(k, label_amount[k])
    '''

    '''
    Animated for one pic at a time x 40
    and shows the suggested labels on
    a bar chart below the image.
    '''

    fig, axs = plt.subplots(2)
    value_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                  12, 13, 14, 15, 16, 17, 18, 19, 20]
    k_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    for i in range(40):
        rand_val = random.randint(0, 9785)
        axs[0].cla()
        axs[1].cla()
        axs[0].imshow(testing_images[rand_val])
        distance_list = calculate_distances_for_set\
            (testing_images[rand_val], training_images)
        sorted_list, indexes = \
            mhd.k_nearest_with_heap_search(20, distance_list)
        labels, label, label_amount = get_labels(indexes)
        axs[1].bar(label_amount.keys(), label_amount.values())
        axs[1].set_xticks(k_values)
        axs[1].set_yticks(value_list)
        axs[0].set_title("frame {}".format(i))
        axs[1].set_title("frame {}".format(i))
        plt.pause(0.8)
    '''
    Performance test
    Takes about 40 mins to run
    1. Runs 5 x 8 loops for images and iterates
    every image 100 times to count avg sort times.
    
    2. Calculates run times for calculating
    distances between a testing images and
    training set. Iterated 100 times.
    
    3. Calculates accuracies for three
    different distance calculations
    and also calculates accuracies
    when edge images are used and 
    not used.
    '''
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
                performance_tests.time_with_pythons_sort(dist_list, 100, k)
            heap_mean, heap_max, heap_min = \
                performance_tests.time_with_heap_search(dist_list, 100, k)
            heapify_mean, heapify_max, heapify_min = \
                performance_tests.time_with_using_heapify(dist_list, 100, k)
            fivek_mean_py_list[i, ik] = mean_py
            fivek_heap_mean_list[i, ik] = heap_mean
            fivek_heapify_mean_list[i, ik] = heapify_mean

    py_sort_mean = np.mean(fivek_mean_py_list, axis=0)
    heap_sort_mean = np.mean(fivek_heap_mean_list, axis=0)
    heapify_sort_mean = np.mean(fivek_heapify_mean_list, axis=0)


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
    percentages = np.mean(hits_table, axis=0)
    d23_percentages = np.mean(d23_hits_table, axis=0)
    d23_no_mean_percentages = np.mean(d23_no_mean_hits_table, axis=0)

    no_edge_prctg = \
        np.mean(hits_table_no_edge, axis=0)
    d23_no_edge_prctg = \
        np.mean(d23_hits_no_edge, axis=0)
    no_mean_no_edge_prctg = \
        np.mean(d23_no_mean_hits_table, axis=0)



    box_plot_secs = [1, 2, 3, 4, 5]
    bar_plot_y_axis = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    x_axis = np.arange(len(k_values))
    fig, axs = plt.subplots(2, 3)

    axs[0, 0].plot(k_values, py_sort_mean)
    axs[0, 0].plot(k_values, heap_sort_mean)
    axs[0, 0].plot(k_values, heapify_sort_mean)
    axs[0, 0].set_xlabel("k-values")
    axs[0, 0].set_ylabel("Seconds")
    axs[0, 0].set_title("Sort times")
    axs[0, 0].legend(['sorted()', 'heapq', 'heapify'])
    axs[0, 1].boxplot(time_list)
    axs[0, 1].set_title("Distance calculation times")
    axs[0, 1].set_ylabel("Seconds")
    axs[0, 1].set_yticks(box_plot_secs)
    axs[0, 2].boxplot(time_list_heap)
    axs[0, 2].set_title("Distance calculations with heap, k=5")
    axs[0, 2].set_ylabel("Seconds")
    axs[0, 2].set_yticks(box_plot_secs)
    axs[1, 0].bar\
        (x_axis +0.20, percentages, width=0.2)
    axs[1, 0].bar\
        (x_axis +0.20*2, d23_percentages, width=0.2)
    axs[1, 0].bar\
        (x_axis +0.20*3, d23_no_mean_percentages, width=0.2)
    axs[1, 0].set_xticks(x_axis, k_values)
    axs[1, 0].set_xlabel("k-values")
    axs[1, 0].set_ylabel("percentage")
    axs[1, 0].set_yticks(bar_plot_y_axis)
    axs[1, 0].legend(['D22', 'D23', 'D23 no mean'])
    axs[1, 1].bar\
        (x_axis -0.20, no_edge_prctg, width=0.2)
    axs[1, 1].bar\
        (x_axis, d23_no_edge_prctg, width=0.2)
    axs[1, 1].bar\
        (x_axis +0.20, no_mean_no_edge_prctg, width=0.2)
    axs[1, 1].set_xticks(x_axis, k_values)
    axs[1, 1].set_xlabel("k-values")
    axs[1, 1].set_ylabel("percentages")
    axs[1, 1].set_yticks(bar_plot_y_axis)
    axs[1, 1].legend(['D22', 'D23', 'D23 no mean'])
    axs[1, 2].set_visible(False)
    plt.show()
    '''
if __name__ == "__main__":
    main()
