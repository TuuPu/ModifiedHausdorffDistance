from tensorflow.keras.datasets import mnist # pylint: disable=E0611, E0401
from dataset import image_processing
from distance import mhd
from collections import Counter
import time
import random
import numpy as np
# NOTE: Importing the mnist database takes about 7-10 seconds
# the program itself runs in about 1.5 seconds.

(x_train, y_train), (x_test, y_test) = mnist.load_data()

def calculate_distances_for_set(test_image, training_image_set):
    distance_list = []
    for i, image in enumerate(training_image_set):
        distance_list.append([i, mhd.mhd_d22(test_image, image)])
    return distance_list

def get_labels(indexes):
    labels = []
    for i in indexes:
        labels.append(int(i/1000))
    return labels

def main():
    '''
    Runs the program.

    On wednesday 6.4, I tried implementing a heap for quicker
    sorting of min-distances according to the value k.
    To my surprise using a heap was slower. The differences
    were between a half a second and a second. I used the
    heapq to return nsmallest(k, distance_list).
    I tried both ways, appending a list and then
    returning heapq.nsmallest(k, distance_list) and
    iterating through the training set and only pushing
    into the heap when a smaller distance was found after
    k:th iteration of the loop. The only times when the heap
    was actually faster were when k>300.
    Right now the run time is somewhere around 2.5 seconds, when
    value of k is sensible.
    '''

    testing_images, selected_test_labels = image_processing.sort_images_and_threshold(x_test, y_test)
    training_images, selected_train_labels = image_processing.sort_images_and_threshold(x_train, y_train)
    edge_training_set = image_processing.create_binary_edge_image(training_images)
    edge_testing_set = image_processing.create_binary_edge_image(testing_images)

    random_value = random.randint(0, 9786)

    start = time.time()
    distance_list = calculate_distances_for_set(edge_testing_set[random_value], edge_training_set)
    sorted_distances, indexes = mhd.k_nearest(5, distance_list)
    labels = get_labels(indexes)
    label = Counter(labels)
    label = label.most_common(1)[0][0]
    stop = time.time()
    print("time", stop-start)
    print(edge_testing_set[random_value])
    print(testing_images[random_value])
    print("actual label", selected_test_labels[random_value])
    print("value of random int", random_value)
    print("labels suggested", labels)
    print("label suggested", label)

if __name__ == "__main__":
    main()

