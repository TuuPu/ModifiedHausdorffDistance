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
    Calculates  distances for a set using mhd D22
    :param test_image: test image to be tested
    :param training_image_set: training set of images
    :return: distances between test image and training set
    '''
    distance_list = []
    for i, image in enumerate(training_image_set):
        distance_list.append([mhd.mhd_d22(test_image, image), i])
    return distance_list

def calculate_distances_for_set_mhd23(test_image, training_image_set):
    '''
    Calculates distances for a set using mhd D23
    :param test_image: test image to be tested
    :param training_image_set: training set of images
    :return: distances between test image and training images
    '''
    distance_list = []
    for i, image in enumerate(training_image_set):
        distance_list.append([mhd.mhd_d23(test_image, image), i])
    return distance_list

def calculate_distancses_for_set_mhd23_wo_mean(test_image, training_image_set):
    '''
    Calculates distances for a set using mhd D23 without mean
    :param test_image: test image to be tested
    :param training_image_set: training set of images
    :return: distances between test image and training images
    '''
    distance_list = []
    for i, image in enumerate(training_image_set):
        distance_list.append([mhd.mhd_d23_without_mean(test_image, image), i])
    return distance_list

def get_labels(indexes):
    '''
    Used to find which images are found in k-nearest
    set.
    :param indexes: Indexes of images in a list
    :return: all labels of found matches, most common
    label and amount of all found labels
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
    Test values below have been gathered on 28.04.2022
    Plots can be drawn using these for quicker plotting
    e.g. no need to run performance tests to see how the
    UI will look like
    explanation of variables:
    
    py_sort_mean = sort time averages using python's sorted()
    heap_sort_mean = sort time using heapq's sort method
    heapify_sort_mean = turns a list into a heap and uses heapq's sort method
    time_list = calculation times
    time_list_heap = calculation times but uses heap to pick the right amount of distances
    different percentages = accuracies of predictions using different calculation methods
    and images.
    '''

    py_sort_mean = [0.0038259, 0.00386101, 0.00380454, 0.00379981,
                    0.00378461, 0.00382192, 0.00381077, 0.00406356]
    heap_sort_mean = [0.00051095, 0.00064698, 0.00065882, 0.00068283, 0.0006949,  0.0007486,
                      0.0009254,  0.00128585]
    heapify_sort_mean = [0.00176822, 0.00191377, 0.00199613,
                         0.00186642, 0.00186563, 0.00188232, 0.00192246, 0.0021892 ]
    time_list = [2.4359920024871826, 2.4784111976623535, 2.1383979320526123,
                 2.128864049911499, 2.380826711654663, 2.314026117324829,
                 2.4956979751586914, 2.277935266494751, 2.4299941062927246,
                 2.128006935119629, 2.143930196762085, 2.387631893157959,
                 2.603800058364868, 2.5084240436553955, 2.504251003265381,
                 2.1467559337615967, 2.542153835296631, 2.6290059089660645,
                 2.5101070404052734, 2.4011189937591553, 2.1348769664764404,
                 2.4728338718414307, 2.5703511238098145, 2.554208993911743,
                 2.3955161571502686, 2.3371477127075195, 2.3762669563293457,
                 2.4127681255340576, 2.4485650062561035, 2.6074349880218506,
                 2.564429759979248, 2.507097005844116, 2.173560857772827,
                 3.091848850250244, 4.362096309661865, 3.851698875427246,
                 3.3582730293273926, 2.8097259998321533, 2.375066041946411,
                 2.4793851375579834, 2.6838090419769287, 2.3103909492492676,
                 2.664947032928467, 2.6497421264648438, 2.9536941051483154,
                 2.88440203666687, 2.509298086166382, 2.36384916305542,
                 2.634658098220825, 2.619663953781128, 2.3447110652923584,
                 2.4700469970703125, 2.4804229736328125, 2.5257279872894287,
                 2.162435293197632, 2.426861047744751, 2.1649410724639893,
                 2.5703999996185303, 2.261463165283203, 2.3226678371429443,
                 2.401913642883301, 2.557936191558838, 2.361257314682007,
                 2.6952521800994873, 2.4115829467773438, 2.4356679916381836,
                 2.4134912490844727, 2.33894419670105, 2.355590343475342,
                 2.132258892059326, 2.1115262508392334, 2.3679161071777344,
                 2.554084062576294, 2.6085479259490967, 2.217747211456299,
                 2.5567378997802734, 2.5744850635528564, 2.242285966873169,
                 2.367542028427124, 2.4882559776306152, 2.70107102394104,
                 2.5392251014709473, 2.3874568939208984, 2.5580661296844482,
                 2.364421844482422, 2.4509530067443848, 2.30914306640625,
                 2.572899103164673, 2.5348548889160156, 2.248337984085083,
                 2.644655227661133, 2.5612540245056152, 2.3892197608947754,
                 2.5525457859039307, 2.459672212600708, 2.776792049407959,
                 2.5966320037841797, 2.1427628993988037, 2.859476089477539,
                 2.3953328132629395]
    time_list_heap = [2.5341272354125977, 2.6045777797698975, 2.2409169673919678,
                      2.292330026626587, 2.4791839122772217, 2.4221558570861816,
                      2.6120009422302246, 2.313959836959839, 2.552747964859009,
                      2.248044013977051, 2.2834107875823975, 2.612999200820923,
                      2.5345418453216553, 2.621958017349243, 2.6631150245666504,
                      2.24606990814209, 2.6518378257751465, 2.6754839420318604,
                      2.624392032623291, 2.5177512168884277, 2.2526540756225586,
                      2.6072609424591064, 2.6476659774780273, 2.7098548412323,
                      2.5194759368896484, 2.458778142929077, 2.5256378650665283,
                      2.521087884902954, 2.5566720962524414, 2.5134289264678955,
                      2.6977200508117676, 2.6131439208984375, 2.35127592086792,
                      2.8911080360412598, 4.647523880004883, 3.6192502975463867,
                      3.2930569648742676, 2.682900905609131, 2.521437883377075,
                      2.5245890617370605, 2.8213257789611816, 2.3920748233795166,
                      2.783450126647949, 2.7671618461608887, 2.963916778564453,
                      2.7982490062713623, 2.624756097793579, 2.51658296585083,
                      2.628687858581543, 2.7206978797912598, 2.443147897720337,
                      2.510477066040039, 2.5391528606414795, 2.6138951778411865,
                      2.2900848388671875, 2.4985432624816895, 2.2644991874694824,
                      2.6707191467285156, 2.3842780590057373, 2.4618968963623047,
                      2.5102832317352295, 2.59313702583313, 2.4741439819335938,
                      2.7459299564361572, 2.505012035369873, 2.54681396484375,
                      2.523116111755371, 2.4606881141662598, 2.458515167236328,
                      2.2674291133880615, 2.2277488708496094, 2.5287108421325684,
                      2.632244110107422, 2.7122678756713867, 2.346721887588501,
                      2.6661500930786133, 2.669651985168457, 2.3659818172454834,
                      2.4851808547973633, 2.595679998397827, 2.840704917907715,
                      2.660524845123291, 2.5195748805999756, 2.6051478385925293,
                      2.5070583820343018, 2.592315196990967, 2.4388649463653564,
                      2.6791939735412598, 2.657755136489868, 2.3712289333343506,
                      2.6789350509643555, 2.662574052810669, 2.5009491443634033,
                      2.657947301864624, 2.4550538063049316, 2.8192880153656006,
                      2.7697439193725586, 2.2997140884399414, 2.836836099624634,
                      2.5080511569976807]
    percentages = [0.95, 0.95, 0.96, 0.93, 0.92, 0.93, 0.93, 0.93]
    d23_percentages = [0.92, 0.97, 0.98, 0.96, 0.97, 0.96, 0.95, 0.93]
    d23_no_mean_percentages = [0.93, 0.96, 0.95, 0.96, 0.95, 0.95, 0.94, 0.94]
    no_edge_prctg = [0.93, 0.95, 0.95, 0.95, 0.95, 0.95, 0.96, 0.95]
    d23_no_edge_prctg = [0.93, 0.95, 0.95, 0.95, 0.96, 0.96, 0.97, 0.96]
    no_mean_no_edge_prctg =  [0.93, 0.96, 0.95, 0.96, 0.95, 0.95, 0.94, 0.94]



    '''Used for labeling the plots'''
    k_values = [1, 3, 5, 11, 15, 21, 51, 101]
    value_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                  12, 13, 14, 15, 16, 17, 18, 19, 20]
    k_values_zero_to_nine = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    box_plot_secs = [1, 2, 3, 4, 5]
    bar_plot_y_axis = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    x_axis = np.arange(len(k_values))
    fig, axs = plt.subplots(3, 3)


    '''Undo comments for performance testing'''
    '''
    py_sort_mean, heap_sort_mean, heapify_sort_mean =\
        performance_tests.sort_time_means(edge_testing_set, edge_training_set)              
    time_list, time_list_heap = performance_tests.calculate_times(edge_testing_set, edge_training_set)
    percentages, d_23_percentages, d23_nm_percentages, \
    no_edge_prctg, d23_no_edge_prctg, no_mean_no_edge_prctg = \
        performance_tests.calculate_all_accuracies(edge_testing_set, edge_training_set,
                                                   testing_images, training_images,
                                                   selected_test_labels)
    '''

    '''Prints to get updated data'''
    '''
    print("pysort mean", py_sort_mean)
    print("heap  sort mean", heap_sort_mean)
    print("Heapify sort mean", heapify_sort_mean)
    print("Time list", time_list)
    print("Time  list heap", time_list_heap)
    print("percentages", percentages)
    print("d23 percentages", d_23_percentages)
    print("d23 nm percentages", d23_nm_percentages)
    print("no edge percentage", no_edge_prctg)
    print("d23 no edge", d23_no_edge_prctg)
    print("no mean no edge", no_mean_no_edge_prctg)
    '''

    '''Runs the UI using matplotlib graphs'''
    for i in range(40):
        rand_val = random.randint(0, 9785)
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
        axs[1, 2].cla()
        axs[2, 2].cla()
        axs[1, 2].imshow(testing_images[rand_val])
        distance_list = calculate_distances_for_set \
            (testing_images[rand_val], training_images)
        sorted_list, indexes = \
            mhd.k_nearest_with_heap_search(20, distance_list)
        labels, label, label_amount = get_labels(indexes)
        axs[2, 2].bar(label_amount.keys(), label_amount.values())
        axs[2, 2].set_xticks(k_values_zero_to_nine)
        axs[2, 2].set_yticks(value_list)
        axs[1, 2].set_title("frame {}".format(selected_test_labels[rand_val]))
        axs[2, 2].set_title("frame {}".format("vote"))
        axs[2, 0].set_visible(False)
        axs[2, 1].set_visible(False)
        plt.pause(0.8)

if __name__ == "__main__":
    main()
