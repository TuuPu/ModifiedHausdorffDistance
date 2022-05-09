#pylint: disable-all
import random
from tensorflow.keras.datasets import mnist # pylint: disable=E0611, E0401
import numpy as np
from dataset import image_processing
from distance import mhd
from performance import performance_tests
import matplotlib.pyplot as plt
# NOTE: Importing the mnist database takes about 7-10 seconds

(x_train, y_train), (x_test, y_test) = mnist.load_data()


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
    Test values below have been gathered on 08.05.2022
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

    py_sort_mean = [0.0047638, 0.00429706, 0.00432985, 0.00424477, 0.0040553, 0.00409049,
                    0.00406271, 0.00419157]
    heap_sort_mean = [0.00068033, 0.00079879, 0.00076212, 0.0008914, 0.00077605, 0.00079536,
                    0.00107182, 0.00137821]
    heapify_sort_mean = [0.00231094, 0.0021143, 0.0023489, 0.00228722, 0.00207502, 0.00206829,
                        0.00219555, 0.00209914]
    time_list = [2.312958240509033, 2.4681777954101562, 2.7978360652923584,
                 2.498765707015991, 2.360558032989502, 2.364879846572876,
                 2.3625471591949463, 2.5253639221191406, 2.511155843734741,
                 2.3693931102752686, 2.413285255432129, 2.2289280891418457,
                 2.373210906982422, 2.397304058074951, 2.2597200870513916,
                 2.232243061065674, 2.409240245819092, 2.6261839866638184,
                 2.403010129928589, 2.4126219749450684, 3.0074968338012695,
                 3.249073028564453, 2.411289930343628, 2.7969820499420166,
                 3.4074289798736572, 3.697234869003296, 2.5649030208587646,
                 2.558454990386963, 2.4174201488494873, 2.400458812713623,
                 3.4959800243377686, 2.3746869564056396, 2.688620090484619,
                 3.5233848094940186, 2.4654829502105713, 2.3181140422821045,
                 2.4567019939422607, 2.561408042907715, 2.5790700912475586,
                 2.5377891063690186, 2.3457741737365723, 2.5786819458007812,
                 2.111706018447876, 2.2093732357025146, 2.3318490982055664,
                 2.122241973876953, 2.381155252456665, 2.3951289653778076,
                 2.2671711444854736, 2.515148878097534, 2.515289068222046,
                 2.461945056915283, 2.2062079906463623, 2.44447922706604,
                 2.2794442176818848, 2.2217178344726562, 2.650120258331299,
                 2.2944629192352295, 2.4328010082244873, 2.1125619411468506,
                 2.30633807182312, 2.4953248500823975, 2.3751816749572754,
                 2.5973780155181885, 2.4950101375579834, 2.0896718502044678,
                 2.0827791690826416, 2.537163019180298, 2.499885082244873,
                 2.422661304473877, 2.4970717430114746, 2.3823091983795166,
                 2.4465110301971436, 2.565936803817749, 2.5609259605407715,
                 2.3110671043395996, 2.090177059173584, 2.39119815826416,
                 2.4247119426727295, 2.222524881362915, 2.4394869804382324,
                 2.4156148433685303, 2.5908820629119873, 2.3422162532806396,
                 2.3354861736297607, 2.422823905944824, 2.4294848442077637,
                 2.6801979541778564, 2.4876949787139893, 2.271387815475464,
                 2.1120667457580566, 2.5890049934387207, 2.2006468772888184,
                 2.501893997192383, 2.4111008644104004, 2.6232969760894775,
                 2.4750149250030518, 2.5883638858795166, 2.505402088165283, 2.566509962081909]
    time_list_heap = [2.461364984512329, 2.631883144378662, 2.7667229175567627,
                      2.5524520874023438, 2.4670143127441406, 2.4820101261138916,
                      2.477707862854004, 2.4365339279174805, 2.655881404876709,
                      2.465930938720703, 2.484636068344116, 2.3329620361328125,
                      2.579860210418701, 2.5162248611450195, 2.361788034439087,
                      2.3382251262664795, 2.559821605682373, 2.415750026702881,
                      2.4840309619903564, 2.541926860809326, 3.3592729568481445,
                      2.934512138366699, 2.457395076751709, 3.003113269805908,
                      4.025400161743164, 3.5027971267700195, 2.559830904006958,
                      2.5703959465026855, 2.5722098350524902, 3.003406047821045,
                      2.7909069061279297, 2.5718278884887695, 3.3783211708068848,
                      2.9405570030212402, 2.489932060241699, 2.4360299110412598,
                      2.5476021766662598, 2.6522350311279297, 2.637115955352783,
                      2.5164847373962402, 2.4413459300994873, 2.6818089485168457,
                      2.2816050052642822, 2.3524792194366455, 2.443025827407837,
                      2.35798716545105, 2.4305837154388428, 2.5519192218780518,
                      2.3438820838928223, 2.648771047592163, 2.522117853164673,
                      2.4952218532562256, 2.3414478302001953, 2.5719287395477295,
                      2.405059814453125, 2.337163209915161, 2.6964118480682373,
                      2.402029037475586, 2.5235559940338135, 2.198068857192993,
                      2.4232659339904785, 2.5420353412628174, 2.5142951011657715,
                      3.1748740673065186, 2.586674928665161, 2.2323200702667236,
                      2.1934189796447754, 2.6197900772094727, 2.5907857418060303,
                      2.5329959392547607, 2.586167097091675, 2.452528953552246,
                      2.521436929702759, 2.644892930984497, 2.638123035430908,
                      2.363722801208496, 2.3598170280456543, 2.57039737701416,
                      2.5452167987823486, 2.3365609645843506, 2.5533320903778076,
                      2.4998860359191895, 2.6356828212738037, 2.3511698246002197,
                      2.420888662338257, 2.5257439613342285, 2.604912281036377,
                      2.727800130844116, 2.5787956714630127, 2.397455930709839,
                      2.2528390884399414, 2.7054409980773926, 2.3344738483428955,
                      2.5927200317382812, 2.5504510402679443, 2.6325430870056152,
                      2.5690231323242188, 2.675368070602417, 2.584561824798584, 2.6070919036865234]
    percentages = [0.95, 0.96, 0.95, 0.95, 0.93, 0.92, 0.9,  0.88]
    d23_percentages = [0.96, 0.93, 0.95, 0.94, 0.94, 0.93, 0.9,  0.87]
    d23_no_mean_percentages = [0.97, 0.94, 0.94, 0.94, 0.93, 0.94, 0.9,  0.85]
    no_edge_prctg = [0.96, 0.98, 0.95, 0.98, 0.96, 0.95, 0.94, 0.9]
    d23_no_edge_prctg = [0.98, 0.98, 0.98, 0.97, 0.97, 0.97, 0.92, 0.9]
    no_mean_no_edge_prctg = [0.97, 0.94, 0.94, 0.94, 0.93, 0.94, 0.9,  0.85]

    '''Used for labeling the plots'''
    k_values = [1, 3, 5, 11, 15, 21, 51, 101]
    k_values_zero_to_nine = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    box_plot_secs = [1, 2, 3, 4, 5]
    bar_plot_y_axis = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    x_axis = np.arange(len(k_values))

    '''Runs the UI using matplotlib graphs
    Choosing 1 shows general statistics from gathered data
    Choosing 2 shows one random image and the result of the k-nearest vote
    Choosing 3 shows 20 random images and results of the vote
    NOTE Both 2 and 3 use a k-value of user's choice
    Choosing 4 shows and ASCII version of a random image (this was just for fun)
    Choosing 5 runs the performance tests and gives updated data so general statistics (1)
    can be performed with fresh data.
    Choosing q quits the program.
    '''
    running = True
    while (running):
        print("What do you want to see?")
        print("1: general statistics")
        print("2: Test a random number")
        print("3: Run a loop of 20 random numbers")
        print("4: ASCII gorgeousness")
        print("5: Run performance tests (takes about an hour)")
        print("q: Quit program")
        operation = input("Select one: ")

        if operation == '1':
            fig, axs = plt.subplots(2, 3)
            axs[0, 0].plot(k_values, py_sort_mean, color='blue')
            axs[0, 0].plot(k_values, heap_sort_mean, color='orange')
            axs[0, 0].plot(k_values, heapify_sort_mean, color='green')
            axs[0, 0].set_xlabel("k-values")
            axs[0, 0].set_ylabel("Seconds")
            axs[0, 0].set_title("Sort times")
            axs[0, 0].legend(['sorted()', 'heapq', 'heapify'])
            axs[0, 0].set_title("Sort times")
            axs[0, 1].boxplot(time_list)
            axs[0, 1].set_title("Distance calculation times")
            axs[0, 1].set_ylabel("Seconds")
            axs[0, 1].set_yticks(box_plot_secs)
            axs[0, 2].boxplot(time_list_heap)
            axs[0, 2].set_title("Distance calculations with heap, k=5")
            axs[0, 2].set_ylabel("Seconds")
            axs[0, 2].set_yticks(box_plot_secs)
            axs[1, 0].bar \
                (x_axis + 0.20, percentages, width=0.2, color='blue')
            axs[1, 0].bar \
                (x_axis + 0.20 * 2, d23_percentages, width=0.2, color='orange')
            axs[1, 0].bar \
                (x_axis + 0.20 * 3, d23_no_mean_percentages, width=0.2, color='green')
            axs[1, 0].set_xticks(x_axis, k_values)
            axs[1, 0].set_xlabel("k-values")
            axs[1, 0].set_ylabel("percentage")
            axs[1, 0].set_yticks(bar_plot_y_axis)
            axs[1, 0].legend(['D22', 'D23', 'D23 no mean'], loc='lower right')
            axs[1, 0].set_title("Accuracy with edge images")
            axs[1, 1].bar \
                (x_axis - 0.20, no_edge_prctg, width=0.2, color='blue')
            axs[1, 1].bar \
                (x_axis, d23_no_edge_prctg, width=0.2, color='orange')
            axs[1, 1].bar \
                (x_axis + 0.20, no_mean_no_edge_prctg, width=0.2, color='green')
            axs[1, 1].set_xticks(x_axis, k_values)
            axs[1, 1].set_xlabel("k-values")
            axs[1, 1].set_ylabel("percentages")
            axs[1, 1].set_yticks(bar_plot_y_axis)
            axs[1, 1].legend(['D22', 'D23', 'D23 no mean'], loc='lower right')
            axs[1, 1].set_title("Accuracy with binary images")
            axs[1, 2].set_axis_off()
            axs[1, 2].annotate(' k-values represent the amount of '
                                'closest neighbours chosen. \n For example k=5 picks five'
                                ' of the closest \n distances to vote from. Voting means'
                                ' that if k=5 \n and those five shortest distances \n'
                               ' represent numbers'
                                ' 3, 5, 3, 3, 5. \n Then 3 is chosen as an output of the'
                                ' program.', (0.1, 0.5), xycoords='axes fraction', va='center')
            plt.show()

        elif operation == '2':
            while True:
                try:
                    k = int(input("Choose a k value (preferably an odd number):"))
                    k_list = range(int(k)+1)
                except ValueError:
                    print("Please enter a valid integer")
                    continue
                break
            rand_val = random.randint(0, 9785)
            fig, axs = plt.subplots(2)
            axs[0].imshow(testing_images[rand_val])
            distance_list = performance_tests.calculate_distances_for_set \
                (testing_images[rand_val], training_images)
            sorted_list, indexes = \
                mhd.k_nearest_with_heap_search(int(k), distance_list)
            labels, label, label_amount = performance_tests.get_labels(indexes)
            axs[1].bar(label_amount.keys(), label_amount.values())
            axs[1].set_xticks(k_values_zero_to_nine)
            axs[1].set_yticks(k_list)
            axs[0].set_title("number {}".format(selected_test_labels[rand_val]))
            axs[1].set_title("test number {}".format("vote"))
            fig.tight_layout()
            plt.show()

        elif operation == '3':
            fig, axs = plt.subplots(2)
            while True:
                try:
                    k = int(input("Choose a k value (preferably an odd number):"))
                    k_list = range(int(k)+1)
                except ValueError:
                    print("Please enter a valid integer")
                    continue
                break
            for i in range(20):
                axs[0].cla()
                axs[1].cla()
                rand_val = random.randint(0, 9785)
                axs[0].imshow(testing_images[rand_val])
                distance_list = performance_tests.calculate_distances_for_set \
                    (testing_images[rand_val], training_images)
                sorted_list, indexes = \
                    mhd.k_nearest_with_heap_search(int(k), distance_list)
                labels, label, label_amount = performance_tests.get_labels(indexes)
                axs[1].bar(label_amount.keys(), label_amount.values())
                axs[1].set_xticks(k_values_zero_to_nine)
                axs[1].set_yticks(k_list)
                axs[0].set_title("number {}".format(selected_test_labels[rand_val]))
                axs[1].set_title("test number {}".format("vote"))
                fig.tight_layout()
                plt.pause(0.8)

        elif operation == '4':
            rand_val = random.randint(0, 9785)
            print(testing_images[rand_val])

        elif operation == '5':
            py_sort_mean, heap_sort_mean, heapify_sort_mean = \
                performance_tests.sort_time_means(edge_testing_set, edge_training_set)
            time_list, time_list_heap = performance_tests.calculate_times(edge_testing_set, edge_training_set)
            percentages, d_23_percentages, d23_nm_percentages, \
            no_edge_prctg, d23_no_edge_prctg, no_mean_no_edge_prctg = \
                performance_tests.calculate_all_accuracies(edge_testing_set, edge_training_set,
                                                           testing_images, training_images,
                                                           selected_test_labels)
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

        elif operation == 'q':
            running = False
        else:
            print("Not cool, don't try to break this")



if __name__ == "__main__":
    main()
