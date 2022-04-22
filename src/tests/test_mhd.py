import unittest
from tensorflow.keras.datasets import mnist
from scipy.spatial import cKDTree
import numpy as np
from dataset import image_processing
from distance import mhd
import heapq
import math

class TestMhd(unittest.TestCase):

    def setUp(self):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        self.training_images, selected_training_labels = image_processing.sort_images_and_threshold(x_train, y_train)
        self.test_images, selected_testing_labels = image_processing.sort_images_and_threshold(x_test, y_test)
        self.edge_training = image_processing.create_binary_edge_image(self.training_images)
        self.edge_testing = image_processing.create_binary_edge_image(self.test_images)

    def test_min_distance_pairwise(self):
        img_1 = np.array([[0, 1, 0], [0, 0, 0]])
        img_2 = np.array([[0, 0, 0], [0, 1, 0]])
        img1_transp = np.array(np.where(img_1))
        img2_transp = np.array(np.where(img_2))
        dist_squared = np.sum(np.square(img1_transp-img2_transp))
        d1, d2 = mhd.calculate_minimum_distance_pairwise(img_1, img_2)
        self.assertTrue(d1 == dist_squared)
        self.assertTrue(d2 == dist_squared)



    def test_mhd22(self):
        img_1 = np.array([[0, 1, 0], [0, 0, 0]])
        img_2 = np.array([[0, 0, 0], [0, 1, 0]])
        img1_transp = np.array(np.where(img_1))
        img2_transp = np.array(np.where(img_2))
        dist_squared = np.sum(np.square(img1_transp-img2_transp))
        dist_squared2 = np.sum(np.square(img2_transp-img1_transp))
        distance_1 = dist_squared.mean()
        distance_2 = dist_squared2.mean()
        max_length = max(distance_1, distance_2)


        d1 = mhd.mhd_d22(img_1, img_2)
        self.assertEqual(d1, max_length)

    def test_mhd23(self):
        img_1 = np.array([[0, 1, 0], [0, 0, 0]])
        img_2 = np.array([[0, 0, 0], [0, 1, 0]])
        img1_transp = np.array(np.where(img_1))
        img2_transp = np.array(np.where(img_2))
        dist_squared = np.sum(np.square(img1_transp-img2_transp))
        dist_squared2 = np.sum(np.square(img2_transp-img1_transp))
        distance_1 = dist_squared.mean()
        distance_2 = dist_squared2.mean()
        function_3 = (distance_1+distance_2)/2

        d1 = mhd.mhd_d23(img_1, img_2)
        self.assertEqual(d1, function_3)

    def test_mhd23_without_mean(self):
        img_1 = np.array([[0, 1, 0], [0, 0, 0]])
        img_2 = np.array([[0, 0, 0], [0, 1, 0]])
        img1_transp = np.array(np.where(img_1))
        img2_transp = np.array(np.where(img_2))
        dist_squared = np.sum(np.square(img1_transp-img2_transp))
        dist_squared2 = np.sum(np.square(img2_transp-img1_transp))
        distance_1 = dist_squared.sum()
        distance_2 = dist_squared2.sum()
        function_3 = (distance_1 + distance_2)/2
        d1 = mhd.mhd_d23_without_mean(img_1, img_2)
        self.assertEqual(d1, function_3)

    def test_k_nearest(self):
        distance_list = []
        for i, image in enumerate(self.edge_training):
            distance_list.append([mhd.mhd_d22(self.edge_testing[0], image), i])
        three_distances, indexes = mhd.k_nearest(3, distance_list)
        self.assertEqual(three_distances, [[0.5774284771915368, 218], [0.5774284771915368, 625], [0.5898000448436316, 951]])

    def test_k_nearest_with_heap_search(self):
        test_list = [[3, 2], [4, 7], [5, 10], [2, 8], [1, 9]]
        sorted_list, indexes = mhd.k_nearest_with_heap_search(5, test_list)
        self.assertEqual(sorted_list, [[1, 9], [2, 8], [3, 2], [4, 7], [5, 10]])

    def test_heapify_sort(self):
        number_list = [3, 5, 1, 4, 6, 7]
        heapq.heapify(number_list)
        sorted_distances = heapq.nsmallest(3, number_list)
        self.assertEqual(mhd.k_nearest_with_heapify(3, number_list), sorted_distances)