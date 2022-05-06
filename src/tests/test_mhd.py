import unittest
from tensorflow.keras.datasets import mnist
from scipy.spatial import cKDTree
import numpy as np
from dataset import image_processing
from distance import mhd
import heapq
import math


def create_matrices():
    '''Creates two random binary matrices of size 28x28'''
    arr1 = np.random.rand(28, 28) > 0.5
    arr2 = np.random.rand(28, 28) > 0.5
    img_1 = arr1.reshape(28, 28)
    img_2 = arr2.reshape(28, 28)
    img1_transp = np.array(np.where(img_1))
    img2_transp = np.array(np.where(img_2))
    return img_1, img_2, img1_transp, img2_transp

def distances(img1, img2):
    '''Calculates distances for two matrices in a manual way
    e.g not using a kd-tree or other pre-built tool for distance calculations
    '''
    dist_squared=[]
    for coord in zip(img1[0,:], img1[1,:]):
        dist_squared.append((np.sqrt((img2[0,:]-coord[0])**2+(img2[1,:]-coord[1])**2)).min())
    return np.array(dist_squared)

class TestMhd(unittest.TestCase):

    def setUp(self):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        self.training_images, selected_training_labels = image_processing.sort_images_and_threshold(x_train, y_train)
        self.test_images, selected_testing_labels = image_processing.sort_images_and_threshold(x_test, y_test)
        self.edge_training = image_processing.create_binary_edge_image(self.training_images)
        self.edge_testing = image_processing.create_binary_edge_image(self.test_images)

    def test_min_distance_pairwise(self):
        '''
        Calculates distances manually
        and compares that the used function in the
        program returns the same value.

        This is for pairwise distance for two matrices
        '''
        img1, img2, img1_transp, img2_transp = create_matrices()
        dist_squared = distances(img1_transp, img2_transp)
        dist_squared2 = distances(img2_transp, img1_transp)
        dist_squared = np.array(dist_squared)
        dist_squared2 = np.array(dist_squared2)

        d1, d2 = mhd.calculate_minimum_distance_pairwise(img1, img2)
        np.testing.assert_allclose(dist_squared2, d1)
        np.testing.assert_allclose(dist_squared, d2)




    def test_mhd22(self):
        '''
        Calculates distances manually
        and compares that the used function in the
        program returns the same value.

        This is for a set of matrices.
        '''
        img1, img2, img1_transp, img2_transp = create_matrices()
        dist_squared = distances(img1_transp, img2_transp)
        dist_squared2 = distances(img2_transp, img1_transp)
        dist_squared = np.array(dist_squared)
        dist_squared2 = np.array(dist_squared2)
        distance_1 = dist_squared.mean()
        distance_2 = dist_squared2.mean()
        max_length = max(distance_1, distance_2)


        d1 = mhd.mhd_d22(img1, img2)
        self.assertEqual(d1, max_length)

    def test_mhd23(self):
        '''
        Does the same thing as test_mhd22
        but uses mhd23 as a distance measure
        '''
        img1, img2, img1_transp, img2_transp = create_matrices()
        dist_squared = distances(img1_transp, img2_transp)
        dist_squared2 = distances(img2_transp, img1_transp)
        dist_squared = np.array(dist_squared)
        dist_squared2 = np.array(dist_squared2)
        distance_1 = dist_squared.mean()
        distance_2 = dist_squared2.mean()
        function_3 = (distance_1+distance_2)/2

        d1 = mhd.mhd_d23(img1, img2)
        self.assertEqual(d1, function_3)

    def test_mhd23_without_mean(self):
        '''
        Does the same thing as test_mhd22
        but uses mhd23 without mean
        as a distance measure
        '''
        img1, img2, img1_transp, img2_transp = create_matrices()
        dist_squared = distances(img1_transp, img2_transp)
        dist_squared2 = distances(img2_transp, img1_transp)
        dist_squared = np.array(dist_squared)
        dist_squared2 = np.array(dist_squared2)
        distance_1 = dist_squared.sum()
        distance_2 = dist_squared2.sum()
        function_3 = (distance_1 + distance_2)/2
        d1 = mhd.mhd_d23_without_mean(img1, img2)
        self.assertEqual(d1, function_3)

    def test_k_nearest(self):
        '''
        Checks that values returned
        are smallest. Uses three smallest values
        '''
        distance_list = []
        for i, image in enumerate(self.edge_training):
            distance_list.append([mhd.mhd_d22(self.edge_testing[0], image), i])
        three_distances, indexes = mhd.k_nearest(3, distance_list)
        self.assertEqual(three_distances, [[0.5774284771915368, 218], [0.5774284771915368, 625], [0.5898000448436316, 951]])

    def test_k_nearest_with_heap_search(self):
        '''
        Uses a premade list of distances
        and indexes and checks that sorting
        the list in ascending order works
        '''
        test_list = [[3, 2], [4, 7], [5, 10], [2, 8], [1, 9]]
        sorted_list, indexes = mhd.k_nearest_with_heap_search(5, test_list)
        self.assertEqual(sorted_list, [[1, 9], [2, 8], [3, 2], [4, 7], [5, 10]])

    def test_heapify_sort(self):
        '''
        Uses a premade list of values
        and tests that it is transformed
        as a heap structure and picks the
        three smallest values from that list.
        '''
        number_list = [3, 5, 1, 4, 6, 7]
        heapq.heapify(number_list)
        sorted_distances = heapq.nsmallest(3, number_list)
        self.assertEqual(mhd.k_nearest_with_heapify(3, number_list), sorted_distances)