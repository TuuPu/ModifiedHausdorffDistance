import unittest
from tensorflow.keras.datasets import mnist
from scipy.spatial import cKDTree
from dataset import image_processing
from distance import mhd

class TestMhd(unittest.TestCase):

    def setUp(self):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        self.training_images = image_processing.sort_images_and_threshold(x_train, y_train, True)
        self.test_images = image_processing.sort_images_and_threshold(x_test, y_test, False)
        self.edge_training = image_processing.create_binary_edge_image(self.training_images)
        self.edge_testing = image_processing.create_binary_edge_image(self.test_images)

    def test_min_distance_pairwise(self):
        coordinates_test_img = image_processing.coordinates(self.edge_testing[0])
        coordinates_training_img = image_processing.coordinates(self.edge_training[0])
        tree1 = cKDTree(coordinates_test_img)
        tree2 = cKDTree(coordinates_training_img)
        distance_1 = tree1.query(coordinates_training_img)[0]
        distance_2 = tree2.query(coordinates_test_img)[0]

        d1, d2 = mhd.calculate_minimum_distance_pairwise(self.edge_testing[0], self.edge_training[0])
        self.assertTrue((d1 == distance_1).all())
        self.assertTrue((d2 == distance_2).all())

    def test_mhd22(self):
        coordinates_test_img = image_processing.coordinates(self.edge_testing[0])
        coordinates_training_img = image_processing.coordinates(self.edge_training[0])
        tree1 = cKDTree(coordinates_test_img)
        tree2 = cKDTree(coordinates_training_img)
        distance_1 = tree1.query(coordinates_training_img)[0]
        distance_2 = tree2.query(coordinates_test_img)[0]
        distance_1 = distance_1.mean()
        distance_2 = distance_2.mean()
        max_length = max(distance_1, distance_2)


        d1 = mhd.mhd_d22(self.edge_testing[0], self.edge_training[0])
        self.assertEqual(d1, max_length)