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
        d1, d2 = mhd.calculate_minimum_distance_pairwise(self.edge_testing[0], self.edge_training[0])
        self.assertEqual(d1.shape, (86,))
        self.assertEqual(d2.shape, (75,))

    def test_mhd22(self):
        d1 = mhd.mhd_d22(self.edge_testing[0], self.edge_training[0])
        self.assertEqual(d1, 0.9252521966042869)