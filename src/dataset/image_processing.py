import numpy as np
import scipy.ndimage.morphology as mrph

# About keras mnist dataset:
# A dataset of 60 000 greyscale training images 28x28 pixels each.
# 10 000 images used for testing.

# data split to training and testing

#(x_train, y_train), (x_test, y_test) = mnist.load_data()

# By using label values, select 10k training images
# and make a list where train_pictures[0] holds 1000 images
# with class 0, train_pictures[1] holds 1k images of class 1 and so on.
# Classes represent numbers from 0 to 9.
# Also convert greyscale images from mnist to binary images
# using a threshold of 100, so from the original
# greyscale images, every pixel that has a
# value over 100 is turned to 1 (black) and every pixel that has a value
# under 100 is turned to 0 (white).

def sort_images_and_threshold(pictures, labels, test):
    thresholded_images = []
    for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
        idxs = np.argwhere(labels == i)
        if test:
            idxs = idxs[:1000]
        else:
            idxs = idxs[:10000]
        tmp_figures = pictures[idxs, :, :]
        tmp_figures = np.where(tmp_figures <= 100, 0, 1)
        thresholded_images.append(tmp_figures)
    return thresholded_images

#https://en.wikipedia.org/wiki/Erosion_(morphology)
#Creating edge images to help with the distance calculations.
def create_binary_edge_image(image_set, s_e=None):
    edge_images = []
    image_set = np.concatenate(image_set, axis=0)
    image_set = np.squeeze(image_set)
    for i in range(image_set.shape[0]):
        img = image_set[i,:,:]
        if s_e is None:
            s_e = mrph.generate_binary_structure(2, 1)

        # Sets pictures with elements as booleans according to 0 and 1
        s_erosion = img.astype(np.bool) ^ mrph.binary_erosion(img, s_e)
        # appends pictures to list and flips boolean types to ints.
        edge_images.append(s_erosion.astype(int))
    return np.array(edge_images)

#Returns a 2D array in shape of (75, 2) for example.
def coordinates(image):
    image_coordinates = np.array(np.where(image)).T
    return image_coordinates



# Prints a binary image of the number 0.
# If we want to test this for number 8 for example, we just type
# train_pictures[8][n], where n is a value between 0 and 999.
# NOTE: test_pictures is here only to show that the function works
# for testing set too. It is not meant to be labeled
# to the use of getting the answer. Testing will be done
# in a correct way, using k-nearest and MHD, as discussed.
def print_training_image(image_set):
    print(image_set.shape)
    print(image_set[0])
    print(coordinates(image_set[0]).shape)

# Additional comments:
# Apparently keras is a bit of a heavy library and even without
# a simple for loop, it takes a few seconds to run this
# module. I noticed it from the beginning when I tried to
# just print the shapes of the datasets before writing the for
# loop. So the for loop isn't the time consuming part here, it is keras.
