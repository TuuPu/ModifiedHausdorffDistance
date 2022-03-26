from tensorflow.keras.datasets import mnist # pylint: disable=E0611, E0401
import numpy as np

# About keras mnist dataset:
# A dataset of 60 000 greyscale training images 28x28 pixels each.
# 10 000 images used for testing.

# data split to training and testing

(x_train, y_train), (x_test, y_test) = mnist.load_data()

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



# Prints a binary image of the number 0.
# If we want to test this for number 8 for example, we just type
# train_pictures[8][n], where n is a value between 0 and 999.
# NOTE: test_pictures is here only to show that the function works
# for testing set too. It is not meant to be labeled
# to the use of getting the answer. Testing will be done
# in a correct way, using k-nearest and MHD, as discussed.
def print_training_image():
    train_pictures = sort_images_and_threshold(x_train, y_train, True)
    test_pictures = sort_images_and_threshold(x_test, y_test, False)
    image, index = input("Enter a number you want to view (0 to 9  range) "
                         "and index number between 0 and 999: ").split()
    print('A picture of value ', image,
          ' printed below in binary form' + "\n" + str(train_pictures[int(image)][int(index)]))
    print(test_pictures[0][0])

# Additional comments:
# Apparently keras is a bit of a heavy library and even without
# a simple for loop, it takes a few seconds to run this
# module. I noticed it from the beginning when I tried to
# just print the shapes of the datasets before writing the for
# loop. So the for loop isn't the time consuming part here, it is keras.
