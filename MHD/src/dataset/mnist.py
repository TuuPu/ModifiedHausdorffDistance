from tensorflow.keras.datasets import mnist
import numpy as np

# About keras mnist dataset:
# A dataset of 60 000 greyscale training images 28x28 pixels each. 10 000 images used for testing.

# data split to training and testing

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# By using label values, select 10k training images and make a list where train_pictures[0] holds 1000 images
# with class 0, train_pictures[1] holds 1k images of class 1 and so on. Classes represent numbers from 0 to 9.
# Also convert greyscale images from mnist to binary images using a threshold of 100, so from the original
# greyscale images, every pixel that has a value over 100 is turned to 1 (black) and every pixel that has a value
# under 100 is turned to 0 (white).

def sort_images_and_threshold():
    train_pictures = []
    for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
        idxs = np.argwhere(y_train == i)
        idxs = idxs[:1000]
        tmp_figures = x_train[idxs, :, :]
        tmp_figures = np.where(tmp_figures <= 100, 0, 1)
        train_pictures.append(tmp_figures)
    return train_pictures

# Prints a binary image of the number 0. If we want to test this for number 8 for example, we just type
# train_pictures[8][n], where n is a value between 0 and 999.
def print_training_image():
    train_pictures = sort_images_and_threshold()
    k, n = input("Enter a number you want to view (0 to 9  range) and index number between 0 and 999: ").split()
    print('A picture of value ', k, ' printed below in binary form' + "\n" + str(train_pictures[int(k)][int(n)]))

# Additional comments:
# Apparently keras is a bit of a heavy library and even without a simple for loop, it takes a few seconds to run this
# module. I noticed it from the beginning when I tried to just print the shapes of the datasets before writing the for
# loop. So the for loop isn't the time consuming part here, it is keras.

#Time used tuesday 10.00 - 17.00 with an hour long lunch break. Mainly used my time to read about keras and how to use
#mnist database (loading the data and how to actually handle it). Another big time consumer was learning numpy. I haven't
#used it before. So learning how to handle numpy arrays and such took a lot of time.