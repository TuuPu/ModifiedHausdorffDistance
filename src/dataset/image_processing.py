import numpy as np
import scipy.ndimage.morphology as mrph

# data split to training and testing

#(x_train, y_train), (x_test, y_test) = mnist.load_data()


def sort_images_and_threshold(pictures, labels, test):
    '''
    Selects 10k  images and sorts them to a list
    using labels. If you call thresholded_images[0], you get
    1k images of the number zero in binary form.
    So it trims down the training set to 10k pictures and
    manipulates them from greyscale to binary form
    using thresholding. Threshold value is set to 100,
    so that every pixel that is below the value 100
    is 0 (white) and every pixel over the value 100
    is 1 (black).
    '''
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

def create_binary_edge_image(image_set, s_e=None):
    '''
    Creates edge images to help with distance calculations.
    Edge images consits of only the outer lines of processed image.
    More info about erosion in morphology can be found at
    https://en.wikipedia.org/wiki/Erosion_(morphology)
    Also trims down the datasets to lesser dimensions
    for easier manipulation of images.
    Returns a binary image with only outer lines left.
    '''
    edge_images = []
    image_set = np.concatenate(image_set, axis=0)
    image_set = np.squeeze(image_set)
    for i in range(image_set.shape[0]):
        img = image_set[i,:,:]
        if s_e is None:
            s_e = mrph.generate_binary_structure(2, 1)

        # Sets pictures with elements as booleans according to 0 and 1
        s_erosion = img.astype(bool) ^ mrph.binary_erosion(img, s_e)
        # appends pictures to list and flips boolean types to ints.
        edge_images.append(s_erosion.astype(int))
    return np.array(edge_images)


def coordinates(image):
    '''
    Takes in an image and returns an array
    of shape (75, 2) for example. The array consists
    of the coordinates of the values 1 in the picture
    '''
    image_coordinates = np.array(np.where(image)).T
    return image_coordinates



def print_training_image(image_set):
    '''
    This function is purely here for testing that everything works
    as supposed to. It will be removed in the end, once the project
    is finished.
    '''
    print(image_set.shape)
    print(image_set[0])
    print(coordinates(image_set[0]).shape)
