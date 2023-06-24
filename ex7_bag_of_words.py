import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from PIL import Image
from lib.visualization.image import put_text


class BoW:
    def __init__(self, n_clusters, n_features):
        # Create the ORB detector
        self.extractor = cv2.ORB_create(nfeatures=n_features)
        self.n_clusters = n_clusters
        # Make a kmeans cluster
        self.kmeans = KMeans(self.n_clusters, verbose=0)
        self.histograms = []

    def train(self, imgs):
        """
        Make the bag of "words"

        Parameters
        ----------
        imgs (list): A list with training images. Shape (n_images)
        """
        descriptors = []
        for img in imgs:
            # Compute the descriptors for each training img
            kp = self.extractor.detect(img, None)
            kp, des = self.extractor.compute(img, kp)
            # Concatenate the list of lists of descriptors to create a list of descriptors
            descriptors.append(des)
            # Create a KMeans object with n clusters, and fit the list of descriptors
            # Use the hist function to create a histogram database by calculating the histogram for each img
            # (i.e. call the hist function on each list of descriptors in the list of lists)
            histogram = self.hist(des)
            # Hint -- save class variables like 'self.object = create_object()' in order to use them in other functions
            self.histograms.append(histogram)


    def hist(self, descriptors):
        """
        Make the histogram for words in the descriptors

        Parameters
        ----------
        descriptors (ndarray): The ORB descriptors. Shape (n_features, 32)

        Returns
        -------
        hist (ndarray): The histogram. Shape (n_clusters)
        """
        # Input - list of descriptors for a single image
        # Use the fitted kmeans model to label the descriptors
        kmeans = self.kmeans.fit(descriptors)
        centroids = kmeans.cluster_centers_
        # Make a histogram of the labels using np.histogram. Remember to specify the amount of bins (n_clusters)
        # and the range (0, n_clusters - 1)
        counts, bins = np.histogram(centroids, bins=self.n_clusters, range=(0, self.n_clusters))
        # return the histogram
        return counts, bins

    def predict(self, img):
        """
        Finds the closest match in the training set to the given image

        Parameters
        ----------
        img (ndarray): The query image. Shape (height, width [,3])

        Returns
        -------
        match_idx (int): The index of the training image there was closest, -1 if there was non descriptors in the image
        """
        # Compute descriptors for the image
        kp = self.extractor.detect(img, None)
        kp, des = self.extractor.compute(img, kp)
        # Calculate the histogram using hist()
        counts, bins = self.hist(des)
        # Calculate the distance to each histogram in the database
        smallest_dist = float('inf')
        for h in self.histograms:
            # Euclidean distance
            dist = np.linalg.norm(h[0] - counts)
            if dist < smallest_dist:
                smallest_dist = dist

        # Return the index of the histogram in the database with the smallest distance
        return smallest_dist


def split_data(dataset, test_size=0.1):
    """
    Loads the images and split it into a train and test set

    Parameters
    ----------
    dataset (str): The path to the dataset
    test_size (float): Represent the proportion of the dataset to include in the test split

    Returns
    -------
    train_img (list): The images in the training set. Shape (n_images)
    test_img (list): The images in the test set. Shape (n_images)
    """
    # Load the images and split it into a train and test set using train_test_split from sklearn
    images = []
    for file in os.listdir(dataset):
        img = Image.open(dataset + file)
        images.append(np.array(img))

    train_imgs, test_imgs = train_test_split(images, test_size=test_size, random_state=42)
    return train_imgs, test_imgs


def make_stackimage(query_image, match_image=None):
    """
    hstack the query and match image

    Parameters
    ----------
    query_image (ndarray): The query image. Shape (height, width [,3])
    match_image (ndarray): The match image. Shape (height, width [,3])

    Returns
    -------
    stack_image (ndarray): The stack image. Shape (height, 2*width [,3])
    """
    match_found = True
    if match_image is None:
        match_image = np.zeros_like(query_image)
        match_found = False

    if len(query_image.shape) != len(match_image.shape):
        if len(query_image.shape) != 3:
            query_image = cv2.cvtColor(cv2.COLOR_GRAY2BGR, query_image)
        if len(match_image.shape) != 3:
            match_image = cv2.cvtColor(cv2.COLOR_GRAY2BGR, match_image)

    height1, width1, *_ = query_image.shape
    height2, width2, *_ = match_image.shape
    height = max([height1, height2])
    width = max([width1, width2])

    if len(query_image.shape) == 2:
        stack_shape = (height, width * 2)
    else:
        stack_shape = (height, width * 2, 3)

    if match_found:
        put_text(query_image, "top_center", "Query")
        put_text(match_image, "top_center", "Match")

    stack_image = np.zeros(stack_shape, dtype=match_image.dtype)
    stack_image[0:height1, 0:width1] = query_image
    stack_image[0:height2, width:width + width2] = match_image

    if not match_found:
        put_text(stack_image, "top_center", "No features found")

    return stack_image


if __name__ == "__main__":
    dataset = '../data/COIL20/images/'  # '../data/COIL20/images' # '../data/StanfordDogs/images'
    n_clusters = 50
    n_features = 100

    # Split the data
    train_img, test_img = split_data(dataset)

    # Make the BoW and train it on the training data
    bow = BoW(n_clusters, n_features)
    bow.train(train_img)

    win_name = "query | match"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1024, 600)

    # Find matches to every test image
    for i, img in enumerate(test_img):
        # Find the closest match in the training set
        idx = int(bow.predict(img))
        if idx != -1:
            # If a match was found make a show_image with the query and match image
            show_image = make_stackimage(img, train_img[idx])
        else:
            # If a match was not found make a show_image with the query image
            print("No features found")
            show_image = make_stackimage(img)

        # Show the result
        put_text(show_image, "bottom_center", f"Press any key.. ({i}/{len(test_img)}). ESC to stop")
        cv2.imshow(win_name, show_image)
        key = cv2.waitKey()
        if key == 27:
            break
