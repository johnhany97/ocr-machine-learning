"""Dummy classification system.

Skeleton code for a assignment solution.

To make a working solution you will need to rewrite parts
of the code below. In particular, the functions
reduce_dimensions and classify_page currently have
dummy implementations that do not do anything useful.

version: v1.0
"""
import numpy as np
import utils.utils as utils
from scipy import ndimage
import scipy
import cv2
from skimage.util import random_noise

def reduce_dimensions(features, model, mode="Train", split=0, start=1, end=60):
    """Dummy methods that just takes 1st 10 pixels.

    Params:
    feature_vectors_full - feature vectors stored as rows
       in a matrix
    model - a dictionary storing the outputs of the model
       training stage
    """
        # """Dimension reducation function
        # This function performs PCA dimension reduction then reduces them using
        # the best_pca function
        # Params:
        # feature_vectors_full - feature vectors stored as rows
        #    in a matrix
        # model - a dictionary storing the outputs of the model
        #    training stage
        # Returns:
        # Features with reduced dimensions to 10 dimensions
        # """
    if (mode == "Train"): #dimensionality reduction for train data
        # Generate PCA
        # grab training data and their labels from model
        labels_train_clean = np.array(model['labels_train'])[0:split + 1]
        labels_train_noisy = np.array(model['labels_train'])[split + 1: features.shape[0]]
        clean = features[0:split + 1]
        noisy = features[split + 1: features.shape[0]]
        # For clean data
        # Compute 40 principal components
        covx_clean = np.cov(clean, rowvar=0)
        N_clean = covx_clean.shape[0]
        w_clean, v_clean = scipy.linalg.eigh(covx_clean, eigvals=(N_clean-end, N_clean-start))
        # project data onto principal components
        pca_data_clean = np.dot((clean - np.mean(clean)), v_clean)
        # choose a subset from the above principal components
        print("-- CLEAN TRAIN DATA --")
        subset_clean = reduce_principal_components(pca_data_clean, labels_train_clean)
        # given the select best principal components, add to the model
        model['v_clean'] = v_clean[:, subset_clean].tolist()
        # For noisy data
        # Compute 40 principal components
        covx_noisy = np.cov(noisy, rowvar=0)
        N_noisy = covx_noisy.shape[0]
        w_noisy, v_noisy = scipy.linalg.eigh(covx_noisy, eigvals=(N_noisy-end, N_noisy-start))
        # project data onto principal components
        pca_data_noisy = np.dot((noisy - np.mean(noisy)), v_noisy)
        # choose a subset from the above principal components
        print("-- NOISY TRAIN DATA --")
        subset_noisy = reduce_principal_components(pca_data_noisy, labels_train_noisy)
        # given the select best principal components, add to the model
        model['v_noisy'] = v_noisy[:, subset_noisy].tolist()
        return pca_data_clean[:, subset_clean], pca_data_noisy[:, subset_noisy] #return the selected principal components on the data
    else: #dimensionality reduction for test data
        #compute how noisy
        count = 0
        for i in range(features.shape[0]):
            count += np.sum(features[i])
        determine =  count/(features.shape[0] * features.shape[1])
        if (determine > 240):
            v = np.array(model['v_clean'])
            return np.dot((features - np.mean(features)), v)
        else:
            v = np.array(model['v_noisy'])
            return np.dot((features - np.mean(features)), v)

def extract_char_given(fvectors_train, labels_train, label):
    """Extract character given
    This function takes a specific character(label) and returns all the
    occurances of this label from the training data
    Params:
    fvectors_train - training data (each row is a sample)
    labels_train - labels of the training data provided
    label - character representing the label we're extracting
    Returns:
    Array of all the occurances of the character (subset of training data)
    """
    data = []
    for i in range(labels_train.shape[0]):
        if (labels_train[i]==label):
            data.append(fvectors_train[i,:])
    return np.array(data)

def divergence(class1, class2, show=False):
    """compute a vector of 1-D divergences
    class1 - data matrix for class 1, each row is a sample
    class2 - data matrix for class 2
    returns: d12 - a vector of 1-D divergence scores
    """

    # Compute the mean and variance of each feature vector element
    m1 = np.mean(class1, axis=0)
    m2 = np.mean(class2, axis=0)
    v1 = np.var(class1, axis=0)
    v2 = np.var(class2, axis=0)
    if (show):
        print("Mean1: ", m1)
        print("Mean2: ", m2)
        print("Var1: ", var1)
        print("Var2: ", var2)

    # Plug mean and variances into the formula for 1-D divergence.
    # (Note that / and * are being used to compute multiple 1-D
    #  divergences without the need for a loop)
    d12 = 0.5 * (v1 / v2 + v2 / v1 - 2) + 0.5 * ( m1 - m2 ) * (m1 - m2) * (1.0 / v1 + 1.0 / v2)

    return d12

def reduce_principal_components(train_data, train_labels):
    allLetters = np.unique(train_labels)
    # Compute divergence for train data
    print("-- Computing divergence for train data")
    divergences = []
    for char1 in range(allLetters.shape[0] - 1):
        char1_data = extract_char_given(train_data, train_labels, allLetters[char1])
        if not (char1_data.shape[0] <= 1):
            for char2 in range(char1 + 1, allLetters.shape[0]):
                char2_data = extract_char_given(train_data, train_labels, allLetters[char2])
                if not (char2_data.shape[0] <= 1):
                    d12 = divergence(char1_data, char2_data)
                    divergences.append(d12)
    divNP = np.array(divergences)
    divNP.transpose()
    # sort highest scoring features
    print("-- Sorting highest scoring features")
    sorted_indexes = []
    for i in range(divNP.shape[0]):
        sorted_indexes.append(np.argsort(-divNP[i, :])[0:10])
    sortNP = np.array(sorted_indexes)
    # Compute how many times a specific feature was repeated
    print("-- Computing how many times a specific feature was repeated")
    feature_counter = {}
    for i in range(sortNP.shape[0]):
        for j in range(10):
            if sortNP[i, j] in feature_counter:
                feature_counter[sortNP[i, j]] += 1
            else:
                feature_counter[sortNP[i, j]] = 1
    # sort and keep only the feature number
    repeated_features = sorted(feature_counter, key=feature_counter.get, reverse=True)
    print("-- Compute best 10 from them using a sequential forward search")
    # compute best 10 from these
    a = -1
    b = -1
    c = -1
    d = -1
    e = -1
    f = -1
    g = -1
    h = -1
    x = -1
    y = -1
    scoreMax = -1
    for i in repeated_features:
        score = classify_training(train_data, train_labels, [i])
        if (score > scoreMax):
            a = i
            scoreMax = score
    scoreMax = -1
    for i in repeated_features:
        if not i == a:
            score = classify_training(train_data, train_labels, [a, i])
            if (score > scoreMax):
                b = i
                scoreMax = score
    scoreMax = -1
    for i in repeated_features:
        if not (i == a or i == b):
            score = classify_training(train_data, train_labels, [a, b, i])
            if (score > scoreMax):
                c = i
                scoreMax = score
    scoreMax = -1
    for i in repeated_features:
        if not (i == a or i == b or i == c):
            score = classify_training(train_data, train_labels, [a, b, c, i])
            if (score > scoreMax):
                d = i
                scoreMax = score
    scoreMax = -1
    for i in repeated_features:
        if not (i == a or i == b or i == c or i == d):
            score = classify_training(train_data, train_labels, [a, b, c, d, i])
            if (score > scoreMax):
                e = i
                scoreMax = score
    scoreMax = -1
    for i in repeated_features:
        if not (i == a or i == b or i == c or i == d or i == e):
            score = classify_training(train_data, train_labels, [a, b, c, d, e, i])
            if (score > scoreMax):
                f = i
                scoreMax = score
    scoreMax = -1
    for i in repeated_features:
        if not (i == a or i == b or i == c or i == d or i == e or i == f):
            score = classify_training(train_data, train_labels, [a, b, c, d, e, f, i])
            if (score > scoreMax):
                g = i
                scoreMax = score
    scoreMax = -1
    for i in repeated_features:
        if not (i == a or i == b or i == c or i == d or i == e or i == f or i == g):
            score = classify_training(train_data, train_labels, [a, b, c, d, e, f, g, i])
            if (score > scoreMax):
                h = i
                scoreMax = score
    scoreMax = -1
    for i in repeated_features:
        if not (i == a or i == b or i == c or i == d or i == e or i == f or i == g or i == h):
            score = classify_training(train_data, train_labels, [a, b, c, d, e, f, g, h, i])
            if (score > scoreMax):
                x = i
                scoreMax = score
    scoreMax = -1
    for i in repeated_features:
        if not (i == a or i == b or i == c or i == d or i == e or i == f or i == g or i == h or i == x):
            score = classify_training(train_data, train_labels, [a, b, c, d, e, f, g, h, x, i])
            if (score > scoreMax):
                y = i
                scoreMax = score
    return [a, b, c, d, e, f, g, h, x, y]

def classify_training(data, labels, features):
    """Nearest neighbour classification.
    Params
    train - data matrix storing training data, one sample per row (whatever number of rows and columns has features)
    train_label - a vector storing the training data labels
    features - a vector if indices that select the feature to use
    returns: (score) - a percentage of correct labels
    """
    # # Select the desired features from the training and test data
    split = int(data.shape[0] * 0.5)
    train = data[0:split, features]
    test = data[split:data.shape[0], features]
    train_labels = labels[0:split]
    test_labels = labels[split:data.shape[0]]
    # Super compact implementation of nearest neighbour
    x= np.dot(test, train.transpose())
    modtest = np.sqrt(np.sum(test * test, axis=1))
    modtrain = np.sqrt(np.sum(train * train, axis=1))
    dist = x / np.outer(modtest, modtrain.transpose()); # cosine distance
    nearest = np.argmax(dist, axis=1)
    mdist = np.max(dist, axis=1)
    label = train_labels[nearest]
    score = (100.0 * sum(test_labels[:] == label)) / label.shape[0]
    return score

def get_bounding_box_size(images):
    """Compute bounding box size given list of images."""
    height = max(image.shape[0] for image in images)
    width = max(image.shape[1] for image in images)
    return height, width

def correct_errors(page, labels, bboxes, model):
    """Dummy error correction. Returns labels unchanged.

    parameters:

    page - 2d array, each row is a feature vector to be classified
    labels - the output classification label for each feature vector
    bboxes - 2d array, each row gives the 4 bounding box coords of the character
    model - dictionary, stores the output of the training stage
    """

    return labels


def images_to_feature_vectors(images, bbox_size=None):
    """Reformat characters into feature vectors.

    Takes a list of images stored as 2D-arrays and returns
    a matrix in which each row is a fixed length feature vector
    corresponding to the image.abs

    Params:
    images - a list of images stored as arrays
    bbox_size - an optional fixed bounding box size for each image
    """

    # If no bounding box size is supplied then compute a suitable
    # bounding box by examining sizes of the supplied images.
    if bbox_size is None:
        bbox_size = get_bounding_box_size(images)

    bbox_h, bbox_w = bbox_size
    nfeatures = bbox_h * bbox_w
    fvectors = np.empty((len(images), nfeatures))
    for i, image in enumerate(images):
        padded_image = np.ones(bbox_size) * 255
        h, w = image.shape
        h = min(h, bbox_h)
        w = min(w, bbox_w)
        padded_image[0:h, 0:w] = image[0:h, 0:w]
        fvectors[i, :] = padded_image.reshape(1, nfeatures)
    return fvectors

# The three functions below this point are called by train.py
# and evaluate.py and need to be provided.
def process_training_data(train_page_names):
    """Perform the training stage and return results in a dictionary.

    Params:
    train_page_names - list of training page names
    """
    print('- Reading data')
    images_train = []
    labels_train = []
    for page_name in train_page_names:
        images_train = utils.load_char_images(page_name, images_train)
        labels_train = utils.load_labels(page_name, labels_train)
    labels_train = np.array(labels_train)
    print('- Extracting features from training data')
    bbox_size = get_bounding_box_size(images_train)
    fvectors_train_full = images_to_feature_vectors(images_train, bbox_size)
    clean = fvectors_train_full[0::2]
    noisy = fvectors_train_full[1::2]
    model_data = dict()
    model_data['bbox_size'] = bbox_size
    model_data['labels_train'] = np.concatenate((labels_train[0::2], labels_train[1::2])).tolist()
    print('- Adding noise to a half of the data')
    # # Gaussian noise
    # for i in range(noisy.shape[0]):
    #     row = noisy[i].shape[0]
    #     mean = 0
    #     var = 0.1
    #     sigma = var**0.5
    #     gauss = np.random.normal(mean,sigma,(row))
    #     gauss = gauss.reshape(row)
    #     noisy[i] += gauss
    # Salt and pepper noise
    for i in range(noisy.shape[0]):
        # Makr a copy
        copy = noisy[i]
        # Convert to floats between and inclusive to 0 and 1
        copy.astype(np.float16, copy=False)
        copy = np.multiply(copy, (1/255))
        # Create some noise
        noise = np.random.randint(20, size=(copy.shape[0]))
        # When the noise has a zero, add a pepper to the copy
        # Pepper
        copy = np.where(noise==0, 0, copy)
        # When the noise has a value equal to the top, add a salt to the copy
        # Salt
        copy = np.where(noise==(19), 1, copy)
        # Convert back to values out of 255 (RGB)
        noisy[i] = np.multiply(copy, (255))

    print('- Reducing to 10 dimensions')
    fvectors_train_clean, fvectors_train_noisy = reduce_dimensions(np.concatenate((clean, noisy), axis=0), model_data, "Train", noisy.shape[0])

    model_data['fvectors_train'] = np.concatenate((fvectors_train_clean, fvectors_train_noisy)).tolist()

    return model_data


def load_test_page(page_name, model):
    """Load test data page.

    This function must return each character as a 10-d feature
    vector with the vectors stored as rows of a matrix.

    Params:
    page_name - name of page file
    model - dictionary storing data passed from training stage
    """
    bbox_size = model['bbox_size']
    images_test = utils.load_char_images(page_name)
    fvectors_test = images_to_feature_vectors(images_test, bbox_size)
    #compute how noisy
    count = 0
    for i in range(fvectors_test.shape[0]):
        count += np.sum(fvectors_test[i])
    determine =  count/(fvectors_test.shape[0] * fvectors_test.shape[1])
    # denoise images by applying median filter
    if (determine < 239.0):
        for i in range(fvectors_test.shape[0]):
            fvectors_test[i] = ndimage.median_filter(fvectors_test[i], 3)
    # Perform the dimensionality reduction.
    fvectors_test_reduced = reduce_dimensions(fvectors_test, model, "Test")
    return fvectors_test_reduced

def classify_page(page, model):
    """Dummy classifier. Always returns first label.

    parameters:

    page - matrix, each row is a feature vector to be classified
    model - dictionary, stores the output of the training stage
    """
    fvectors_train = np.array(model['fvectors_train'])
    labels_train = np.array(model['labels_train'])
    x = np.dot(page, fvectors_train.transpose())
    modtest = np.sqrt(np.sum(page * page, axis=1))
    modtrain = np.sqrt(np.sum(fvectors_train * fvectors_train, axis=1))
    dist = x / np.outer(modtest, modtrain.transpose()); # cosine distance
    nearest = np.argmax(dist, axis=1)

    return labels_train[nearest]

#########################################################################
# OLD VERSION OF THIS FILE
#########################################################################
# Differences:
# For the train data:
#   We don't seperate or modify the data, just reduce their dimensions
#   using pca and find the best 10 principal components that can classify them
# For the classifier:
#   There are two implementations of K-Nearest neighbor below and a Nearest
#   neighbor implementation (that scores better than the other two)
# All data uses the same feature set to reduce
# Attempts were made to fix errors as well, however it slowed down the classifier
# by a significant amount, so not used in the final implementation
# Best result acquired:
# Page 1: score = 97.0% correct
# Page 2: score = 97.4% correct
# Page 3: score = 82.1% correct
# Page 4: score = 56.5% correct
# Page 5: score = 38.6% correct
# Page 6: score = 28.8% correct
#########################################################################
# import numpy as np
# import utils.utils as utils
# import scipy.linalg
# from scipy import ndimage
# import difflib
#
# def reduce_dimensions(feature_vectors_full, model):
#     """Dimension reducation function
#     This function performs PCA dimension reduction then reduces them using
#     the best_pca function
#     Params:
#     feature_vectors_full - feature vectors stored as rows
#        in a matrix
#     model - a dictionary storing the outputs of the model
#        training stage
#     Returns:
#     Features with reduced dimensions to 10 dimensions
#     """
#     labels_train = np.array(model['labels_train'])
#     #Compute 40 principal components (from the second one)
#     covx = np.cov(feature_vectors_full, rowvar=0)
#     N = covx.shape[0]
#     w, v = scipy.linalg.eigh(covx, eigvals=(N-50, N-1)) #N-11, N-2
#     v = np.fliplr(v)
#     #Project data onto principal components
#     pca_data = np.dot((feature_vectors_full - np.mean(feature_vectors_full)), v)
#     #Compute best 10 out of them
#     subset = best_pca(pca_data, labels_train)
#     model['v'] = v[:, subset].tolist()
#     return pca_data[:, subset]
#
# def get_bounding_box_size(images):
#     """Compute bounding box size given list of images."""
#     height = max(image.shape[0] for image in images)
#     width = max(image.shape[1] for image in images)
#     return height, width
#
# def images_to_feature_vectors(images, bbox_size=None):
#     """Reformat characters into feature vectors.
#     Takes a list of images stored as 2D-arrays and returns
#     a matrix in which each row is a fixed length feature vector
#     corresponding to the image.abs
#     Params:
#     images - a list of images stored as arrays
#     bbox_size - an optional fixed bounding box size for each image
#     """
#
#     # If no bounding box size is supplied then compute a suitable
#     # bounding box by examining sizes of the supplied images.
#     if bbox_size is None:
#         bbox_size = get_bounding_box_size(images)
#
#     bbox_h, bbox_w = bbox_size
#     nfeatures = bbox_h * bbox_w
#     fvectors = np.empty((len(images), nfeatures))
#     for i, image in enumerate(images):
#         padded_image = np.ones(bbox_size) * 255
#         h, w = image.shape
#         h = min(h, bbox_h)
#         w = min(w, bbox_w)
#         padded_image[0:h, 0:w] = image[0:h, 0:w]
#         fvectors[i, :] = padded_image.reshape(1, nfeatures)
#
#     return fvectors
#
# def process_training_data(train_page_names):
#     """Perform the training stage and return results in a dictionary.
#     Params:
#     train_page_names - list of training page names
#     """
#     print('- Reading data')
#     images_train = []
#     labels_train = []
#     for page_name in train_page_names:
#         images_train = utils.load_char_images(page_name, images_train)
#         labels_train = utils.load_labels(page_name, labels_train)
#     labels_train = np.array(labels_train)
#
#     print('- Extracting features from training data')
#     bbox_size = get_bounding_box_size(images_train)
#     fvectors_train_full = images_to_feature_vectors(images_train, bbox_size)
#
#     model_data = dict()
#     model_data['labels_train'] = labels_train.tolist()
#     model_data['bbox_size'] = bbox_size
#
#     print('- Load in dictionary')
#     english_dictionary = np.genfromtxt('code/data/wordsEn.txt', dtype='str')
#     model_data['english'] = english_dictionary.tolist()
#     print('- Reducing to 10 dimensions')
#     fvectors_train = reduce_dimensions(fvectors_train_full, model_data)
#     model_data['fvectors_train'] = fvectors_train.tolist()
#     return model_data
#
# def best_pca(train_data, train_labels):
#     """Best principal components
#     This function attempts to choose the best subset of the given principal
#     components for the data
#     Params:
#     train_data - Training data with PCA computed upon (40 or 50 features (more than 10))
#     train_labels - Labels for the training data identifying the samples
#     Process:
#     1) Computes divergence for all classes
#     2) Sorts highest scoring features for all divergences
#     3) Computes how many times a specific feature was repeated
#     4) Does sequential forward search to choose best 10 features (principal components)
#     Returns:
#     Array of indices of 10 best scoring principal components
#     """
#     allLetters = np.unique(train_labels)
#     # Compute divergence for train data
#     print("-- Computing divergence for train data")
#     divergences = []
#     for char1 in range(allLetters.shape[0] - 1):
#         char1_data = extract_char_given(train_data, train_labels, allLetters[char1])
#         if not (char1_data.shape[0] <= 1):
#             for char2 in range(char1 + 1, allLetters.shape[0]):
#                 char2_data = extract_char_given(train_data, train_labels, allLetters[char2])
#                 if not (char2_data.shape[0] <= 1):
#                     d12 = divergence(char1_data, char2_data)
#                     divergences.append(d12)
#     divNP = np.array(divergences)
#     divNP.transpose()
#     # sort highest scoring features
#     print("-- Sorting highest scoring features")
#     sorted_indexes = []
#     for i in range(divNP.shape[0]):
#         sorted_indexes.append(np.argsort(-divNP[i, :])[0:10])
#     sortNP = np.array(sorted_indexes)
#     # Compute how many times a specific feature was repeated
#     print("-- Computing how many times a specific feature was repeated")
#     feature_counter = {}
#     for i in range(sortNP.shape[0]):
#         for j in range(10):
#             if sortNP[i, j] in feature_counter:
#                 feature_counter[sortNP[i, j]] += 1
#             else:
#                 feature_counter[sortNP[i, j]] = 1
#     # sort and keep only the feature number
#     repeated_features = sorted(feature_counter, key=feature_counter.get, reverse=True)
#     print("-- Compute best 10 from them using a sequential forward search")
#     # compute best 10 from these
#     a = -1
#     b = -1
#     c = -1
#     d = -1
#     e = -1
#     f = -1
#     g = -1
#     h = -1
#     x = -1
#     y = -1
#     scoreMax = -1
#     for i in repeated_features:
#         score = classify_training(train_data, train_labels, [i])
#         if (score > scoreMax):
#             a = i
#             scoreMax = score
#     scoreMax = -1
#     for i in repeated_features:
#         if not i == a:
#             score = classify_training(train_data, train_labels, [a, i])
#             if (score > scoreMax):
#                 b = i
#                 scoreMax = score
#     scoreMax = -1
#     for i in repeated_features:
#         if not (i == a or i == b):
#             score = classify_training(train_data, train_labels, [a, b, i])
#             if (score > scoreMax):
#                 c = i
#                 scoreMax = score
#     scoreMax = -1
#     for i in repeated_features:
#         if not (i == a or i == b or i == c):
#             score = classify_training(train_data, train_labels, [a, b, c, i])
#             if (score > scoreMax):
#                 d = i
#                 scoreMax = score
#     scoreMax = -1
#     for i in repeated_features:
#         if not (i == a or i == b or i == c or i == d):
#             score = classify_training(train_data, train_labels, [a, b, c, d, i])
#             if (score > scoreMax):
#                 e = i
#                 scoreMax = score
#     scoreMax = -1
#     for i in repeated_features:
#         if not (i == a or i == b or i == c or i == d or i == e):
#             score = classify_training(train_data, train_labels, [a, b, c, d, e, i])
#             if (score > scoreMax):
#                 f = i
#                 scoreMax = score
#     scoreMax = -1
#     for i in repeated_features:
#         if not (i == a or i == b or i == c or i == d or i == e or i == f):
#             score = classify_training(train_data, train_labels, [a, b, c, d, e, f, i])
#             if (score > scoreMax):
#                 g = i
#                 scoreMax = score
#     scoreMax = -1
#     for i in repeated_features:
#         if not (i == a or i == b or i == c or i == d or i == e or i == f or i == g):
#             score = classify_training(train_data, train_labels, [a, b, c, d, e, f, g, i])
#             if (score > scoreMax):
#                 h = i
#                 scoreMax = score
#     scoreMax = -1
#     for i in repeated_features:
#         if not (i == a or i == b or i == c or i == d or i == e or i == f or i == g or i == h):
#             score = classify_training(train_data, train_labels, [a, b, c, d, e, f, g, h, i])
#             if (score > scoreMax):
#                 x = i
#                 scoreMax = score
#     scoreMax = -1
#     for i in repeated_features:
#         if not (i == a or i == b or i == c or i == d or i == e or i == f or i == g or i == h or i == x):
#             score = classify_training(train_data, train_labels, [a, b, c, d, e, f, g, h, x, i])
#             if (score > scoreMax):
#                 y = i
#                 scoreMax = score
#     return [a, b, c, d, e, f, g, h, x, y]
#
# def classify_training(data, labels, features):
#     """Nearest neighbour classification.
#     Params
#     train - data matrix storing training data, one sample per row (whatever number of rows and columns has features)
#     train_label - a vector storing the training data labels
#     features - a vector if indices that select the feature to use
#     returns: (score) - a percentage of correct labels
#     """
#
#     # Select the desired features from the training and test data
#     train = data[0::2, features]
#     test = data[1::2, features]
#     train_labels = labels[0::2]
#     test_labels = labels[1::2]
#     # Super compact implementation of nearest neighbour
#     x= np.dot(test, train.transpose())
#     modtest = np.sqrt(np.sum(test * test, axis=1))
#     modtrain = np.sqrt(np.sum(train * train, axis=1))
#     dist = x / np.outer(modtest, modtrain.transpose()); # cosine distance
#     nearest = np.argmax(dist, axis=1)
#     mdist = np.max(dist, axis=1)
#     label = train_labels[nearest]
#     score = (100.0 * sum(test_labels[:] == label)) / label.shape[0]
#
#     return score
#
# def extract_char_given(fvectors_train, labels_train, label):
#     """Extract character given
#     This function takes a specific character(label) and returns all the
#     occurances of this label from the training data
#     Params:
#     fvectors_train - training data (each row is a sample)
#     labels_train - labels of the training data provided
#     label - character representing the label we're extracting
#     Returns:
#     Array of all the occurances of the character (subset of training data)
#     """
#     data = []
#     for i in range(labels_train.shape[0]):
#         if (labels_train[i]==label):
#             data.append(fvectors_train[i,:])
#     return np.array(data)
#
# def divergence(class1, class2, show=False):
#     """compute a vector of 1-D divergences
#     class1 - data matrix for class 1, each row is a sample
#     class2 - data matrix for class 2
#     returns: d12 - a vector of 1-D divergence scores
#     """
#
#     # Compute the mean and variance of each feature vector element
#     m1 = np.mean(class1, axis=0)
#     m2 = np.mean(class2, axis=0)
#     v1 = np.var(class1, axis=0)
#     v2 = np.var(class2, axis=0)
#     if (show):
#         print("Mean1: ", m1)
#         print("Mean2: ", m2)
#         print("Var1: ", var1)
#         print("Var2: ", var2)
#
#     # Plug mean and variances into the formula for 1-D divergence.
#     # (Note that / and * are being used to compute multiple 1-D
#     #  divergences without the need for a loop)
#     d12 = 0.5 * (v1 / v2 + v2 / v1 - 2) + 0.5 * ( m1 - m2 ) * (m1 - m2) * (1.0 / v1 + 1.0 / v2)
#
#     return d12
#
# def load_test_page(page_name, model):
#     """Load test data page.
#     This function must return each character as a 10-d feature
#     vector with the vectors stored as rows of a matrix.
#     Params:
#     page_name - name of page file
#     model - dictionary storing data passed from training stage
#     """
#     bbox_size = model['bbox_size']
#     images_test = utils.load_char_images(page_name)
#     fvectors_test = images_to_feature_vectors(images_test, bbox_size)
#     # denoise images by applying median filter
#     for i in range(fvectors_test.shape[0]):
#         fvectors_test[i] = ndimage.median_filter(fvectors_test[i], 3)
#     # Perform the dimensionality reduction.
#     v = np.array(model['v'])
#     fvectors_test_reduced = np.dot((fvectors_test - np.mean(fvectors_test)), v)
#     return fvectors_test_reduced
#
# def correct_errors(page, labels, bboxes, model):
#     """Error correction function
#     Takes labels, divides them into words then starts checking if words have potential matches
#     if they do, we check if the first one is the exact word (no need to update)
#     if they're not the same, we check the potential matches if there's a string with only one letter difference
#     problems:
#         - inefficient
#         - slows down computation
#     parameters:
#     page - 2d array, each row is a feature vector to be classified
#     labels - the output classification label for each feature vector
#     bboxes - 2d array, each row gives the 4 bounding box coords of the character
#     model - dictionary, stores the output of the training stage
#     """
#     # english = model['english']
#     # all_words = divide_to_words(labels, bboxes)
#     # for i in range(len(all_words)):
#     #     potential_match = difflib.get_close_matches(all_words[i], english)
#     #     if (len(all_words[i]) > 1 and len(potential_match) > 0 and potential_match[0] != all_words[i]):
#     #         print('checking word ', all_words[i], ' potential matches')
#     #         for j in potential_match:
#     #             difference = sum (j != all_words[i] for i in range(len(j)))
#     #             if (difference == 1):
#     #                 print('we will change: ',  all_words[i], ' to ', j)
#
#     return labels
#
# def divide_to_words(labels, bboxes):
#     all_words = []
#     current_word = labels[0]
#     for i in range(1, labels.shape[0]):
#         if (bboxes[i, 0] - bboxes[i-1, 2] < 6):
#             if (labels[i] != '.' and labels[i] != ',' and labels[i] != ':' and labels[i] != '!' and labels[i] != '?' and labels[i] != ';'):
#                 current_word += labels[i]
#         else:
#             all_words.append(current_word)
#             if (labels[i] != '.' and labels[i] != ',' and labels[i] != ':' and labels[i] != '!' and labels[i] != '?' and labels[i] != ';'):
#                 current_word = labels[i]
#     return all_words
#
#
# def classify_page(page, model):
#     """Nearest neighbour classification.
#     parameters:
#     page - matrix, each row is a feature vector to be classified
#     model - dictionary, stores the output of the training stage
#     returns:
#     classification labels
#     """
#     # Extract training data and their labels
#     fvectors_train = np.array(model['fvectors_train'])
#     labels_train = np.array(model['labels_train'])
#
#     # k-nearest neighbor implementation
#     # k = int(page.shape[0]**0.5)
#     # Page 1: score = 89.3% correct
#     # Page 2: score = 89.4% correct
#     # Page 3: score = 77.7% correct
#     # Page 4: score = 54.7% correct
#     # Page 5: score = 36.6% correct
#     # Page 6: score = 27.8% correct
#     # k = 25
#     # Page 1: score = 93.2% correct
#     # Page 2: score = 92.9% correct
#     # Page 3: score = 78.0% correct
#     # Page 4: score = 52.5% correct
#     # Page 5: score = 32.4% correct
#     # Page 6: score = 23.9% correct
#     # k = 20
#     # Page 1: score = 94.0% correct
#     # Page 2: score = 93.5% correct
#     # Page 3: score = 78.4% correct
#     # Page 4: score = 52.1% correct
#     # Page 5: score = 31.9% correct
#     # Page 6: score = 23.2% correct
#     # labels = []
#     # k = 20#int(page.shape[0]**0.5)
#     # for i in range(page.shape[0]):
#     #     # euclidean_dist = []
#     #     # for j in range(fvectors_train.shape[0]):
#     #     #     euclidean_dist.append(np.linalg.norm(page[i]-fvectors_train[i]))
#     #     euclidean_dist = [np.linalg.norm(page[i]-fvectors_train[q]) for q in range(fvectors_train.shape[0])]
#     #     # euclidean_dist = np.linalg.norm(page[i]-fvectors_train)
#     #     dist_n = np.array(euclidean_dist)
#     #     sort_dist_k = np.argsort(dist_n)[0:k]
#     #     label_dist = []
#     #     for w in range(k):
#     #         # print(sort_dist_k[w])
#     #         label_dist.append(labels_train[sort_dist_k[w]])
#     #         # label_dist.append(labels_train[euclidean_dist[w]])
#     #     labels.append(max(set(label_dist), key=label_dist.count))
#     # return np.array(labels)
#
#     # Super compact implementation of nearest neighbour
#     x = np.dot(page, fvectors_train.transpose())
#     modtest = np.sqrt(np.sum(page * page, axis=1))
#     modtrain = np.sqrt(np.sum(fvectors_train * fvectors_train, axis=1))
#     dist = x / np.outer(modtest, modtrain.transpose()); # cosine distance
#     nearest = np.argmax(dist, axis=1)
#     return labels_train[nearest]
# #     #Scores:
# #     #Page 1: score = 11.3% correct
# #     #Page 2: score = 12.3% correct
# #     #Page 3: score = 12.9% correct
# #     #Page 4: score = 12.7% correct
# #     #Page 5: score = 12.9% correct
# #     #Page 6: score = 12.6% correct
# #     # k = 40
# #     # labels = []
# #     # for i in range(page.shape[0]):
# #     #     neighbors = knn_get_neighbors(fvectors_train, page[i, :], k)
# #     #     labels.append(knn_get_label(neighbors, labels_train))
# #     #
# #     # return np.array(labels)
# #
# # def knn_distance_euclidean(sample1, sample2, length):
# #     distance = 0
# #     for x in range(length):
# #         distance += pow((sample1[x] - sample2[x]), 2) # (a - b) ^ 2
# #         return math.sqrt(distance) # (distance) ^ 0.5
# #
# # def knn_get_neighbors(training_samples, unidentified, k):
# #     distances = []
# #     length = unidentified.shape[0] # amount of features in a sample
# #     for i in range(training_samples.shape[0]):
# #         distances.append((i, knn_distance_euclidean(unidentified, training_samples[i], length)))
# #     # sort based on the distances we found (to find the closeset)
# #     distances.sort(key=operator.itemgetter(1))
# #     neighbors = []
# #     for i in range(k):
# #         neighbors.append(distances[i][0])
# #     return neighbors
# #
# # def knn_get_label(neighbors, labels):
# #     potential_labels = {}
# #     for i in range(len(neighbors)):
# #         label = labels[i]
# #         if label in potential_labels:
# #             potential_labels[label] += 1
# #         else:
# #             potential_labels[label] = 1
# #     sort_list = sorted(potential_labels.items(), key=operator.itemgetter(1), reverse=True)
# #     return sort_list[0][0]
