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
