"""Classification system

Implemented by John Ayad
Registration # 160153363

version: v1.0
"""
import numpy as np
import utils.utils as utils
from scipy import ndimage
import scipy
import cv2
from skimage.util import random_noise
import operator
from nltk.corpus import brown
from autocorrect import spell

def reduce_dimensions(features, model, mode="Train", split=0, start=1, end=60):
    """Dimension reducation function

    This function performs PCA dimension reduction then reduces the dimensions principal
    components to exactly 10 by calling the reduce_principal_components function

    Params:
    feature_vectors_full - feature vectors stored as rows in a matrix
    model - a dictionary storing the outputs of the model training stage
    mode - String, either "Train" or "Test" to define type of dimensionality reduction being performed
    split - int to represent where data needs to be split if any (used mainly by training) (default = 0)
    start - int representing which principal component to start at (default = 1)
    end - int representing which principal component to end at (default = 60)

    Returns:
    In case of training:
        Tuple of features with reduced dimensions to 10 dimensions, first one is for clean data and second is for noisy
    In case of testing:
        Features with reduced dimensions to 10
    """
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
        # compute how noisy
        count = 0
        for i in range(features.shape[0]):
            count += np.sum(features[i])
        determine =  count/(features.shape[0] * features.shape[1])
        if (determine > 240): # considered clean
            v = np.array(model['v_clean'])
            return np.dot((features - np.mean(features)), v)
        else: #considered noisy
            v = np.array(model['v_noisy'])
            return np.dot((features - np.mean(features)), v)

def extract_char_given(fvectors_train, labels_train, label):
    """Extract character given

    This function takes a specific character (label) and returns all the
    occurances of this label from the training data

    Paramaters:
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
    """Computs a vector of 1-D Divergence

    Parameters:
    class1 - data matrix for class 1, each row is a sample
    class2 - data matrix for class 2, each row is a sample
    show - boolean that if true, prints out means and variances (used in debugging)

    Returns:
    d12 - a vector of 1-D divergence scores
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
    """Reduce principal components function

    This function is given some data and their labels and performs the following algorithm
    to choose the best set of principal components.
    1) Computes divergence between all possible pairs of the labels on the data
    2) Sorts highest scoring features(by their divergence) and choose best 20 for each
    3) Compute from al the 20s how many time each feature was repeated (also
        used to get only unique values and to have a good starting value for the following step)
    4) Sort them
    5) Compute best 10 features (principal components) using a sequential forward search (scoring used is classify_training function)

    This function is used in the training stage and is to be called twice during
    the training stage. Once on clean training data and another time on the noisy
    training data to obtain the best principal components for each scenario.

    Parameters:
    train_data - training data (each row is a sample)
    train_labels - labels of the training data provided

    Returns:
    A list of the selected 10 indices of the best scoring principal commponents
    """
    # All possible labels that could be found in given training data
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
    print("-- Sorting highest scoring features and maintaining best 20 for each one")
    sorted_indexes = []
    for i in range(divNP.shape[0]):
        sorted_indexes.append(np.argsort(-divNP[i, :])[0:20])
    sortNP = np.array(sorted_indexes)
    # Compute how many times a specific feature was repeated
    print("-- Computing how many times a specific feature was repeated from the sets of 20s")
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

    This classifier takes in data and their actual labels, attempt to classify
    using nearest neighbor then checks how accurate it was and returns that score

    Paramaters:
    data - data matrix storing training data, one sample per row (whatever number of rows and columns has features)
    labels - a vector storing the training data labels
    features - a vector if indices that select the feature to use

    Returns:
    score - a percentage of correct labels
    """
    # Select the desired features from the training and test data
    split = int(data.shape[0] * 0.5)
    train = data[0:split, features]
    test = data[split:data.shape[0], features]
    train_labels = labels[0:split]
    test_labels = labels[split:data.shape[0]]
    # Nnearest neighbor
    x= np.dot(test, train.transpose())
    modtest = np.sqrt(np.sum(test * test, axis=1))
    modtrain = np.sqrt(np.sum(train * train, axis=1))
    dist = x / np.outer(modtest, modtrain.transpose()); # cosine distance
    # Find nearest neighbors
    nearest = np.argmax(dist, axis=1)
    # Compute labels
    label = train_labels[nearest]
    # Return score
    return (100.0 * sum(test_labels[:] == label)) / label.shape[0]

def get_bounding_box_size(images):
    """Compute bounding box size given list of images."""
    height = max(image.shape[0] for image in images)
    width = max(image.shape[1] for image in images)
    return height, width

def correct_errors(page, labels, bboxes, model, use=True, version='One'):
    """Error correction function

    Version One: (Uses NLTK and autocorrect libraries)
    Takes labels, divides them into words then starts checking if they're spelled correctly,
    if not, we autocorrect the word immediately.
    Else, we move on to the following word
    Results obtained from this method:
    Page 1: score = 94.3% correct
    Page 2: score = 95.1% correct
    Page 3: score = 88.0% correct
    Page 4: score = 60.5% correct
    Page 5: score = 60.0% correct
    Page 6: score = 49.4% correct

    Version Two: (Required "English" dictionary in the model)
    Takes labels, divides them into words then starts checking if words have potential matches
    if they do, we check if the first one is the exact word (no need to update)
    if they're not the same, we check the potential matches if there's a string with only one letter difference

    Problems:
        - inefficient
        - slows down computation
        - defining word-seperating threshold is inconsistent

    Parameters:
    page - 2d array, each row is a feature vector to be classified
    labels - the output classification label for each feature vector
    bboxes - 2d array, each row gives the 4 bounding box coords of the character
    model - dictionary, stores the output of the training stage
    use - Boolean, True if we're to use the error correction code (default = False)
    version - 'One'/'Two' to define which implementation to use (default = 'One')

    Returns:
    Labels after being subjected to error correction, if any
    """
    if (use): #Do some error correction
        all_words, current_labels = divide_to_words(labels, bboxes)
        if (version=='One'): #Uses nltk and autocorrect libraries
            english = set(brown.words())
            # Check if word is in set
            for i in range(len(all_words)):
                if not (all_words[i] in english):
                    old = all_words[i]
                    all_words[i] = spell(all_words[i])
                    if (len(old) < len(all_words[i])):
                        for j in range(len(old)):
                            if old[j] != all_words[i][j]:
                                indices = current_labels[i]
                                labels[int(indices[j])] = all_words[i][j]
                    else:
                        for j in range(len(all_words[i])):
                            if old[j] != all_words[i][j]:
                                indices = current_labels[i]
                                labels[int(indices[j])] = all_words[i][j]
        else: # Version 2 (Slower, less-accurate and required dictionary in model)
            if 'english' in model: #checks that model has the dictionary
                english = model['english']
                potential_match = difflib.get_close_matches(all_words[i], english)
                if (len(all_words[i]) > 1 and len(potential_match) > 0 and potential_match[0] != all_words[i]):
                    print('checking word ', all_words[i], ' potential matches')
                    for j in potential_match:
                        difference = sum (j != all_words[i] for i in range(len(j)))
                        if (difference == 1):
                            if (len(all_words[i]) < len(potential_match[j])):
                                for k in range(len(all_words[i])):
                                    if all_words[i][k] != potential_match[j][k]:
                                        indices = current_labels[i]
                                        labels[int(indices[j])] = all_words[i][j]
                            else:
                                for k in range(len(potential_match[j])):
                                    if all_words[j][k] != potential_match[j][k]:
                                        indices = current_labels[i]
                                        labels[int(indices[k])] = all_words[i][k]
    return labels

def divide_to_words(labels, bboxes):
    """Divide to Words function

    This function takes in some labels and their bounding boxes.
    It then starts to divide the labels into sequences of words by the help
    of somme thresholds defining when a difference between two bounding boxes should
    be considered as a space between two words

    Parameters:
    labels - each row is a label
    bboxes - each row is the bounding box for this label

    Returns:
    all_words - Array of strings containing words
    all_labels - 2D Array of Strings where each index corresponds to all_words' word label indices (Used to be able to update a specific label)
    """
    all_words = []
    all_labels = []
    current_word = labels[0]
    current_label = ['0']
    for i in range(1, labels.shape[0]):
        if (bboxes[i, 0] - bboxes[i-1, 2] < 7):
            if (labels[i] != '.' and labels[i] != ',' and labels[i] != ':' and labels[i] != '!' and labels[i] != '?' and labels[i] != ';'):
                current_word += labels[i]
                current_label.append(str(i))
        else:
            all_words.append(current_word)
            all_labels.append(current_label)
            if (labels[i] != '.' and labels[i] != ',' and labels[i] != ':' and labels[i] != '!' and labels[i] != '?' and labels[i] != ';'):
                current_word = labels[i]
                current_label = [str(i)]
    return all_words, all_labels

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
def process_training_data(train_page_names, noise='saltandpepper'):
    """Perform the training stage and return results in a dictionary.

    Params:
    train_page_names - list of training page names
    noise - String, default is "saltandpepper", other option is "gaussian" and is
        used to determine type of noise to use
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
    # combine labels differently to match the way we use train data
    model_data['labels_train'] = np.concatenate((labels_train[0::2], labels_train[1::2])).tolist()
    print('- Adding noise to a half of the data')
    if (noise == 'gaussian'):
        # Gaussian noise
        print('-- Gaussian noise')
        for i in range(noisy.shape[0]):
            gauss = np.random.normal(0, 0.1**0.5, (noisy[i].shape[0])).reshape(noisy[i].shape[0])
            noisy[i] += gauss
    else:
        # Salt and pepper noise
        print('-- Salt and pepper noise')
        for i in range(noisy.shape[0]):
            # Makr a copy
            copy = noisy[i]
            # Convert to floats between and inclusive to 0 and 1
            copy.astype(np.float16, copy=False)
            copy = np.multiply(copy, (1/255))
            # Create some noise
            noise = np.random.randint(20, size=(copy.shape[0]))
            # When the noise has a zero, add a pepper to the copy
            copy = np.where(noise==0, 0, copy) # pepper (black is = 0)
            # When the noise has a value equal to the top, add a salt to the copy
            copy = np.where(noise==(19), 1, copy) # salt (white is = 1)
            # Convert back to values out of 255 (RGB)
            noisy[i] = np.multiply(copy, (255))

    print('- Reducing to 10 dimensions')
    fvectors_train_clean, fvectors_train_noisy = reduce_dimensions(np.concatenate((clean, noisy), axis=0), model_data, "Train", noisy.shape[0])
    # add training clean and noisy samples together and save in model
    model_data['fvectors_train'] = np.concatenate((fvectors_train_clean, fvectors_train_noisy)).tolist()

    return model_data

def load_test_page(page_name, model):
    """Load test data page.

    This function must return each character as a 10-d feature
    vector with the vectors stored as rows of a matrix. It also saves in the model
    a determine on whether a page was noisy or not. If a page was determined to be noisy,
    a median-filter is applied which has shown to make characters on a page more
    clear for the classifier.

    Params:
    page_name - name of page file
    model - dictionary storing data passed from training stage
    """
    bbox_size = model['bbox_size']
    images_test = utils.load_char_images(page_name)
    fvectors_test = images_to_feature_vectors(images_test, bbox_size)
    # compute how noisy
    count = 0
    for i in range(fvectors_test.shape[0]):
        count += np.sum(fvectors_test[i])
    determine =  count/(fvectors_test.shape[0] * fvectors_test.shape[1])
    # denoise images by applying median filter
    if (determine < 239.0):
        for i in range(fvectors_test.shape[0]):
            fvectors_test[i] = ndimage.median_filter(fvectors_test[i], 3)
    # save the fact it was noisy if it was
    if 'test_noisy' in model:
        x = np.array(model['test_noisy'])
        if (determine < 239.0):
            x = np.append(x, True)
        else:
            x = np.append(x, False)
        model['test_noisy'] = x.tolist()
    else:
        if (determine < 239.0):
            model['test_noisy'] = [True]
        else:
            model['test_noisy'] = [False]
    # Perform the dimensionality reduction.
    fvectors_test_reduced = reduce_dimensions(fvectors_test, model, "Test")
    return fvectors_test_reduced

def classify_page(page, model):
    """Classifier.

    The following function has two implementations of a classifier.

    First one is a nearest neighbor classifier. Distance being used is "cosine distance"
    and this implementation has shown to be quite efficient for clear test pages.

    Second implementation is K-Nearest neighbor using the cosine distance as well
    and the k is determined by square rooting the amount of samples and dividing by 2.
    This has shown to be quite more efficient with noisier pages.

    Both implementatioins make usage of the noise determine earlier on saved in
    the model while loading the test page. Said determine is removed from the model
    upon usage to symbolise that the page has been already classified.

    Only limitation is that pages need to be classified in the same order they've been loaded with.

    parameters:

    page - matrix, each row is a feature vector to be classified
    model - dictionary, stores the output of the training stage
    """
    # Obtain training data and labels
    fvectors_train = np.array(model['fvectors_train'])
    labels_train = np.array(model['labels_train'])
    # Calculate distance
    x = np.dot(page, fvectors_train.transpose())
    modtest = np.sqrt(np.sum(page * page, axis=1))
    modtrain = np.sqrt(np.sum(fvectors_train * fvectors_train, axis=1))
    dist = x / np.outer(modtest, modtrain.transpose()); # cosine distance
    # Check noise determine from the model
    m = np.array(model['test_noisy'])
    determine_noisy = m[0]
    new_m = np.delete(m, 0) # delete from the model
    model['test_noisy'] = new_m.tolist() # save new model
    if (determine_noisy): #Noisy page
        # Calculate K
        k = int((fvectors_train.shape[0] ** 0.5)/2)
        # Find k-nearest neighbors
        nearest_k = np.argsort(-dist, axis=1, kind='quicksort')[:, 0:k]
        # k-nearest neighbor
        labels = []
        for i in range(nearest_k.shape[0]):
            potential_labels = {}
            for j in range(nearest_k.shape[1]):
                label = labels_train[nearest_k[i, j]]
                if label in potential_labels:
                    potential_labels[label] += 1
                else:
                    potential_labels[label] = 1
            sort_list = sorted(potential_labels.items(), key=operator.itemgetter(1), reverse=True)
            labels.append(sort_list[0][0]) #The actual label
        return np.array(labels)
    else:
        # nearest-neighbor
        nearest = np.argmax(dist, axis=1)
        return labels_train[nearest]
