"""Evaluate the classifier.
usage:
  python evaluate.py dev

DO NOT ALTER THIS FILE.

version: v1.0
"""

import os
import sys
import numpy as np

import utils.utils as utils
import system

NUM_TEST_PAGES = 6  # Number of pages in the test set
EXPECTED_DIMENSIONALITY = 10  # Expected feature vector dimensionality
MAX_MODEL_SIZE = 3145728  # Max size of model file in bytes

def validate_test_data(page_data_all_pages):
    """Check that test data has the correct dimensionality."""
    for page_data in page_data_all_pages:
        if page_data.shape[1] != EXPECTED_DIMENSIONALITY:
            return False
    return True


def evaluate(testset):
    """Run the classifer evaluation on a give testset."""

    # Check model file is compliant with the max size rule
    statinfo = os.stat('data/model.json.gz')
    if statinfo.st_size > MAX_MODEL_SIZE:
        print('Error: model.json.gz exceeds allowed size limit.')
        exit()

    # Load the results of the training process
    model = utils.load_jsongz('data/model.json.gz')

    # Construct a list of all the test set pages.
    page_names = ['data/{}/page.{}'.format(testset, page_num)
                  for page_num in range(1, NUM_TEST_PAGES+1)]

    # Load the correct labels for each test page
    true_labels = [utils.load_labels(page_name)
                   for page_name in page_names]

    # Load the 10-dimensional feature data for each test page
    page_data_all_pages = [system.load_test_page(page_name, model)
                           for page_name in page_names]

    # Check that load_test_page is returning 10-dimensional data
    if not validate_test_data(page_data_all_pages):
        print('Test data must be 10 dimensional')
        exit()

    # Run the classifier on each of the test pages
    output_labels = [system.classify_page(page_data, model)
                     for page_data in page_data_all_pages]

    # Compute the percentage correct classifications for each test page
    scores = [(100.0 * np.sum(output_label == true_label)) / output_label.shape[0]
              for output_label, true_label in zip(output_labels, true_labels)]

    # Print out the score for each test page.
    for i, score in enumerate(scores):
        print('Page {}: score = {:3.1f}% correct'.format(i+1, score))


if __name__ == '__main__':
    evaluate(testset=sys.argv[1])
