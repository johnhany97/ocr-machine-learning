"""Load/save functions for supporting OCR assignment.

DO NOT ALTER THIS FILE.

version: v1.0
"""
import gzip
import json

import numpy as np
from PIL import Image


def load_char_images(page_name, char_images=None):
    """Load the image page pixel data."""
    if char_images is None:
        char_images = []
    im = np.array(Image.open(page_name + '.png'))
    height = im.shape[0]
    with open(page_name + '.bb.csv', 'r') as f:
        for line in f:
            data = line.split(',')
            x1 = int(data[0])
            y1 = height - int(data[3])
            x2 = int(data[2])
            y2 = height - int(data[1])
            char_images.append(im[y1:y2, x1:x2])
    return char_images


def load_labels(page_name, char_labels=None):
    """Load the image label data."""
    if char_labels is None:
        char_labels = []
    with open(page_name + '.label.txt', 'r') as f:
        for line in f:
            char_labels.append(line[0])
    return char_labels


def save_jsongz(filename, data):
    """Save a dictionary to a gzipped json file."""
    with gzip.GzipFile(filename, 'wb') as fp:
        json_str = json.dumps(data) + '\n'
        json_bytes = json_str.encode('utf-8')
        fp.write(json_bytes)


def load_jsongz(filename):
    """Load a gzipped json file."""
    with gzip.GzipFile(filename, 'r') as fp:
        json_bytes = fp.read()
        json_str = json_bytes.decode('utf-8')
        model = json.loads(json_str)
    return model
