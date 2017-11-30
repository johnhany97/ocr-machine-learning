#!/bin/bash

# Prepare data for COM2004 book ORC assignment

echo 'Running assignment'

echo 'Training the model'
python train.py

echo 'Evaluate using both dev and eval data'
python evaluate.py dev
python evaluate.py eval