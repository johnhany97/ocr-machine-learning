# OCR assignment report

## Feature Extraction (Max 200 Words)
Feature extraction has been divided into two different modes.
First is "Training mode". In this mode, training data is given and is told where
we should separate as the first half consists of clean data while the second half
consists of noisy data. We then start computing PCA for both of these portions of
the data, then we reduce the principal components by calling another function that
does so by the use of divergences and forward sequential search. After computing
the best scoring principal components, the data is reduced and is sent back to be
saved in the model. Second is "Testing mode". In this mode, data given is just the
test data. We compute how noisy the data given is, using a threshold value figured
out from multiple experiments, and save in the model that the data was noisy or
clean (because we use that knowledge later on in the classifier) and we start
reducing the dimension by using the PCA relevant to the type of data (We use
the one we computed from the noisy train data or the clean train data based
on whether the testing data was considered to be noisy or clean)

## Classifier (Max 200 Words)
I have implemented two versions for the classifier. First is "nearest-neighbor"
and the second is "k-nearest-neighbor". Both classifiers currently use cosine
distance as their measure. I have, after multiple experiments, figured that the
kNN works better on noisy pages and NN works better on clean data. Therefore,
the classifier takes advantage of the previously calculated determine while
loading the pages to identify a noisy vs a clean page. I have experimented with
different distance measure (Euclidean distance) only to find that the scores were
much worse, hence the decision on not using it in the final implementation.
I faced a difficulty in choosing an appropriate K, so after multiple experiments,
I found that the best K was equal to half the square root of the number
of samples to be tested.

## Error Correction (Max 200 Words)
Results:
- 94.4%, 95.3%, 87.7%, 60.6%, 59.6%, 49.2%
Two variants have been attempted. First implementation (currently called version two in the code) was dependent on an English dictionary being imported into the model file while training. It also only changed the word if the amount of characters between the word and the suggestion was only 1. This is because allowing changes that involved more than just one character would have been probably changing the word into something completely different. Problems faced with this implementation was that it was extremely slow and its results were terribly poor. The second implementation (currently called version one in the code) works differently. It uses two libraries. NLTK is being used to check if a word is spelled correct. Autocorrect is used to correct this word. Results obtained from the second experiment are the ones mentioned above this paragraph. Error correction is not being used in the final submission. Even though the second implementation works within the time  limit, It's producing worse results (except for the minor improvement in pages 3 and 4).
Both implementations use a word-dividing function which made use of the bounding boxes and some threshold. Defining the thresholds that separate words from each other has shown to be quite problematic.

## Performance
- Page 1: score = 97.4% correct
- Page 2: score = 96.4% correct
- Page 3: score = 86.2% correct
- Page 4: score = 60.4% correct
- Page 5: score = 60.9% correct
- Page 6: score = 50.7% correct

## Other information (Optional, Max 100 words)
In the training stage. After dividing the given train data into two parts, I
attempted two different noise techniques. First one was "Gaussian noise" and the second
was "Salt and Pepper noise". The second one has shown much better results in the
evaluation which led to me keeping it as the default noise-applying mechanism.
In addition, when I retrieve the test data, if after analysis I determine it's noisy,
I apply a median filter to the data which after many experiments has shown to have an
improvement on the results.
