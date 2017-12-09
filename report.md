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
Given the labels the classifier did for the page, I first attempted to divide
the labels into words. Problems faced with this step were the fact that the
spaces between the words weren’t consistent. So, results weren't always accurate
representations of the words. After that, I attempted going through all words
and identifying potential matches from an English dictionary. If any potential
matches were found and the first match found was the same as the word, we assume
it's been correctly classified. If the word is not the same as the first
potential match, we go through all the potential matches and see if any of them
have only one letter difference with the word. This is because if we allow more
than 1 character, it's probably an unrelated match. Problems faced with the above
algorithm however were the fact that it slowed down the classification, and that
the words dividing wasn't returning the words correctly consistently.
Therefore, after seeing no major improvement but rather a very minor
improvement in the result, I decided not to run that step in the
classification of the test data.

## Performance
-Page 1: score = 97.4% correct
-Page 2: score = 96.4% correct
-Page 3: score = 86.2% correct
-Page 4: score = 60.4% correct
-Page 5: score = 60.9% correct
-Page 6: score = 50.7% correct

## Other information (Optional, Max 100 words)
In the training stage. After dividing the given train data into two parts, I
attempted two different noise techniques. First one was "Gaussian noise" and the second
was "Salt and Pepper noise". The second one has shown much better results in the
evaluation which led to me keeping it as the default noise-applying mechanism.
In addition, when I retrieve the test data, if after analysis I determine it's noisy,
I apply a median filter to the data which after many experiments has shown to have an
improvement on the results.
