# OCR assignment report

## Feature Extraction (Max 200 Words)
Given all the page's letters' features (train data) in a matrix where each
row is a sample, I compute first 50 PCA components for each sample, then
divergences are computed for each class with every other class on the data
after applying the PCA dimensionality reduction and then we pick the most
repeated features, sort them, then perform a sequential forward search to
pick the best 10 features (principal components). I have attempted using
correlation to pick the least correlated principal components, however, I
got better results using divergences between the principal components.
After picking the best, we store v in the dictionary to be later on used
on test data. This implementation has shown significant improvement from just taking
first 10 principal components.

## Classifier (Max 200 Words)
Classifier being used is nearest neighbor. It doesn't require a training stage
but however, makes use of the training data in the process. I have attempted to
implement k-nearest neighbor. However, I decided against using it in the final
implementation as the results obtained from that classifier were poor, I faced
difficulty in choosing an appropriate k that would result in good classification
and because it took much longer to compute compared the nearest neighbor classifier.
Distance being used as a metric in the nearest neighbor classifier is cosine
distance, which accounts for spread and shown better results compared to using
Euclidean distance in that classifier. Also, I do a pre-processing step to the
data before I run the classifier. I apply a median filter on the data to attempt
to denoise the data. This has shown a very minor decrease in correctness for the
first two pages but a much more significant improvement in the other 4 pages.

## Error Correction (Max 200 Words)
Given the labels the classifier did for the page, I first attempted to divide the
labels into words. Problems faced with this step were the fact that the spaces between
the words weren’t consistent. So, results weren't always accurate representations
of the words. After that, I attempted going through all words and identifying potential
matches from an English dictionary. If any potential matches were found and the
first match found was the same as the word, we assume it's been correctly classified.
If the word is not the same as the first potential match, we go through all the
potential matches and see if any of them have only one letter difference with the
word. This is because if we allow more than 1 character, it's probably an unrelated match.
Problems faced with the above algorithm however were the fact that it slowed down
the classification, and that the words dividing wasn't returning the words correctly
consistently. Therefore, after seeing no major improvement but rather a very
minor improvement in the result, I decided not to run that step in the classification
of the test data.

## Performance
- Page 1: score = 97.0% correct
- Page 2: score = 97.4% correct
- Page 3: score = 82.1% correct
- Page 4: score = 56.5% correct
- Page 5: score = 38.6% correct
- Page 6: score = 28.8% correct

## Other information (Optional, Max 100 words)
[Optional: highlight any significant aspects of your system that are
NOT covered in the sections above]
