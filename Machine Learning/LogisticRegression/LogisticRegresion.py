from keras.datasets import imdb
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt


def preprocess_data(texts, labels):
        # Convert word indices back to text
        m = 5000  # size of the vocabulary
        n = 25    # number of most frequent words to skip
        k = 10    # number of most rare words to skip
        texts = [' '.join([str(word) for word in doc]) for doc in texts]

        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(texts)

        word_counts = np.array(X.sum(axis=0))[0]
        words = np.array(vectorizer.get_feature_names_out())

        # Sort words based on frequency
        sorted_indices = np.argsort(word_counts)[::-1]

        # Select vocabulary excluding n most frequent and k most rare words
        vocab = words[sorted_indices[n:n + 3000]]

        # Filter data based on selected vocabulary
        X_filtered = X[:, np.isin(words,vocab)]

        return X_filtered.toarray(), labels
max_words =10000
maxlen=500
(x_train,y_train),(x_test,y_test) = imdb.load_data(num_words=1000)
word_index = imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in x_train[0]])

x_train, y_train = preprocess_data(x_train, y_train)
x_test, y_test = preprocess_data(x_test, y_test)
log = LogisticRegression()
log.fit(x_train, y_train)
train_sizes = [100, 500, 1000, 2500, 5000, 10000, 15000, 20000]
train_accuracies = []
test_accuracies = []
precisions = []
recalls = []
f1_scores = []

for size in train_sizes:
    x_subset_train = x_train[:size]
    y_subset_train = y_train[:size]

    log = LogisticRegression()
    log.fit(x_subset_train, y_subset_train)

    train_accuracy = accuracy_score(y_subset_train, log.predict(x_subset_train))
    test_accuracy = accuracy_score(y_test, log.predict(x_test))

    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, log.predict(x_test), average='binary')

    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)


# Plotting learning curves
plt.plot(train_sizes, train_accuracies, label='Training Accuracy')
plt.plot(train_sizes, test_accuracies, label='Test Accuracy')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.title('Learning Curves - Logistic Regression')
plt.legend()
plt.show()

#plotting f1,percision and recall
plt.plot(train_sizes, precisions, label='Precision')
plt.plot(train_sizes, recalls, label='Recall')
plt.plot(train_sizes, f1_scores, label='F1 Score')
plt.xscale('log')
plt.xlabel('Training Set Size')
plt.ylabel('Score')
plt.title('Precision, Recall, and F1 Score vs Training Set Size')
plt.legend()
plt.show()

