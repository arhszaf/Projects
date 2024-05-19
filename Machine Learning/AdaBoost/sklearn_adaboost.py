import time 
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from keras.datasets import imdb
from sklearn.tree import DecisionTreeClassifier
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from matplotlib.ticker import ScalarFormatter
from sklearn.metrics import precision_recall_fscore_support

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
start_time = time.time()
train_sizes = [500, 2500, 2500, 4500, 6500, 10500, 12500,14500,16500,18500,20500,22500,24500]
train_accuracies = []
test_accuracies = []
precisions = []
recalls = []
f1_scores = []

max_words =10000
maxlen=500
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)
word_index = imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in x_train[0]])
# print('training data',x_train,y_train)
# print('testing data',x_test,y_test)
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)
x_train, y_train = preprocess_data(x_train, y_train)
x_test, y_test = preprocess_data(x_test, y_test)
base_estimator =DecisionTreeClassifier(max_depth=1)
adaboost = AdaBoostClassifier(estimator=base_estimator,n_estimators=100,random_state=42)

# x_train, y_train = preprocess_data(x_train, y_train)
# x_test, y_test = preprocess_data(x_test, y_test)

for size in train_sizes:
    x_subset_train = x_train[:size] 
    y_subset_train = y_train[:size]

    adaboost.fit(x_subset_train, y_subset_train)
    
    train_accuracy = accuracy_score(y_subset_train, adaboost.predict(x_subset_train))
    test_accuracy = accuracy_score(y_test, adaboost.predict(x_test))

    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, adaboost.predict(x_test), average='binary')

    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)



plt.plot(train_sizes, train_accuracies, label='Training Accuracy')
plt.plot(train_sizes, test_accuracies, label='Test Accuracy')
plt.xscale('log')  # Use logarithmic scale for better visualization
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Training Set Size')
plt.legend()
# Customize x-axis ticks to display as regular integers
plt.gca().xaxis.set_major_formatter(ScalarFormatter())
plt.show()
print(f"{'Training Set Size': <20} {'Training Accuracy': <20} {'Test Accuracy': <20}")
print('-' * 60)
for size, train_acc, test_acc in zip(train_sizes, train_accuracies, test_accuracies):
    print(f"{size:<20} {train_acc:<20.4f} {test_acc:<20.4f}")

plt.plot(train_sizes, precisions, label='Precision')
plt.plot(train_sizes, recalls, label='Recall')
plt.plot(train_sizes, f1_scores, label='F1 Score')
plt.xscale('log')
plt.xlabel('Training Set Size')
plt.ylabel('Score')
plt.title('Precision, Recall, and F1 Score vs Training Set Size')
plt.legend()
plt.gca().xaxis.set_major_formatter(ScalarFormatter())
end_time = time.time()
plt.show()
print(f"{'Training Set Size': <20} {'Training Accuracy': <20} {'Test Accuracy': <20} {'Precision': <20} {'Recall': <20} {'F1 Score': <20}")
print('-' * 120)
for size, train_acc, test_acc, precision, recall, f1 in zip(train_sizes, train_accuracies, test_accuracies, precisions, recalls, f1_scores):
    print(f"{size:<20} {train_acc:<20.4f} {test_acc:<20.4f} {precision:<20.4f} {recall:<20.4f} {f1:<20.4f}")


elapsed_time = end_time - start_time
print(f"elapsed time: {elapsed_time} seconds")





