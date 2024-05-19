import numpy as np 
from numpy import log, dot, exp, shape
import matplotlib.pyplot as plt
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import CountVectorizer
import time 
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_fscore_support,accuracy_score, auc
from matplotlib.ticker import ScalarFormatter

def standardize(X):
    for i in range(shape(X)[1]):
        X[:, i] = (X[:, i] - np.mean(X[:, i])) / np.std(X[:, i])

def F1_score(y, y_hat):
    tp, tn, fp, fn = 0, 0, 0, 0
    for i in range(len(y)):
        if y[i] == 1 and y_hat[i] == 1:
            tp += 1
        elif y[i] == 1 and y_hat[i] == 0:
            fn += 1
        elif y[i] == 0 and y_hat[i] == 1:
            fp += 1
        elif y[i] == 0 and y_hat[i] == 0:
            tn += 1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * precision * recall / (precision + recall)
    return f1_score

class LogisticRegression:
    def __init__(self, m, n, k,):
        self.m = m
        self.n = n
        self.k = k
        self.vocab = None
        self.alphas = []
        self.G_M = []
        self.M = None
        self.training_errors = []
        self.prediction_errors = []
        
    def sigmoid(self, z):
        clipped_z = np.clip(z, -500, 500)  # Limit z to prevent overflow/underflow
        sig = 1 / (1 + np.exp(-clipped_z))
        return sig

    def initialize(self, X):
        weights = np.random.randn(X.shape[1] + 1, 1) * 0.01  # Initialize weights with small random values
        X = np.c_[np.ones((X.shape[0], 1)), X]
        return weights, X

    def fit(self, X, y, alpha=0.0001, iter=400):
        weights, X = self.initialize(X)

        def cost(theta):
            z = dot(X, theta)
            sig_z = self.sigmoid(z)
            epsilon = 1e-15  # Small constant to avoid division by zero
            cost0 = y.T.dot(np.log(np.maximum(sig_z, epsilon)))  # Add epsilon to prevent log(0)
            cost1 = (1 - y).T.dot(np.log(np.maximum(1 - sig_z, epsilon)))  # Add epsilon to prevent log(0)
            cost = -((cost1 + cost0)) / len(y)
            return cost

        cost_list = np.zeros(iter,)

        for i in range(iter):
            weights = weights - alpha * dot(X.T, self.sigmoid(dot(X, weights)) - np.reshape(y, (len(y), 1)))
            cost_list[i] = cost(weights)

        self.weights = weights
        return cost_list

    def predict(self, X):
        z = dot(self.initialize(X)[1], self.weights)
        lis = []
        for i in self.sigmoid(z):
            if i > 0.5:
                lis.append(1)
            else:
                lis.append(0)
        return lis
    def preprocess_data(self, texts, labels):
        # Convert word indices back to text
        texts = [' '.join([str(word) for word in doc]) for doc in texts]

        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(texts)

        word_counts = np.array(X.sum(axis=0))[0]
        words = np.array(vectorizer.get_feature_names_out())

        # Sort words based on frequency
        sorted_indices = np.argsort(word_counts)[::-1]

        # Select vocabulary excluding n most frequent and k most rare words
        self.vocab = words[sorted_indices[self.n:self.n + 3000]]

        # Filter data based on selected vocabulary
        X_filtered = X[:, np.isin(words, self.vocab)]

        return X_filtered.toarray(), labels
train_sizes=[100,500,1000,2500,5000,10000,15000,20000]
#train_sizes = [500, 2500, 2500, 4500, 6500, 10500, 12500,14500,16500,18500,20500,22500,24500]
train_accuracies = []
test_accuracies = []
precisions = []
recalls = []
f1_scores = []
# Load IMDb data
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)
# After loading the data
word_index = imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in x_train[0]])

# Preprocess the data
maxlen = 500  # Adjust maxlen as needed
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)
m = 2000  # size of the vocabulary
n = 25    # number of most frequent words to skip
k = 10 



# Train Logistic Regression model
logistic_regression = LogisticRegression(m,n,k)
x_train, y_train = logistic_regression.preprocess_data(x_train, y_train)
x_test, y_test = logistic_regression.preprocess_data(x_test, y_test)
#logistic_regression.fit(x_train, y_train)


for size in train_sizes:
    x_subset_train = x_train[:size] 
    y_subset_train = y_train[:size]
    logistic_regression.fit(x_train, y_train)
    
    train_accuracy = accuracy_score(y_subset_train, logistic_regression.predict(x_subset_train))
    test_accuracy = accuracy_score(y_test, logistic_regression.predict(x_test))

    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)

    y_pred = logistic_regression.predict(x_test)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)


y_pred = logistic_regression.predict(x_test)
# print("Predictions:", y_pred)
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
print(f"{'Training Set Size': <20} {'Training Accuracy': <20} {'Test Accuracy': <20} {'Precision': <20} {'Recall': <20} {'F1 Score': <20}")
print('-' * 120)
for size, train_acc, test_acc,recall,perc,f1 in zip(train_sizes, train_accuracies, test_accuracies,recalls,precisions,f1_scores):
    print(f"{size:<20} {train_acc:<20.4f} {test_acc:<20.4f} {recall:<20.4f} {perc:<20.4f} {f1:<20.4f}")

plt.plot(train_sizes, precisions, label='Precision')
plt.plot(train_sizes, recalls, label='Recall')
plt.plot(train_sizes, f1_scores, label='F1 Score')
plt.xscale('log')
plt.xlabel('Training Set Size')
plt.ylabel('Score')
plt.title('Precision, Recall, and F1 Score vs Training Set Size')
plt.legend()
plt.gca().xaxis.set_major_formatter(ScalarFormatter())
plt.show()

