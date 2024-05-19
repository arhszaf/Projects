import numpy as np
import matplotlib.pyplot as plt

from collections import Counter

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Part A: Define a NaiveBayesClassifier class
class NaiveBayesClassifier:
    def __init__(self, m, n, k):
        # Initialize the hyperparameters
        self.m = m
        self.n = n
        self.k = k
        self.vocab = None
        self.class_probs = None
        self.word_probs = None

    def preprocess_data(self, texts, labels):
        # Convert word indices back to text
        texts = [' '.join([str(word) for word in doc]) for doc in texts]

        # Use CountVectorizer to convert text data into document-term matrix
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(texts)

        # Extract word counts and feature names
        word_counts = np.array(X.sum(axis=0))[0]
        words = np.array(vectorizer.get_feature_names_out())

        # Sort words based on frequency
        sorted_indices = np.argsort(word_counts)[::-1]

        # Select vocabulary excluding n most frequent and k most rare words
        self.vocab = words[sorted_indices[self.n:self.n + 3000]]

        # Filter data based on selected vocabulary
        X_filtered = X[:, np.isin(words, self.vocab)]

        return X_filtered.toarray(), labels

    def train(self, X, y):
        # Train the Naive Bayes classifier
        num_docs, num_words = X.shape
        self.class_probs = {c: np.sum(y == c) / num_docs for c in np.unique(y)}

        self.word_probs = {}
        for c in np.unique(y):
            # Laplace smoothing
            alpha = 0.1
            class_word_counts = np.sum(X[y == c], axis=0) + alpha
            total_class_words = np.sum(class_word_counts) + num_words
            self.word_probs[c] = class_word_counts / total_class_words

    def predict(self, X):
        # Make predictions using the trained model
        predictions = []
        for doc in X:
            log_probs = {c: np.log(self.class_probs[c]) + np.sum(np.log(self.word_probs[c][doc == 1])) if np.sum(doc == 1) > 0 and len(self.word_probs[c]) == len(doc) else np.log(self.class_probs[c]) for c in self.class_probs}
            predictions.append(max(log_probs, key=log_probs.get))
        return predictions

# Function to evaluate the performance of the model
def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.train(X_train, y_train)

    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    train_accuracy = accuracy_score(y_train, train_predictions)
    test_accuracy = accuracy_score(y_test, test_predictions)

    train_precision = precision_score(y_train, train_predictions, average='weighted')
    test_precision = precision_score(y_test, test_predictions, average='weighted')

    train_recall = recall_score(y_train, train_predictions, average='weighted')
    test_recall = recall_score(y_test, test_predictions, average='weighted')

    train_f1 = f1_score(y_train, train_predictions, average='weighted')
    test_f1 = f1_score(y_test, test_predictions, average='weighted')

    return {
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'train_precision': train_precision,
        'test_precision': test_precision,
        'train_recall': train_recall,
        'test_recall': test_recall,
        'train_f1': train_f1,
        'test_f1': test_f1,
    }

# Usage with IMDB dataset
from keras.datasets import imdb

# Load the dataset
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=10000)

# Convert the indices back to words
word_index = imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in X_train[0]])

# The hyperparameters 
m = 2000  # size of the vocabulary
n = 25    # number of most frequent words to skip
k = 10    # number of most rare words to skip

# Initialize the NaiveBayesClassifier model
model = NaiveBayesClassifier(m, n, k)

# Preprocess the data
X_train, y_train = model.preprocess_data(X_train, y_train)
X_test, y_test = model.preprocess_data(X_test, y_test)

# Evaluate the model
results = evaluate_model(model, X_train, y_train, X_test, y_test)

# Display the results
print("Training Accuracy:", results['train_accuracy'])
print("Test Accuracy:", results['test_accuracy'])
print("Training Precision:", results['train_precision'])
print("Test Precision:", results['test_precision'])
print("Training Recall:", results['train_recall'])
print("Test Recall:", results['test_recall'])
print("Training F1 Score:", results['train_f1'])
print("Test F1 Score:", results['test_f1'])

# Hyperparameter tuning
best_accuracy = 0
best_params = {}

# Dictionary to store learning curves
learning_curves = {'train_accuracy': [], 'test_accuracy': [], 'train_precision': [], 'test_precision': [],
                   'train_recall': [], 'test_recall': [], 'train_f1': [], 'test_f1': []}

# Iterate over different numbers of training examples
for num_examples in range(500, len(X_train), 500):
    X_train_subset, y_train_subset = X_train[:num_examples], y_train[:num_examples]

    model = NaiveBayesClassifier(m, n, k)
    results = evaluate_model(model, X_train_subset, y_train_subset, X_test, y_test)

    # Update learning curves
    learning_curves['train_accuracy'].append(results['train_accuracy'])
    learning_curves['test_accuracy'].append(results['test_accuracy'])
    learning_curves['train_precision'].append(results['train_precision'])
    learning_curves['test_precision'].append(results['test_precision'])
    learning_curves['train_recall'].append(results['train_recall'])
    learning_curves['test_recall'].append(results['test_recall'])
    learning_curves['train_f1'].append(results['train_f1'])
    learning_curves['test_f1'].append(results['test_f1'])

    # Update best hyperparameters if the current model performs better
    if results['test_accuracy'] > best_accuracy:
        best_accuracy = results['test_accuracy']
        best_params = {'m': m, 'n': n, 'k': k, 'num_examples': num_examples}

# Display the best hyperparameters and test accuracy
print("Best Hyperparameters:", best_params)
print("Best Test Accuracy:", best_accuracy)

# Plot learning curves
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(range(500, len(X_train), 500), learning_curves['train_accuracy'], label='Training Accuracy')
plt.plot(range(500, len(X_train), 500), learning_curves['test_accuracy'], label='Test Accuracy')
plt.xlabel('Number of Training Examples')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(range(500, len(X_train), 500), learning_curves['train_precision'], label='Training Precision')
plt.plot(range(500, len(X_train), 500), learning_curves['test_precision'], label='Test Precision')
plt.xlabel('Number of Training Examples')
plt.ylabel('Precision')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(range(500, len(X_train), 500), learning_curves['train_recall'], label='Training Recall')
plt.plot(range(500, len(X_train), 500), learning_curves['test_recall'], label='Test Recall')
plt.xlabel('Number of Training Examples')
plt.ylabel('Recall')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(range(500, len(X_train), 500), learning_curves['train_f1'], label='Training F1')
plt.plot(range(500, len(X_train), 500), learning_curves['test_f1'], label='Test F1')
plt.xlabel('Number of Training Examples')
plt.ylabel('F1 Score')
plt.legend()

plt.tight_layout()
plt.show()

print(f"{'Training Set Size': <20} {'Training Accuracy': <20} {'Test Accuracy': <20}")
print('-' * 60)
for size, train_acc, test_acc in zip((range(500, len(X_train), 2000)),learning_curves['train_accuracy'],learning_curves['test_accuracy']):
    print(f"{size:<20} {train_acc:<20.4f} {test_acc:<20.4f}")

print(f"{'Training Set Size': <20} {'Training Precision': <20} {'Test Precision': <20}")
print('-' * 60)
for size, train_prec, test_prec in zip((range(500, len(X_train), 2000)),learning_curves['train_precision'],learning_curves['test_precision']):
    print(f"{size:<20} {train_prec:<20.4f} {test_prec:<20.4f}")

print(f"{'Training Set Size': <20} {'Training Recall': <20} {'Test Recall': <20}")
print('-' * 60)
for size, train_re, test_re in zip((range(500, len(X_train), 2000)),learning_curves['train_recall'],learning_curves['test_recall']):
    print(f"{size:<20} {train_re:<20.4f} {test_re:<20.4f}")

print(f"{'Training Set Size': <20} {'Training F1': <20} {'Test F1': <20}")
print('-' * 60)
for size, train_f1, test_f1 in zip((range(500, len(X_train), 2000)),learning_curves['train_f1'],learning_curves['test_f1']):
    print(f"{size:<20} {train_f1:<20.4f} {test_f1:<20.4f}")

# Part B: MultinimialNB from sklearn 

from sklearn.naive_bayes import MultinomialNB

# Use CountVectorizer with the model's vocabulary
vectorizer_sk = CountVectorizer(vocabulary=model.vocab)
X_train_sk = vectorizer_sk.fit_transform([' '.join([str(word) for word in doc]) for doc in X_train])
X_test_sk = vectorizer_sk.transform([' '.join([str(word) for word in doc]) for doc in X_test])

# Train Scikit-learn's Naive Bayes
nb_sk = MultinomialNB()
nb_sk.fit(X_train_sk, y_train)

# Predict and evaluate
train_predictions_sk = nb_sk.predict(X_train_sk)
test_predictions_sk = nb_sk.predict(X_test_sk)

train_accuracy_sk = accuracy_score(y_train, train_predictions_sk)
test_accuracy_sk = accuracy_score(y_test, test_predictions_sk)

train_precision_sk = precision_score(y_train, train_predictions_sk, average='weighted')
test_precision_sk = precision_score(y_test, test_predictions_sk, average='weighted')

train_recall_sk = recall_score(y_train, train_predictions_sk, average='weighted')
test_recall_sk = recall_score(y_test, test_predictions_sk, average='weighted')

train_f1_sk = f1_score(y_train, train_predictions_sk, average='weighted')
test_f1_sk = f1_score(y_test, test_predictions_sk, average='weighted')

# Display the results for Scikit-learn's Naive Bayes
print("Scikit-learn Naive Bayes - Training Accuracy:", train_accuracy_sk)
print("Scikit-learn Naive Bayes - Test Accuracy:", test_accuracy_sk)
print("Scikit-learn Naive Bayes - Training Precision:", train_precision_sk)
print("Scikit-learn Naive Bayes - Test Precision:", test_precision_sk)
print("Scikit-learn Naive Bayes - Training Recall:", train_recall_sk)
print("Scikit-learn Naive Bayes - Test Recall:", test_recall_sk)
print("Scikit-learn Naive Bayes - Training F1 Score:", train_f1_sk)
print("Scikit-learn Naive Bayes - Test F1 Score:", test_f1_sk)

# Plot learning curves for Scikit-learn's Naive Bayes
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(range(500, len(X_train_sk.toarray()), 500), learning_curves['train_accuracy'], label='Your Naive Bayes - Training Accuracy')
plt.plot(range(500, len(X_train_sk.toarray()), 500), learning_curves['test_accuracy'], label='Your Naive Bayes - Test Accuracy')
plt.plot(range(500, len(X_train_sk.toarray()), 500), [train_accuracy_sk] * len(learning_curves['train_accuracy']), linestyle='--', label='Scikit-learn Naive Bayes - Training Accuracy')
plt.plot(range(500, len(X_train_sk.toarray()), 500), [test_accuracy_sk] * len(learning_curves['test_accuracy']), linestyle='--', label='Scikit-learn Naive Bayes - Test Accuracy')
plt.xlabel('Number of Training Examples')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(range(500, len(X_train_sk.toarray()), 500), learning_curves['train_precision'], label='Your Naive Bayes - Training Precision')
plt.plot(range(500, len(X_train_sk.toarray()), 500), learning_curves['test_precision'], label='Your Naive Bayes - Test Precision')
plt.plot(range(500, len(X_train_sk.toarray()), 500), [train_precision_sk] * len(learning_curves['train_precision']), linestyle='--', label='Scikit-learn Naive Bayes - Training Precision')
plt.plot(range(500, len(X_train_sk.toarray()), 500), [test_precision_sk] * len(learning_curves['test_precision']), linestyle='--', label='Scikit-learn Naive Bayes - Test Precision')
plt.xlabel('Number of Training Examples')
plt.ylabel('Precision')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(range(500, len(X_train_sk.toarray()), 500), learning_curves['train_recall'], label='Your Naive Bayes - Training Recall')
plt.plot(range(500, len(X_train_sk.toarray()), 500), learning_curves['test_recall'], label='Your Naive Bayes - Test Recall')
plt.plot(range(500, len(X_train_sk.toarray()), 500), [train_recall_sk] * len(learning_curves['train_recall']), linestyle='--', label='Scikit-learn Naive Bayes - Training Recall')
plt.plot(range(500, len(X_train_sk.toarray()), 500), [test_recall_sk] * len(learning_curves['test_recall']), linestyle='--', label='Scikit-learn Naive Bayes - Test Recall')
plt.xlabel('Number of Training Examples')
plt.ylabel('Recall')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(range(500, len(X_train_sk.toarray()), 500), learning_curves['train_f1'], label='Your Naive Bayes - Training F1')
plt.plot(range(500, len(X_train_sk.toarray()), 500), learning_curves['test_f1'], label='Your Naive Bayes - Test F1')
plt.plot(range(500, len(X_train_sk.toarray()), 500), [train_f1_sk] * len(learning_curves['train_f1']), linestyle='--', label='Scikit-learn Naive Bayes - Training F1')
plt.plot(range(500, len(X_train_sk.toarray()), 500), [test_f1_sk] * len(learning_curves['test_f1']), linestyle='--', label='Scikit-learn Naive Bayes - Test F1')
plt.xlabel('Number of Training Examples')
plt.ylabel('F1 Score')
plt.legend()

plt.tight_layout()
plt.show()

print(f"{'Training Set Size': <20} {'Training Accuracy': <20} {'Test Accuracy': <20}")
print('-' * 60)
for size, train_acc, test_acc in zip((range(500, len(X_train), 2000)),[train_accuracy_sk] * len(learning_curves['train_accuracy']),[test_accuracy_sk] * len(learning_curves['test_accuracy'])):
    print(f"{size:<20} {train_acc:<20.4f} {test_acc:<20.4f}")

print(f"{'Training Set Size': <20} {'Training Precision': <20} {'Test Precision': <20}")
print('-' * 60)
for size, train_prec, test_prec in zip((range(500, len(X_train), 2000)),[train_precision_sk] * len(learning_curves['train_precision']),[test_precision_sk] * len(learning_curves['test_precision'])):
    print(f"{size:<20} {train_prec:<20.4f} {test_prec:<20.4f}")

print(f"{'Training Set Size': <20} {'Training Recall': <20} {'Test Recall': <20}")
print('-' * 60)
for size, train_re, test_re in zip((range(500, len(X_train), 2000)),[train_recall_sk] * len(learning_curves['train_recall']),[test_recall_sk] * len(learning_curves['test_recall'])):
    print(f"{size:<20} {train_re:<20.4f} {test_re:<20.4f}")

print(f"{'Training Set Size': <20} {'Training F1': <20} {'Test F1': <20}")
print('-' * 60)
for size, train_f1, test_f1 in zip((range(500, len(X_train), 2000)),[train_f1_sk] * len(learning_curves['train_f1']),[test_f1_sk] * len(learning_curves['test_f1'])):
    print(f"{size:<20} {train_f1:<20.4f} {test_f1:<20.4f}")
