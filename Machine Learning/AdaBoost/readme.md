# AdaBoost Implementation and Evaluation
This project demonstrates the implementation of an AdaBoost classifier using a custom
implementation and compares it with the scikit-learn implementation on the IMDb movie reviews dataset.
he dataset is preprocessed to exclude the `N most frequent` and `K most rare` words to
form a vocabulary of `M words`. The performance is evaluated based on various metrics including 
training and test accuracy, precision, recall, and F1 score.
## Files
* **`AdaBoost.py`**: Contains the implementation of the custom AdaBoost class.
* **`main.py`**: The main script to load data, preprocess it, train the AdaBoost classifier, and plot the results.

## Usage
### Loading IMDb Data
The IMDb dataset is loaded using Keras:
```
from keras.datasets import imdb

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)
```
### Data Preprocessing
Data preprocessing includes converting word indices to text, vectorizing the text data, and filtering based on the selected vocabulary:
```
from keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import CountVectorizer

maxlen = 500
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)

# Custom preprocessing function
x_train, y_train = preprocess_data(x_train, y_train)
x_test, y_test = preprocess_data(x_test, y_test)
```
### Custom AdaBoost Implementation
The custom AdaBoost class is defined in `AdaBoost.py`:
```
class AdaBoost:
    def __init__(self, m, n, k, base_estimator=None):
        # Initialization code

    def fit(self, X, y, M=150):
        # Fitting code

    def preprocess_data(self, texts, labels):
        # Preprocessing code

    def predict(self, X):
        # Prediction code

# Helper functions
def compute_error(y, y_pred, w_i):
    # Error computation code

def compute_alpha(error):
    # Alpha computation code

def update_weights(w_i, alpha, y, y_pred):
    # Weight update code
```
### Training and Evaluation
The classifier is trained and evaluated using different training set sizes. Metrics such as accuracy, precision, recall, and F1 score are computed and plotted:
```
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Initialize the custom AdaBoost classifier
ada_boost = AdaBoost(m, n, k, base_estimator)

# Train and evaluate the classifier
train_sizes = [100, 500, 1000, 2500, 5000, 10000, 15000, 20000]
train_accuracies, test_accuracies, precisions, recalls, f1_scores = [], [], [], [], []

for size in train_sizes:
    x_subset_train = x_train[:size]
    y_subset_train = y_train[:size]
    
    ada_boost.fit(x_subset_train, y_subset_train, M=150)
    train_accuracy = accuracy_score(y_subset_train, ada_boost.predict(x_subset_train))
    test_accuracy = accuracy_score(y_test, ada_boost.predict(x_test))

    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)

    y_pred = ada_boost.predict(x_test)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)

# Plotting results
plt.plot(train_sizes, train_accuracies, label='Training Accuracy')
plt.plot(train_sizes, test_accuracies, label='Test Accuracy')
plt.xscale('log')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Training Set Size')
plt.legend()
plt.show()
```
### Scikit-Learn AdaBoost Comparison
A similar process is followed using the scikit-learn implementation of AdaBoost for comparison:
```
from sklearn.ensemble import AdaBoostClassifier

adaboost = AdaBoostClassifier(estimator=base_estimator, n_estimators=100, random_state=42)

for size in train_sizes:
    x_subset_train = x_train[:size]
    y_subset_train = y_train[:size]

    adaboost.fit(x_subset_train, y_subset_train)
    
    train_accuracy = accuracy_score(y_subset_train, adaboost.predict(x_subset_train))
    test_accuracy = accuracy_score(y_test, adaboost.predict(x_test))

    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)

    y_pred = adaboost.predict(x_test)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)

plt.plot(train_sizes, train_accuracies, label='Training Accuracy')
plt.plot(train_sizes, test_accuracies, label='Test Accuracy')
plt.xscale('log')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Training Set Size')
plt.legend()
plt.show()
```
## Results
The results are plotted to visualize the accuracy, precision, recall, and F1 score against the training set size for both the custom and scikit-learn AdaBoost implementations.
## Conclusion
This project provides a hands-on understanding of AdaBoost by implementing it from scratch and comparing it with a well-established library. The performance metrics and plots help in evaluating the efficiency and effectiveness of the boosting algorithm.
