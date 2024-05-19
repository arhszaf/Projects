from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import time 
from sklearn.feature_extraction.text import CountVectorizer
from AdaBoost import AdaBoost
from imdb_data_loader import load_imdb_data
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import pad_sequences
from keras.datasets import imdb
from sklearn.metrics import precision_recall_fscore_support 
from matplotlib.ticker import ScalarFormatter


 

start_time = time.time()
train_sizes=[100,500,1000,2500,5000,10000,15000,20000]
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
k = 10    # number of most rare words to skip
base_estimator = DecisionTreeClassifier(max_depth=1)

# Train the classifier
ada_boost = AdaBoost(m, n, k, base_estimator )
x_train, y_train = ada_boost.preprocess_data(x_train, y_train)
x_test, y_test = ada_boost.preprocess_data(x_test, y_test)
ada_boost.fit(x_train, y_train, M=150)
for size in train_sizes:
    x_subset_train = x_train[:size] 
    y_subset_train = y_train[:size]
    #ada_boost.fit(x_train, y_train, M=150)
    
    train_accuracy = accuracy_score(y_subset_train, ada_boost.predict(x_subset_train))
    test_accuracy = accuracy_score(y_test, ada_boost.predict(x_test))

    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)

    y_pred = ada_boost.predict(x_test)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)

# # Get predictions on the test set
y_pred = ada_boost.predict(x_test)
print("Predictions:", y_pred)
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
end_time = time.time()
plt.show()

elapsed_time = end_time - start_time 
print(f"elapsed time: {elapsed_time} seconds")

