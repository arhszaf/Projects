import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding,Flatten,Dense 
from keras.utils import plot_model
from IPython.display import Image
from keras.datasets import imdb
from sklearn.metrics import precision_recall_curve, accuracy_score,precision_recall_fscore_support
import matplotlib.pyplot as plt 
from keras.callbacks import EarlyStopping
from keras.regularizers import l2

# Load IMDB dataset
max_words = 10000  # Consider only the top 10,000 words
maxlen = 500  # Limit the length of each review to 500 words
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_words)
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)
# Vary the training set size and collect accuracy values
train_sizes = [500, 2500, 2500, 4500, 6500, 10500, 12500,14500,16500,18500,20500,22500,24500]
train_accuracies = []
test_accuracies = []
precisions = []
recalls = []
f1_scores = []
train_losses = []
val_losses = []


for size in train_sizes:
    x_subset_train = pad_sequences(x_train[:size], maxlen=maxlen)
    y_subset_train = y_train[:size]

    model = Sequential()
    model.add(Embedding(input_dim=max_words, output_dim=50, input_length=maxlen))
    model.add(Flatten())
    model.add(Dense(units=32, activation='relu', kernel_regularizer=l2(0.001)))  # L2 regularization added
    model.add(Dense(units=1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    history = model.fit(x=x_subset_train, y=y_subset_train, epochs=30, verbose=0, batch_size=32,
                        validation_split=0.2)
    train_accuracy = history.history['accuracy'][-1]
    val_accuracy = history.history['val_accuracy'][-1]
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    train_accuracy = history.history['accuracy'][-1]
    test_accuracy = model.evaluate(x_test, y_test, verbose=0)[1]

    y_pred = (model.predict(x_test) > 0.5).astype("int32")
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')

    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
#plots to visualize 
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(test_accuracies, label='Test Accuracy')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.title('Learning Curves')
plt.legend()
plt.show()
print(f"{'Training Set Size': <20} {'Training Accuracy': <20} {'Test Accuracy': <20}")
print('-' * 60)
for size, train_acc, test_acc in zip(train_sizes, train_accuracies, test_accuracies):
    print(f"{size:<20} {train_acc:<20.4f} {test_acc:<20.4f}")
plt.plot(precisions, label='Precision')
plt.plot(recalls, label='Recall')
plt.plot(f1_scores, label='F1 Score')
plt.xlabel('Training Set Size')
plt.ylabel('Metrics')
plt.title('Precision, Recall, and F1 Score')
plt.legend()
plt.show()
print(f"{'Training Set Size': <20} {'Training Accuracy': <20} {'Test Accuracy': <20} {'Precision': <20} {'Recall': <20} {'F1 Score': <20}")
print('-' * 120)
for size, train_acc, test_acc, precision, recall, f1 in zip(train_sizes, train_accuracies, test_accuracies, precisions, recalls, f1_scores):
    print(f"{size:<20} {train_acc:<20.4f} {test_acc:<20.4f} {precision:<20.4f} {recall:<20.4f} {f1:<20.4f}")

for i, size in enumerate(train_sizes):
    plt.plot(train_losses[i], label=f'Training Loss (Size={size})')
    plt.plot(val_losses[i], label=f'Validation Loss (Size={size})')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Curves')
plt.legend()
plt.show()




