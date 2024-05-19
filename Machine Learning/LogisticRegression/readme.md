# Logistic Regression Implementation and Comparison
implementation of a Logistic Regression classifier from scratch and compares it
with the Logistic Regression implementation from Scikit-learn. We use the IMDb movie
review dataset to evaluate the performance of both models on text classification tasks. The goal is to 
predict whether a given movie review is positive or negative.

## Implementation Details
### Custom Logistic Regression
The custom Logistic Regression implementation involves the following steps:
1. **Data Preprocessing**:
  * Load the IMDb dataset.
  * Convert the reviews from sequences of word indices to text.
  * Vectorize the text data using CountVectorizer.
  * Filter the vocabulary by excluding the most frequent and most rare words.
2. **Model Training**:
  * Initialize the weights.
  * Compute the sigmoid function.
  * Update the weights using gradient descent.
  * Calculate the cost function to monitor the training process.
3. **Evaluation**:
    * Calculate accuracy, precision, recall, and F1 score.
### Scikit-learn Logistic Regression
The Scikit-learn Logistic Regression implementation involves:
1. **Data Preprocessing**:
  * Similar to the custom implementation, load and preprocess the IMDb dataset.
2. **Model Training**:
  * Use the `LogisticRegression` class from Scikit-learn to fit the model on the training data.
3.**Prediction and Evaluation**:
  * Use the trained model to make predictions on the test data.
  * Evaluate the model using accuracy, precision, recall, and F1 score.
    
## Results
The performance of both the custom implementation and Scikit-learn's implementation is evaluated on various training set sizes.
The following metrics are compared:
* **Training Accuracy**
* **Test Accuracy**
* **Precision**
* **Recall**
* **F1 Score**
The results are visualized using plots for better understanding.
## Learning Curves
The custom implementation and the Scikit-learn implementation show comparable performance,
with Scikit-learn's implementation being more efficient due to optimized and built-in functions.
