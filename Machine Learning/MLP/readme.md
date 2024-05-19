# Sentiment Analysis on IMDb Movie Reviews using Neural Networks
This project demonstrates the use of a neural network for sentiment analysis on the IMDb movie 
review dataset. The neural network is implemented using Keras and TensorFlow, and it is trained to 
classify movie reviews as positive or negative. The project includes varying the training set size to
analyze the model's performance.
## Implementation Details
### Neural Network Model
The neural network model is built using Keras and consists of the following layers:
* **Embedding Layer**: Converts word indices into dense vectors of fixed size.
* **Flatten Layer**: Flattens the input.
* **Dense Layer**: Fully connected layer with 32 units and ReLU activation, with L2 regularization to prevent overfitting.
* **Output Layer**: Fully connected layer with 1 unit and sigmoid activation for binary classification.
### Data Preprocessing
1. **Load IMDb Dataset**: Load the dataset using Keras's built-in function, considering only the top 10,000 words.
2. **Pad Sequences**: Pad the sequences to ensure that all reviews are of the same length (500 words).

### Training and Evaluation
1. **Training**: Train the model on subsets of the training data of varying sizes.
2. **Evaluation**: Evaluate the model's performance on the test data using accuracy, precision, recall, and F1 score.

## Results
The performance of the neural network is evaluated on various training set sizes.
The following metrics are compared:

1. **Training Accuracy**
2. **Test Accuracy**
3. **Precision**
4. **Recall**
5. **F1 Score**
6. **Training Loss**
7. **Validation Loss**
The results are visualized using plots for better understanding.
