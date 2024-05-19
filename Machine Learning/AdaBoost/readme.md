# AdaBoost Implementation and Evaluation
This project demonstrates the implementation of an AdaBoost classifier using a custom
implementation and compares it with the scikit-learn implementation on the IMDb movie reviews dataset.
he dataset is preprocessed to exclude the `N most frequent` and `K most rare` words to
form a vocabulary of `M words`. The performance is evaluated based on various metrics including 
training and test accuracy, precision, recall, and F1 score.
## Files
* **`AdaBoost.py`**: Contains the implementation of the custom AdaBoost class.
* **`main.py`**: The main script to load data, preprocess it, train the AdaBoost classifier, and plot the results.
